"""Torch fp64 reference twin — the correctness oracle / tie-breaker.

A second, independent autograd implementation of the same dense + MTP model. Its
gradients are ground truth (computed in float64) for the R1 gradient diff. The
architecture mirrors the upstream ANE model exactly (see ``lilbro/configs``).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from lilbro.configs import Config
from .params import MTP_LAMBDA

_DT = torch.float64


def _rmsnorm(x, w, eps):
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(ms + eps) * w


def _rope(x, theta):
    # x: [B, H, S, hd], interleaved (GPT-J) pairs (2i, 2i+1).
    B, H, S, hd = x.shape
    half = hd // 2
    i = torch.arange(half, dtype=_DT, device=x.device)
    freq = theta ** (-2.0 * i / hd)              # [half]
    pos = torch.arange(S, dtype=_DT, device=x.device)
    ang = pos[:, None] * freq[None, :]           # [S, half]
    cos = torch.cos(ang)[None, None]             # [1,1,S,half]
    sin = torch.sin(ang)[None, None]
    xp = x.reshape(B, H, S, half, 2)
    x0, x1 = xp[..., 0], xp[..., 1]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return torch.stack((out0, out1), dim=-1).reshape(B, H, S, hd)


def _attention(q, k, v, cfg: Config):
    # q: [B, H, Sq, hd]; k, v: [B, KVH, Sk, hd]  (Sq == Sk here, self-attn).
    B, H, Sq, hd = q.shape
    if cfg.gqa_ratio > 1:
        # Interleaved grouping: q-head h -> kv-head h % kv_heads, i.e. tiled head
        # order [kv0..kvN, kv0..kvN, ...]. This is what the ANE forward kernel
        # computes — concat(interleave=false) over gqa_ratio copies of k_rope
        # (mil_dynamic.h) — so repeat (NOT repeat_interleave) is correct here.
        k = k.repeat(1, cfg.gqa_ratio, 1, 1)
        v = v.repeat(1, cfg.gqa_ratio, 1, 1)
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
    mask = torch.triu(torch.full((Sq, Sq), float("-inf"), dtype=_DT, device=q.device), 1)
    scores = scores + mask
    attn = F.softmax(scores, dim=-1) @ v           # [B,H,Sq,hd]
    return attn.transpose(1, 2).reshape(B, Sq, H * hd)


def _block(x, p, prefix, cfg: Config):
    B, S, _ = x.shape
    h = _rmsnorm(x, p[f"{prefix}.rms_att"], cfg.norm_eps)
    q = (h @ p[f"{prefix}.wq"].T).view(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
    k = (h @ p[f"{prefix}.wk"].T).view(B, S, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
    v = (h @ p[f"{prefix}.wv"].T).view(B, S, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
    q = _rope(q, cfg.rope_theta)
    k = _rope(k, cfg.rope_theta)
    a = _attention(q, k, v, cfg)
    o = a @ p[f"{prefix}.wo"].T
    x = x + cfg.res_alpha * o
    h2 = _rmsnorm(x, p[f"{prefix}.rms_ffn"], cfg.norm_eps)
    g = F.silu(h2 @ p[f"{prefix}.w1"].T) * (h2 @ p[f"{prefix}.w3"].T)
    f = g @ p[f"{prefix}.w2"].T
    return x + cfg.res_alpha * f


def _head(h, p, cfg: Config):
    return _rmsnorm(h, p["rms_final"], cfg.norm_eps) @ p["embed"].T


def forward_loss(p: dict, x, targets, cfg: Config):
    """p: name->tensor; x/targets: long [B,S]. Returns scalar loss tensor."""
    h = p["embed"][x]                              # [B,S,D]
    for l in range(cfg.n_layers):
        h = _block(h, p, f"layer.{l}", cfg)
    trunk = h
    V = cfg.vocab
    logits = _head(trunk, p, cfg)
    loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))

    if cfg.mtp_depth > 0:
        S = x.shape[1]
        h_prev = trunk
        mtp_losses = []
        for kk in range(1, cfg.mtp_depth + 1):
            Sk = S - kk
            if Sk <= 0:
                break
            hp = h_prev[:, :Sk, :]
            emb_tok = targets[:, kk - 1:kk - 1 + Sk]
            e = p["embed"][emb_tok]
            normed = torch.cat(
                [_rmsnorm(hp, p[f"mtp.{kk}.rms_h"], cfg.norm_eps),
                 _rmsnorm(e, p[f"mtp.{kk}.rms_e"], cfg.norm_eps)], dim=-1)
            hk_in = normed @ p[f"mtp.{kk}.proj"].T
            hk = _block(hk_in, p, f"mtp.{kk}", cfg)
            logits_k = _head(hk, p, cfg)
            tgt_k = targets[:, kk:kk + Sk]
            mtp_losses.append(F.cross_entropy(logits_k.reshape(-1, V), tgt_k.reshape(-1)))
            h_prev = hk
        if mtp_losses:
            loss = loss + MTP_LAMBDA * torch.stack(mtp_losses).mean()
    return loss


def loss_only(params_np: dict, x_np, targets_np, cfg: Config) -> float:
    """Forward-only loss (no backward), fp64 — for validation eval."""
    with torch.no_grad():
        p = {n: torch.tensor(a, dtype=_DT) for n, a in params_np.items()}
        loss = forward_loss(p, torch.tensor(x_np, dtype=torch.long),
                            torch.tensor(targets_np, dtype=torch.long), cfg)
    return float(loss)


def loss_and_grads(params_np: dict, x_np, targets_np, cfg: Config):
    """Single forward+backward. Returns (loss_float, {name: grad ndarray})."""
    p = {n: torch.tensor(a, dtype=_DT, requires_grad=True) for n, a in params_np.items()}
    x = torch.tensor(x_np, dtype=torch.long)
    targets = torch.tensor(targets_np, dtype=torch.long)
    loss = forward_loss(p, x, targets, cfg)
    loss.backward()
    grads = {n: t.grad.detach().numpy().copy() for n, t in p.items()}
    return float(loss.detach()), grads
