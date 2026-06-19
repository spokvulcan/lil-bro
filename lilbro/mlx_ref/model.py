"""MLX twin — the correctness oracle (fp32) and GPU energy baseline.

An independent autograd implementation of the dense + MTP model in MLX, running
on the Metal GPU. Architecture mirrors the upstream ANE model (see
``lilbro/configs``) and matches the torch oracle in ``twin_torch.py`` exactly, so
agreement between the two is a real correctness signal.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from lilbro.configs import Config
from .params import MTP_LAMBDA


def _lin(x, w):
    # w is [out, in]; apply x @ w.T  (no bias).
    return x @ mx.transpose(w)


def _rmsnorm(x, w, eps):
    ms = mx.mean(x * x, axis=-1, keepdims=True)
    return x * mx.rsqrt(ms + eps) * w


def _rope(x, theta):
    B, H, S, hd = x.shape
    half = hd // 2
    i = mx.arange(half).astype(mx.float32)
    freq = theta ** (-2.0 * i / hd)
    pos = mx.arange(S).astype(mx.float32)
    ang = pos[:, None] * freq[None, :]            # [S, half]
    cos = mx.cos(ang)[None, None]
    sin = mx.sin(ang)[None, None]
    xp = x.reshape(B, H, S, half, 2)
    x0, x1 = xp[..., 0], xp[..., 1]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return mx.stack([out0, out1], axis=-1).reshape(B, H, S, hd)


def _causal_mask(S):
    idx = mx.arange(S)
    return mx.where(idx[None, :] > idx[:, None], -mx.inf, 0.0).astype(mx.float32)


def _attention(q, k, v, cfg: Config):
    B, H, Sq, hd = q.shape
    if cfg.gqa_ratio > 1:
        # Interleaved grouping: q-head h -> kv-head h % kv_heads, tiled head order
        # [kv0..kvN, kv0..kvN, ...]. Matches the ANE forward kernel's
        # concat(interleave=false) over gqa_ratio copies of k_rope (mil_dynamic.h).
        k = mx.concatenate([k] * cfg.gqa_ratio, axis=1)
        v = mx.concatenate([v] * cfg.gqa_ratio, axis=1)
    scores = (q @ mx.swapaxes(k, -2, -1)) * (1.0 / math.sqrt(hd))
    scores = scores + _causal_mask(Sq)
    attn = mx.softmax(scores, axis=-1) @ v
    return mx.swapaxes(attn, 1, 2).reshape(B, Sq, H * hd)


def _block(x, p, prefix, cfg: Config):
    B, S, _ = x.shape
    h = _rmsnorm(x, p[f"{prefix}.rms_att"], cfg.norm_eps)
    q = _lin(h, p[f"{prefix}.wq"]).reshape(B, S, cfg.n_heads, cfg.head_dim)
    k = _lin(h, p[f"{prefix}.wk"]).reshape(B, S, cfg.kv_heads, cfg.head_dim)
    v = _lin(h, p[f"{prefix}.wv"]).reshape(B, S, cfg.kv_heads, cfg.head_dim)
    q = _rope(mx.swapaxes(q, 1, 2), cfg.rope_theta)
    k = _rope(mx.swapaxes(k, 1, 2), cfg.rope_theta)
    v = mx.swapaxes(v, 1, 2)
    a = _attention(q, k, v, cfg)
    o = _lin(a, p[f"{prefix}.wo"])
    x = x + cfg.res_alpha * o
    h2 = _rmsnorm(x, p[f"{prefix}.rms_ffn"], cfg.norm_eps)
    g = (h2 @ mx.transpose(p[f"{prefix}.w1"]))
    g = (g * mx.sigmoid(g)) * (h2 @ mx.transpose(p[f"{prefix}.w3"]))
    f = g @ mx.transpose(p[f"{prefix}.w2"])
    return x + cfg.res_alpha * f


def _head(h, p, cfg: Config):
    return _rmsnorm(h, p["rms_final"], cfg.norm_eps) @ mx.transpose(p["embed"])


def _ce(logits, targets, V):
    lse = mx.logsumexp(logits, axis=-1)
    gathered = mx.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return mx.mean(lse - gathered)


def forward_loss(p: dict, x, targets, cfg: Config):
    """p: name->mx.array; x/targets: int mx.array [B,S]. Returns scalar mx.array."""
    h = mx.take(p["embed"], x, axis=0)             # [B,S,D]
    for l in range(cfg.n_layers):
        h = _block(h, p, f"layer.{l}", cfg)
    trunk = h
    V = cfg.vocab
    loss = _ce(_head(trunk, p, cfg), targets, V)

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
            e = mx.take(p["embed"], emb_tok, axis=0)
            normed = mx.concatenate(
                [_rmsnorm(hp, p[f"mtp.{kk}.rms_h"], cfg.norm_eps),
                 _rmsnorm(e, p[f"mtp.{kk}.rms_e"], cfg.norm_eps)], axis=-1)
            hk_in = _lin(normed, p[f"mtp.{kk}.proj"])
            hk = _block(hk_in, p, f"mtp.{kk}", cfg)
            tgt_k = targets[:, kk:kk + Sk]
            mtp_losses.append(_ce(_head(hk, p, cfg), tgt_k, V))
            h_prev = hk
        if mtp_losses:
            loss = loss + MTP_LAMBDA * mx.mean(mx.stack(mtp_losses))
    return loss


def forward_logits(p: dict, x, cfg: Config):
    """Main-head logits [B, S, V] (no MTP). Used by the generation sampler."""
    h = mx.take(p["embed"], x, axis=0)
    for l in range(cfg.n_layers):
        h = _block(h, p, f"layer.{l}", cfg)
    return _head(h, p, cfg)


def loss_only(params_np: dict, x_np, targets_np, cfg: Config) -> float:
    """Forward-only loss (no backward) — for validation eval."""
    p = {n: mx.array(a.astype(np.float32)) for n, a in params_np.items()}
    loss = forward_loss(p, mx.array(x_np.astype(np.int32)),
                        mx.array(targets_np.astype(np.int32)), cfg)
    mx.eval(loss)
    return float(loss)


def loss_and_grads(params_np: dict, x_np, targets_np, cfg: Config):
    """Single forward+backward on the GPU. Returns (loss_float, {name: grad ndarray})."""
    p = {n: mx.array(a.astype(np.float32)) for n, a in params_np.items()}
    x = mx.array(x_np.astype(np.int32))
    targets = mx.array(targets_np.astype(np.int32))

    def loss_fn(pp):
        return forward_loss(pp, x, targets, cfg)

    loss, grads = mx.value_and_grad(loss_fn)(p)
    mx.eval(loss, grads)
    grads_np = {n: np.array(g, dtype=np.float64) for n, g in grads.items()}
    return float(loss), grads_np
