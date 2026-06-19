"""The 32K TinyStories tokenizer (karpathy/llama2.c ``tokenizer.bin`` format).

The R2/R3 shards are pre-tokenized with the Llama2 32K SentencePiece BPE; this
reads the matching ``tokenizer.bin`` so we can **encode** prompts and **decode**
generated ids back to text for the qualitative coherence check (PRD User Story 5).
Pure Python, no dependency (the no-deps rule applies to the ANE side, but keeping
this dependency-free is in the same spirit and avoids a SentencePiece install).

Format (`assets/models/tokenizer.bin`): ``int32 max_token_length``, then for each
of 32000 ids ``float32 score, int32 len, <len> bytes``. The pieces already have
SentencePiece's ``▁`` replaced by a plain space; byte fallbacks are the literal
strings ``<0xXX>`` at ids ``byte + 3``; BOS=1, EOS=2 are control ids.
"""

from __future__ import annotations

import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER = REPO / "assets" / "models" / "tokenizer.bin"

BOS_ID = 1
EOS_ID = 2


class SpmBinTokenizer:
    """Llama2.c-format SentencePiece BPE tokenizer (encode + decode)."""

    def __init__(self, path: str | Path = DEFAULT_TOKENIZER):
        self.vocab: list[bytes] = []
        self.scores: list[float] = []
        data = Path(path).read_bytes()
        (self.max_token_length,) = struct.unpack_from("<i", data, 0)
        off = 4
        while off < len(data):
            (score,) = struct.unpack_from("<f", data, off); off += 4
            (ln,) = struct.unpack_from("<i", data, off); off += 4
            self.vocab.append(data[off:off + ln]); off += ln
            self.scores.append(score)
        self._id = {piece: i for i, piece in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # --- encode: greedy score-maximizing BPE merge (matches llama2.c encode) ---
    def encode(self, text: str, bos: bool = True, eos: bool = False) -> list[int]:
        ids: list[int] = []
        if text:
            sp = self._id.get(b" ")          # llama2.c "dummy prefix" space
            if sp is not None:
                ids.append(sp)
        for ch in text:
            b = ch.encode("utf-8")
            idx = self._id.get(b)
            if idx is not None:
                ids.append(idx)
            else:
                ids.extend(byte + 3 for byte in b)   # byte fallback -> <0xXX>

        while True:
            best_score, best_id, best_at = -1e30, None, None
            for i in range(len(ids) - 1):
                merged = self.vocab[ids[i]] + self.vocab[ids[i + 1]]
                j = self._id.get(merged)
                if j is not None and self.scores[j] > best_score:
                    best_score, best_id, best_at = self.scores[j], j, i
            if best_at is None:
                break
            ids[best_at] = best_id
            del ids[best_at + 1]

        return ([BOS_ID] if bos else []) + ids + ([EOS_ID] if eos else [])

    # --- decode: concatenate piece bytes, expand <0xXX>, drop control ids ---
    def decode(self, ids) -> str:
        out = bytearray()
        for tok in ids:
            tok = int(tok)
            if tok in (BOS_ID, EOS_ID):
                continue
            piece = self.vocab[tok]
            if len(piece) == 6 and piece.startswith(b"<0x") and piece.endswith(b">"):
                try:
                    out.append(int(piece[3:5], 16)); continue
                except ValueError:
                    pass
            out += piece
        return out.decode("utf-8", errors="replace").lstrip(" ")
