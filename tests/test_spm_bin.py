"""The 32K llama2.c SentencePiece tokenizer (R2/R3 generation tooling).

Pins the BPE encode (greedy score-merge) and decode against the committed
``assets/models/tokenizer.bin`` — including the exact id sequence that the
pre-tokenized TinyStories shards begin with, so the encoder provably matches the
tokenization the model was trained on.
"""

import numpy as np
import pytest

from lilbro.eval.spm_bin import DEFAULT_TOKENIZER, BOS_ID, EOS_ID, SpmBinTokenizer


@pytest.fixture(scope="module")
def tok():
    if not DEFAULT_TOKENIZER.exists():
        pytest.skip(f"tokenizer asset missing: {DEFAULT_TOKENIZER}")
    return SpmBinTokenizer()


def test_vocab_size(tok):
    assert tok.vocab_size == 32000


def test_matches_training_tokenization(tok):
    # data01 (and data00) begin with BOS + "Once upon a time" -> these exact ids.
    # Verified against the real shard; pins the BPE merge order.
    assert tok.encode("Once upon a time", bos=True) == [1, 9038, 2501, 263, 931]


@pytest.mark.parametrize("text", [
    "Once upon a time",
    "The little girl was happy.",
    "Tom and his dog ran to the park!",
    "She said, \"Hello!\"",
])
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text, bos=True)) == text


def test_bos_eos(tok):
    ids = tok.encode("hello", bos=True, eos=True)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    # decode drops the control tokens.
    assert "<s>" not in tok.decode(ids) and "</s>" not in tok.decode(ids)
    assert tok.encode("hello", bos=False)[0] != BOS_ID


def test_byte_fallback_round_trips(tok):
    # An emoji has no dedicated piece -> byte fallback (<0xXX> ids), and decode
    # must reassemble the multi-byte UTF-8 codepoint.
    s = "a 🦊 b"
    assert tok.decode(tok.encode(s, bos=False)) == s


def test_ids_in_range(tok):
    ids = tok.encode("The quick brown fox.", bos=True)
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert np.asarray(ids).dtype.kind in "iu"
