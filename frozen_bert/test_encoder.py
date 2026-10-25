"""Checks for the readiness criteria of participant 2.

Real model:            pytest -q test_encoder.py
Another model / path:  EMO_BERT_MODEL=/tmp/tiny_bert pytest -q test_encoder.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Make "import encoder" work whether this file sits next to encoder.py or in a subfolder
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent))
from encoder import EncoderConfig, FrozenBertEncoder  # noqa: E402

MODEL = os.environ.get("EMO_BERT_MODEL", "bert-base-uncased")
MAXLEN = int(os.environ.get("EMO_MAXLEN", "16"))   # small on purpose, so truncation is exercised

TEXTS = [
    "I am happy you came! What a surprise!",
    "Leave me alone.",
    "so sad",
    "",  # empty text must not crash
    "this is very very very very very very very very very very very very very much too long to fit",
]


@pytest.fixture(scope="module", params=["mean", "cls"])
def enc(request):
    # Every test runs once for mean pooling and once for CLS pooling
    return FrozenBertEncoder(EncoderConfig(model_name=MODEL, pooling=request.param,
                                           max_length=MAXLEN, batch_size=2, device="cpu"))


def test_shape_and_finite(enc):
    X = enc.encode(TEXTS)
    assert X.shape == (len(TEXTS), enc.dim)
    assert X.dtype == np.float32 and np.isfinite(X).all()


def test_single_equals_batch(enc):
    """A text encoded alone equals the same text inside a (padded) batch."""
    Xb = enc.encode(TEXTS)
    for i, t in enumerate(TEXTS):
        xs = enc.encode([t])[0]
        np.testing.assert_allclose(xs, Xb[i], atol=1e-4, rtol=1e-4)


def test_order_preserved(enc):
    """Internal sorting by length must not change the row order."""
    X = enc.encode(TEXTS)
    Xr = enc.encode(TEXTS[::-1])[::-1]
    np.testing.assert_allclose(X, Xr, atol=1e-4, rtol=1e-4)


def test_batch_size_invariance(enc):
    X1 = enc.encode(TEXTS)
    enc2 = FrozenBertEncoder(EncoderConfig(model_name=MODEL, pooling=enc.config.pooling,
                                           max_length=MAXLEN, batch_size=64, device="cpu"))
    np.testing.assert_allclose(X1, enc2.encode(TEXTS), atol=1e-4, rtol=1e-4)


def test_reload_is_deterministic(enc):
    """Reloading the model gives the same vectors."""
    enc2 = FrozenBertEncoder(enc.config)
    np.testing.assert_allclose(enc.encode(TEXTS), enc2.encode(TEXTS), atol=1e-6)


def test_model_is_frozen(enc):
    assert not enc.model.training
    assert all(not p.requires_grad for p in enc.model.parameters())
    before = [p.detach().clone() for p in enc.model.parameters()]
    enc.encode(TEXTS)
    assert all(torch.equal(a, b) for a, b in zip(before, enc.model.parameters()))


def test_truncation_stats(enc):
    enc.encode(TEXTS)
    s = enc.last_stats
    assert s["n_texts"] == len(TEXTS)
    assert s["n_truncated"] >= 1          # the last text is longer than MAXLEN
    assert 0 < s["truncated_frac"] <= 1
    assert s["seconds"] >= 0


def test_empty_list_and_bad_input(enc):
    assert enc.encode([]).shape == (0, enc.dim)
    with pytest.raises(TypeError):
        enc.encode("a single string")
    with pytest.raises(TypeError):
        enc.encode(["ok", 123])


def test_mean_differs_from_cls():
    e = FrozenBertEncoder(EncoderConfig(model_name=MODEL, max_length=MAXLEN, device="cpu"))
    both = e.encode_all(TEXTS)
    assert not np.allclose(both["mean"], both["cls"])


def test_exclude_special_tokens_changes_mean():
    a = FrozenBertEncoder(EncoderConfig(model_name=MODEL, max_length=MAXLEN, device="cpu"))
    b = FrozenBertEncoder(EncoderConfig(model_name=MODEL, max_length=MAXLEN, device="cpu",
                                        exclude_special_tokens=True))
    t = ["so sad"]
    assert not np.allclose(a.encode(t), b.encode(t))
