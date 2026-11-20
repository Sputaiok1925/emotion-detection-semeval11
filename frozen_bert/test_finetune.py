"""Checks for finetune.py (freezing logic, metrics, a real training step).

Real model:            pytest -q test_finetune.py
Another model / path:  EMO_BERT_MODEL=/tmp/tiny_bert pytest -q test_finetune.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from transformers import AutoTokenizer

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent))
import finetune as ft  # noqa: E402

MODEL = os.environ.get("EMO_BERT_MODEL", "bert-base-uncased")
TEXTS = ["i am happy you came", "leave me alone", "so sad", "what a surprise", "i am afraid"]
Y = np.array([[0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0]],
             dtype=np.float32)


def build(n_unfreeze):
    torch.manual_seed(0)
    model = ft.EmotionBert(MODEL, pooling="mean")
    ft.apply_freezing(model, n_unfreeze)
    return model


def n_layers():
    return len(ft.EmotionBert(MODEL).bert.encoder.layer)


def test_zero_layers_trains_only_head():
    model = build(0)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("head.") for n in trainable)


def test_top_layers_unfrozen():
    total = n_layers()
    model = build(1)
    layers = model.bert.encoder.layer
    assert all(p.requires_grad for p in layers[total - 1].parameters())
    assert all(not p.requires_grad for p in layers[0].parameters() if total > 1)
    assert all(not p.requires_grad for p in model.bert.embeddings.parameters())


def test_all_layers_unfrozen_includes_embeddings():
    model = build(n_layers())
    assert all(p.requires_grad for p in model.bert.embeddings.parameters())


def test_invalid_number_of_layers():
    with pytest.raises(ValueError):
        build(n_layers() + 1)


def test_modes_frozen_parts_stay_in_eval():
    model = build(1)
    ft.set_mode(model, True)
    layers = model.bert.encoder.layer
    assert layers[-1].training
    assert not model.bert.embeddings.training
    if len(layers) > 1:
        assert not layers[0].training
    ft.set_mode(model, False)
    assert not model.training and not layers[-1].training


def test_training_step_changes_only_unfrozen_parts():
    total = n_layers()
    model = build(1)
    tok = AutoTokenizer.from_pretrained(MODEL)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = ft.make_optimizer(model, lr_bert=1e-2, lr_head=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    ft.train_epoch(model, tok, TEXTS, Y, opt, sched, nn.BCEWithLogitsLoss(), batch_size=2,
                   max_length=16, device=torch.device("cpu"), rng=np.random.RandomState(0))
    after = dict(model.named_parameters())
    top = f"encoder.layer.{total - 1}."
    changed_top = any(not torch.equal(before[n], after[n]) for n in before if top in n)
    assert changed_top, "unfrozen layer did not change"
    assert any(not torch.equal(before[n], after[n]) for n in before if n.startswith("head."))
    for n in before:
        if n.startswith("bert.") and top not in n:   # embeddings, lower layers, pooler
            assert torch.equal(before[n], after[n]), f"frozen parameter changed: {n}"


def test_best_state_roundtrip():
    model = build(1)
    state = ft.trainable_state(model)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(1.0)
    ft.load_trainable_state(model, state)
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p.detach().cpu(), state[n])


def test_predict_proba_shape_and_range():
    model = build(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    p = ft.predict_proba(model, tok, TEXTS, batch_size=2, max_length=16, device=torch.device("cpu"))
    assert p.shape == (len(TEXTS), 5) and (p >= 0).all() and (p <= 1).all()
    p1 = ft.predict_proba(model, tok, TEXTS[:1], batch_size=2, max_length=16, device=torch.device("cpu"))
    np.testing.assert_allclose(p1[0], p[0], atol=1e-5)   # single text == same text in a batch


def test_threshold_tuning_and_metrics():
    rng = np.random.RandomState(0)
    y = (rng.rand(200, 5) > 0.5).astype(np.float32)
    proba = np.where(y == 1, 0.9, 0.1).astype(np.float32)       # perfectly separable
    thr = ft.tune_thresholds(y, proba)
    assert ft.macro_f1(y, proba, thr) == pytest.approx(1.0)
    assert ft.macro_f1(y, proba, 0.5) == pytest.approx(1.0)
    assert set(ft.per_emotion_f1(y, proba, thr)) == set(ft.EMOTIONS)
    # ties are resolved towards 0.5 (all thresholds between 0.1 and 0.9 give F1 = 1)
    assert np.all(thr == 0.5)
