"""Fine-tune BERT for 5-label emotion detection, training only the top N layers (or all).

N = 0  -> BERT fully frozen, only the new linear head is trained (like the frozen baseline)
N = 4  -> the top 4 of 12 encoder layers + head are trained
N = 12 -> the whole network (embeddings, all layers) + head is trained

Examples (run from the repository root):
    python frozen_bert/finetune.py --data-dir data/processed --unfreeze-layers 4
    python frozen_bert/finetune.py --data-dir data/extended  --unfreeze-layers 12 --seeds 42,43,44

Protocol (same rules as the rest of the project):
  * train.csv is used for learning;
  * dev.csv chooses the best epoch and the per-emotion thresholds;
  * test.csv is touched ONLY with --eval-test, once, for the final chosen configuration.

Outputs go to <out-dir>/<tag>/seed_<seed>/ (metrics.json, thresholds.json, dev/test probabilities)
and <out-dir>/<tag>/summary.json (mean and std over seeds).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise"]   # fixed label order of the project
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(path: Path, limit: Optional[int] = None):
    df = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    missing = [c for c in ["id", "text", *EMOTIONS] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    if limit:
        df = df.iloc[:limit]   # for quick smoke tests only
    return df["id"].tolist(), df["text"].tolist(), df[EMOTIONS].to_numpy(dtype=np.float32)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class EmotionBert(nn.Module):
    """BERT + dropout + one linear layer with 5 outputs (logits, one per emotion)."""

    def __init__(self, model_name: str, revision: Optional[str] = None,
                 pooling: str = "mean", dropout: float = 0.1, n_labels: int = len(EMOTIONS)):
        super().__init__()
        if pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls'")
        self.bert = AutoModel.from_pretrained(model_name, revision=revision)
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.bert.config.hidden_size, n_labels)
        self.train_modules: List[nn.Module] = []   # parts of BERT that are in train mode

    def forward(self, input_ids, attention_mask, token_type_ids=None, **_):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids).last_hidden_state
        if self.pooling == "cls":
            vec = hidden[:, 0]
        else:  # mean over real tokens (padding masked out), same as the frozen features
            m = attention_mask.unsqueeze(-1).to(hidden.dtype)
            vec = (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(self.dropout(vec))


def apply_freezing(model: EmotionBert, n_unfreeze: int) -> None:
    """Freeze all of BERT, then unfreeze the top n_unfreeze encoder layers.
    n_unfreeze == number of layers -> unfreeze everything (embeddings too)."""
    layers = model.bert.encoder.layer
    n_total = len(layers)
    if not 0 <= n_unfreeze <= n_total:
        raise ValueError(f"--unfreeze-layers must be in 0..{n_total}, got {n_unfreeze}")
    for p in model.bert.parameters():
        p.requires_grad_(False)
    if n_unfreeze == n_total:
        for p in model.bert.parameters():
            p.requires_grad_(True)
        model.train_modules = [model.bert]
    else:
        top = list(layers)[n_total - n_unfreeze:]
        for layer in top:
            for p in layer.parameters():
                p.requires_grad_(True)
        model.train_modules = top


def set_mode(model: EmotionBert, train: bool) -> None:
    """Train mode (dropout on) only for the head and the unfrozen BERT parts.
    Frozen layers stay in eval mode, so they behave exactly like the frozen baseline."""
    model.eval()
    if train:
        model.dropout.train()
        for module in model.train_modules:
            module.train()


def trainable_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            params[n].copy_(v.to(params[n].device))


def make_optimizer(model: EmotionBert, lr_bert: float, lr_head: float) -> torch.optim.Optimizer:
    groups = [{"params": list(model.head.parameters()), "lr": lr_head, "weight_decay": 0.0}]
    bert_params = [p for p in model.bert.parameters() if p.requires_grad]
    if bert_params:
        groups.append({"params": bert_params, "lr": lr_bert, "weight_decay": 0.01})
    return torch.optim.AdamW(groups)


# --------------------------------------------------------------------------- #
# Training and prediction
# --------------------------------------------------------------------------- #
def encode_batch(tokenizer, texts, max_length, device):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def train_epoch(model, tokenizer, texts, y, optimizer, scheduler, loss_fn, *,
                batch_size, max_length, device, rng) -> float:
    set_mode(model, True)
    order = rng.permutation(len(texts))
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = 0.0
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = encode_batch(tokenizer, [texts[i] for i in idx], max_length, device)
        target = torch.from_numpy(y[idx]).to(device)
        loss = loss_fn(model(**enc), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(idx)
    return total / len(texts)


@torch.inference_mode()
def predict_proba(model, tokenizer, texts, *, batch_size, max_length, device) -> np.ndarray:
    set_mode(model, False)
    out = np.zeros((len(texts), len(EMOTIONS)), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])          # similar lengths together = less padding
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = encode_batch(tokenizer, [texts[i] for i in idx], max_length, device)
        out[idx] = torch.sigmoid(model(**enc)).float().cpu().numpy()
    return out


# --------------------------------------------------------------------------- #
# Metrics (same definitions as src/evaluate_model.py: sklearn, prediction = proba >= threshold)
# --------------------------------------------------------------------------- #
def predict_labels(proba: np.ndarray, thresholds) -> np.ndarray:
    return (proba >= np.asarray(thresholds)).astype(int)


def macro_f1(y: np.ndarray, proba: np.ndarray, thresholds) -> float:
    return float(f1_score(y.astype(int), predict_labels(proba, thresholds),
                          average="macro", zero_division=0))


def per_emotion_f1(y: np.ndarray, proba: np.ndarray, thresholds) -> Dict[str, float]:
    f = f1_score(y.astype(int), predict_labels(proba, thresholds), average=None, zero_division=0)
    return {e: float(v) for e, v in zip(EMOTIONS, f)}


def tune_thresholds(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Per-emotion threshold with the best F1 on the given data; ties -> closest to 0.5."""
    best = []
    grid = sorted(THRESHOLD_GRID, key=lambda t: abs(t - 0.5))
    for j in range(len(EMOTIONS)):
        scores = [f1_score(y[:, j].astype(int), (proba[:, j] >= t).astype(int), zero_division=0)
                  for t in grid]
        best.append(float(grid[int(np.argmax(scores))]))   # argmax keeps the first (closest to 0.5)
    return np.array(best)


# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_seed(args, seed: int, tokenizer, data, device, out_dir: Path) -> dict:
    set_seed(seed)
    tr_texts, tr_y = data["train"][1], data["train"][2]
    dev_texts, dev_y = data["dev"][1], data["dev"][2]

    model = EmotionBert(args.model, args.revision, args.pooling, args.dropout).to(device)
    apply_freezing(model, args.unfreeze_layers)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[seed {seed}] trainable parameters: {n_train:,} of {n_all:,}")

    optimizer = make_optimizer(model, args.lr_bert, args.lr_head)
    steps = int(np.ceil(len(tr_texts) / args.batch_size)) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    pos_weight = None
    if args.pos_weight == "balanced":     # rare emotions get a larger weight in the loss
        pos = tr_y.sum(0)
        pos_weight = torch.tensor((len(tr_y) - pos) / np.maximum(pos, 1), dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    rng = np.random.RandomState(seed)
    best_f1, best_epoch, best_state, history = -1.0, 0, None, []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, tokenizer, tr_texts, tr_y, optimizer, scheduler, loss_fn,
                           batch_size=args.batch_size, max_length=args.max_length,
                           device=device, rng=rng)
        dev_proba = predict_proba(model, tokenizer, dev_texts, batch_size=args.batch_size * 2,
                                  max_length=args.max_length, device=device)
        f1 = macro_f1(dev_y, dev_proba, 0.5)
        history.append({"epoch": epoch, "train_loss": loss, "dev_macro_f1_at_0.5": f1})
        print(f"[seed {seed}] epoch {epoch}/{args.epochs}  loss {loss:.4f}  dev macro-F1@0.5 {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_epoch, best_state = f1, epoch, trainable_state(model)
    train_seconds = time.perf_counter() - t0

    load_trainable_state(model, best_state)        # go back to the best epoch (chosen on dev)
    dev_proba = predict_proba(model, tokenizer, dev_texts, batch_size=args.batch_size * 2,
                              max_length=args.max_length, device=device)
    thresholds = tune_thresholds(dev_y, dev_proba)
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "trainable_parameters": n_train,
        "history": history,
        "dev_macro_f1_at_0.5": macro_f1(dev_y, dev_proba, 0.5),
        # NOTE: thresholds are tuned on this same dev set, so this number is optimistic
        "dev_macro_f1_tuned": macro_f1(dev_y, dev_proba, thresholds),
        "dev_per_emotion_f1_tuned": per_emotion_f1(dev_y, dev_proba, thresholds),
        "thresholds": dict(zip(EMOTIONS, thresholds.tolist())),
    }

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    np.save(seed_dir / "dev_proba.npy", dev_proba)
    if args.eval_test:
        te_ids, te_texts, te_y = data["test"]
        te_proba = predict_proba(model, tokenizer, te_texts, batch_size=args.batch_size * 2,
                                 max_length=args.max_length, device=device)
        np.save(seed_dir / "test_proba.npy", te_proba)
        result["test_macro_f1_at_0.5"] = macro_f1(te_y, te_proba, 0.5)
        result["test_macro_f1_tuned_on_dev"] = macro_f1(te_y, te_proba, thresholds)
        result["test_micro_f1_tuned_on_dev"] = float(f1_score(
            te_y.astype(int), predict_labels(te_proba, thresholds), average="micro", zero_division=0))
        result["test_per_emotion_f1_tuned_on_dev"] = per_emotion_f1(te_y, te_proba, thresholds)
        print(f"[seed {seed}] TEST macro-F1 (thresholds from dev): "
              f"{result['test_macro_f1_tuned_on_dev']:.4f}")
    (seed_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--unfreeze-layers", type=int, default=4, help="top N encoder layers to train (0 = none, 12 = all)")
    ap.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--lr-bert", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pos-weight", choices=["none", "balanced"], default="balanced")
    ap.add_argument("--seeds", default="42", help="comma-separated, e.g. 42,43,44")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-test", action="store_true", help="also evaluate on test (use once, at the end)")
    ap.add_argument("--limit-train", type=int, default=None, help="use only the first N train rows (smoke test)")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    data_dir = Path(args.data_dir)
    data = {"train": load_split(data_dir / "train.csv", args.limit_train),
            "dev": load_split(data_dir / "dev.csv")}
    if args.eval_test:
        data["test"] = load_split(data_dir / "test.csv")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    tag = args.tag or f"ft{args.unfreeze_layers}_{data_dir.name}_{args.pooling}"
    out_dir = Path(args.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | data: {data_dir} (train {len(data['train'][0])}, dev {len(data['dev'][0])}) "
          f"| unfreeze top {args.unfreeze_layers} layers | seeds {seeds}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    results = [run_seed(args, s, tokenizer, data, device, out_dir) for s in seeds]

    def agg(key):
        vals = [r[key] for r in results if key in r]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None

    summary = {
        "args": vars(args),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "data_sha256": {s: sha256_file(data_dir / f"{s}.csv") for s in data
                        if (data_dir / f"{s}.csv").exists()},
        "n_seeds": len(seeds),
        "dev_macro_f1_at_0.5": agg("dev_macro_f1_at_0.5"),
        "dev_macro_f1_tuned": agg("dev_macro_f1_tuned"),
        "test_macro_f1_tuned_on_dev": agg("test_macro_f1_tuned_on_dev"),
        "runs": [{k: v for k, v in r.items() if k != "history"} for r in results],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSummary over seeds:")
    for k in ("dev_macro_f1_at_0.5", "dev_macro_f1_tuned", "test_macro_f1_tuned_on_dev"):
        if summary[k]:
            print(f"  {k}: {summary[k]['mean']:.4f} +- {summary[k]['std']:.4f}")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
