"""Fine-tune base RoBERTa on BRIGHTER using Artem's encoder and evaluation.py."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation import EMOTIONS, evaluate
from frozen_bert.finetune import (
    EmotionBert, apply_freezing, set_mode, make_optimizer, encode_batch,
    load_split, predict_proba, tune_thresholds, set_seed,
    load_trainable_state,
)


def train_epoch(model, tokenizer, texts, labels, optimizer, scheduler, loss_fn,
                args, device, scaler, amp_dtype, rng):
    set_mode(model, True)
    order = rng.permutation(len(texts))
    effective_batch = args.batch_size * args.accumulation
    total_loss = 0.0
    for start in range(0, len(order), effective_batch):
        group = order[start:start + effective_batch]
        optimizer.zero_grad(set_to_none=True)
        # Accumulate gradients of small batches, then make one optimizer update.
        for offset in range(0, len(group), args.batch_size):
            idx = group[offset:offset + args.batch_size]
            inputs = encode_batch(tokenizer, [texts[i] for i in idx], args.max_length, device)
            target = torch.from_numpy(labels[idx]).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                loss = loss_fn(model(**inputs), target)
            scaler.scale(loss * (len(idx) / len(group))).backward()
            total_loss += loss.item() * len(idx)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            scheduler.step()
        done = start + len(group)
        if (start // effective_batch + 1) % 50 == 0 or done == len(texts):
            print(f"  train {done}/{len(texts)} | loss {total_loss / done:.4f}", flush=True)
    return total_loss / len(texts)


def fit(model, tokenizer, train, dev, args, device, checkpoint, report_dir, start_epoch=0):
    _, texts, labels = train
    dev_ids, dev_texts, dev_labels = dev
    optimizer = make_optimizer(model, args.lr_encoder, args.lr_head)
    updates = math.ceil(len(texts) / (args.batch_size * args.accumulation)) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(updates * 0.1), updates)

    # Same class balancing as Artem: N_negative / N_positive on train only.
    positives = labels.sum(axis=0)
    weights = torch.tensor((len(labels) - positives) / np.maximum(positives, 1), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
    print("Positive class weights:", dict(zip(EMOTIONS, weights.cpu().tolist())))

    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    print("Precision:", str(amp_dtype) if device.type == "cuda" else "float32")
    rng = np.random.RandomState(args.seed + start_epoch)
    best_f1, history = -1.0, []

    def save_best(epoch, scores, f1):
        thresholds = tune_thresholds(dev_labels, scores)
        # Save all parameters, including any previously trained but now frozen layers.
        state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        saved = {"state": state, "model": args.model, "revision": args.revision,
                 "pooling": "mean", "dropout": 0.1, "max_length": args.max_length,
                 "thresholds": thresholds.tolist(), "best_epoch": epoch, "seed": args.seed,
                 "dev_macro_f1_at_0.5": f1,
                 "dev_macro_f1": evaluate(dev_labels, scores, thresholds, verbose=False)["macro_f1"]}
        if args.continue_from:
            saved["continued_from"] = str(args.continue_from)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(saved, temporary)
        temporary.replace(checkpoint)
        np.save(report_dir / "dev_proba.npy", scores)
        (report_dir / "dev_ids.json").write_text(json.dumps(dev_ids), encoding="utf-8")
        print("Saved best model:", checkpoint, flush=True)

    if args.continue_from:
        # The starting checkpoint is also a candidate; do not replace it by a worse epoch.
        scores = predict_proba(model, tokenizer, dev_texts, batch_size=args.batch_size,
                               max_length=args.max_length, device=device)
        best_f1 = evaluate(dev_labels, scores, 0.5, verbose=False)["macro_f1"]
        history.append({"epoch": start_epoch, "train_loss": None,
                        "dev_macro_f1_at_0.5": best_f1, "seconds": 0.0})
        print(f"Starting checkpoint: epoch {start_epoch}, dev Macro-F1 {best_f1:.4f}", flush=True)
        save_best(start_epoch, scores, best_f1)
        (report_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        started = time.perf_counter()
        loss = train_epoch(model, tokenizer, texts, labels, optimizer, scheduler, loss_fn,
                           args, device, scaler, amp_dtype, rng)
        scores = predict_proba(model, tokenizer, dev_texts, batch_size=args.batch_size,
                               max_length=args.max_length, device=device)
        # Keep Artem's epoch selection protocol, using the fixed 0.5 threshold.
        f1 = evaluate(dev_labels, scores, 0.5, verbose=False)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": loss, "dev_macro_f1_at_0.5": f1,
                        "seconds": time.perf_counter() - started})
        print(f"Epoch {epoch}/{start_epoch + args.epochs}: loss {loss:.4f}, dev Macro-F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            save_best(epoch, scores, f1)
        (report_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def evaluate_test(model, tokenizer, saved, data_dir, report_dir, batch_size, device):
    ids, texts, labels = load_split(data_dir / "test.csv")
    scores = predict_proba(model, tokenizer, texts, batch_size=batch_size,
                           max_length=saved["max_length"], device=device)
    thresholds = dict(zip(EMOTIONS, saved["thresholds"]))
    metrics = evaluate(labels, scores, thresholds, output_dir=report_dir)
    metrics.update(model=saved["model"], revision=saved["revision"], seed=saved["seed"],
                   best_epoch=saved["best_epoch"], thresholds=thresholds,
                   dev_macro_f1=saved["dev_macro_f1"], data_dir=str(data_dir))
    if "continued_from" in saved:
        metrics["continued_from"] = saved["continued_from"]
    np.save(report_dir / "test_proba.npy", scores)
    (report_dir / "test_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Test Macro-F1:", metrics["macro_f1"])
    print("Saved reports:", report_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="FacebookAI/roberta-base")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/extended")
    parser.add_argument("--unfreeze-layers", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr-encoder", type=float)
    parser.add_argument("--lr-head", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--continue-from", type=Path, help="Start a new fine-tuning stage from a saved model.pt")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.continue_from and args.eval_only:
        parser.error("--continue-from is a training option, not an evaluation option")
    if args.lr_encoder is None:
        args.lr_encoder = 1e-5 if args.continue_from else 2e-5
    if args.lr_head is None:
        args.lr_head = 2e-4 if args.continue_from else 1e-3
    if min(args.epochs, args.batch_size, args.accumulation) < 1 or not 1 <= args.max_length <= 512:
        parser.error("Positive epochs, batch size and accumulation; max-length 1..512 required")
    tag = f"roberta_{args.data_dir.name}_ft{args.unfreeze_layers}_seed{args.seed}"
    if args.continue_from:
        if not args.continue_from.is_file():
            parser.error(f"Source checkpoint not found: {args.continue_from}")
        tag = args.continue_from.parent.name + "_continued"
    run_dir = args.run_dir or ROOT / "runs" / tag
    report_dir = args.report_dir or ROOT / "reports/roberta_finetune" / tag
    checkpoint = run_dir / "model.pt"
    if args.eval_only and not checkpoint.exists():
        parser.error(f"Checkpoint not found: {checkpoint}")
    if not args.eval_only and checkpoint.exists():
        parser.error("Checkpoint already exists; use --eval-only or a new --run-dir and --report-dir")
    if not args.eval_only and (report_dir / "metrics.json").exists():
        parser.error("Report already exists; choose a separate --report-dir")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in the installed PyTorch")
    print("Device:", device, flush=True)
    if device.type == "cpu" and not args.eval_only:
        print("Full fine-tuning on CPU can be slow. CUDA is recommended.", flush=True)
    set_seed(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = EmotionBert(saved["model"], saved["revision"], "mean", saved["dropout"])
        tokenizer = AutoTokenizer.from_pretrained(saved["model"], revision=saved["revision"])
        load_trainable_state(model, saved["state"])
        model.to(device)
    else:
        train = load_split(args.data_dir / "train.csv")
        dev = load_split(args.data_dir / "dev.csv")
        if set(train[1]) & set(dev[1]):
            raise ValueError("Train and dev contain shared texts")
        start_epoch = 0
        source = None
        if args.continue_from:
            source = torch.load(args.continue_from, map_location="cpu", weights_only=True)
            if source["pooling"] != "mean" or source["dropout"] != 0.1:
                raise ValueError("Continuation requires a Mean Pooling model with dropout 0.1")
            args.model, args.revision, args.max_length = source["model"], source["revision"], source["max_length"]
            start_epoch = source["best_epoch"]
        (report_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
        print(f"Train: {len(train[0])}; dev: {len(dev[0])}; layers: {args.unfreeze_layers}", flush=True)
        model = EmotionBert(args.model, args.revision, "mean", 0.1).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        if source is not None:
            load_trainable_state(model, source["state"])
            del source
        apply_freezing(model, args.unfreeze_layers)
        fit(model, tokenizer, train, dev, args, device, checkpoint, report_dir, start_epoch)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        load_trainable_state(model, saved["state"])
    del saved["state"]
    if not args.train_only:
        evaluate_test(model, tokenizer, saved, args.data_dir, report_dir, args.batch_size, device)
    else:
        print("Training finished. Run with --eval-only to evaluate the best saved epoch.")


if __name__ == "__main__":
    main()
