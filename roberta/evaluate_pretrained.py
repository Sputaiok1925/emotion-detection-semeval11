"""Evaluate GoEmotions RoBERTa on the five BRIGHTER English emotions."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation import EMOTIONS, evaluate

MODEL = "SamLowe/roberta-base-go_emotions"


def predict(model, tokenizer, texts, batch_size, max_length, device):
    import torch
    indices = [model.config.label2id[emotion] for emotion in EMOTIONS]
    probabilities = np.empty((len(texts), len(EMOTIONS)), dtype=np.float32)
    # Similar text lengths in a batch reduce padding; restore original row order.
    order = np.argsort([len(text) for text in texts])
    for start in range(0, len(texts), batch_size):
        rows = order[start:start + batch_size]
        inputs = tokenizer([texts[i] for i in rows], padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities[rows] = torch.sigmoid(logits[:, indices]).float().cpu().numpy()
        done = min(start + batch_size, len(texts))
        if done == len(texts) or start % (10 * batch_size) == 0:
            print(f"Predicted {done}/{len(texts)}", flush=True)
    return probabilities


def tune_thresholds(labels, probabilities):
    # GoEmotions often produces scores much smaller than 0.05 on BRIGHTER.
    low = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04]
    grid = sorted(low + np.round(np.arange(0.05, 0.96, 0.05), 2).tolist(), key=lambda t: abs(t - 0.5))
    return {emotion: float(max(grid, key=lambda t: f1_score(
        labels[:, i], probabilities[:, i] >= t, zero_division=0)))
        for i, emotion in enumerate(EMOTIONS)}


def save_results(dev, test, dev_scores, test_scores, out, metadata):
    thresholds = tune_thresholds(dev[EMOTIONS].to_numpy(), dev_scores)
    print("Dev thresholds:", thresholds)
    out.mkdir(parents=True, exist_ok=True)
    for name, df, scores in [("dev", dev, dev_scores), ("test", test, test_scores)]:
        np.save(out / f"{name}_proba.npy", scores)
        (out / f"{name}_ids.json").write_text(json.dumps(df["id"].tolist()), encoding="utf-8")
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    labels = test[EMOTIONS].to_numpy()
    print("\nTest — thresholds tuned on dev")
    metrics = evaluate(labels, test_scores, thresholds, out)
    metrics.update(metadata, thresholds=thresholds,
                   dev_macro_f1=evaluate(dev[EMOTIONS].to_numpy(), dev_scores,
                                         thresholds, verbose=False)["macro_f1"])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved:", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/extended")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/roberta_final")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--text", action="append", help="Predict a custom English text")
    args = parser.parse_args()
    if args.batch_size < 1 or not 1 <= args.max_length <= 512:
        parser.error("batch-size must be positive; max-length must be in 1..512")
    if args.text and not (args.out / "thresholds.json").exists():
        parser.error("Run evaluation first to obtain dev-tuned thresholds")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"Loading {MODEL} on {device} (downloaded automatically on first run)", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    revision = getattr(model.config, "_commit_hash", None)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision)

    if args.text:
        path = args.out / "thresholds.json"
        thresholds = json.loads(path.read_text())
        scores = predict(model, tokenizer, args.text, args.batch_size, args.max_length, device)
        for text, row in zip(args.text, scores):
            selected = [e for e, p in zip(EMOTIONS, row) if p >= thresholds[e]]
            print(f"\nText: {text}\nEmotions: {', '.join(selected) or 'None'}")
            print("Scores:", ", ".join(f"{e}: {p:.3f}" for e, p in zip(EMOTIONS, row)))
        return

    # Only dev is used to choose thresholds. Test is evaluated after that.
    dev = pd.read_csv(args.data_dir / "dev.csv", keep_default_na=False)
    test = pd.read_csv(args.data_dir / "test.csv", keep_default_na=False)
    dev_scores = predict(model, tokenizer, dev["text"].tolist(), args.batch_size, args.max_length, device)
    test_scores = predict(model, tokenizer, test["text"].tolist(), args.batch_size, args.max_length, device)
    save_results(dev, test, dev_scores, test_scores, args.out,
                 dict(model=MODEL, revision=revision, data_dir=str(args.data_dir),
                      max_length=args.max_length, grouping="direct"))


if __name__ == "__main__":
    main()
