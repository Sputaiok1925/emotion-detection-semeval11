"""Check the saved fine-tuning experiments or run a trained BERT checkpoint."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation import EMOTIONS, evaluate


def read_test(data_dir):
    return pd.read_csv(ROOT / data_dir / "test.csv", keep_default_na=False)


def metrics(actual, predicted):
    return evaluate(actual, predicted, verbose=False)


def verify_data(data_dir, hashes):
    for split, expected in hashes.items():
        raw = (ROOT / data_dir / f"{split}.csv").read_bytes()
        # Git may convert LF to CRLF on Windows.
        if expected not in (hashlib.sha256(raw).hexdigest(),
                            hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()):
            raise ValueError(f"Dataset differs: {data_dir}/{split}.csv")


def evaluate_saved():
    common = read_test("data/extended").set_index("id")
    original = read_test("data/processed").set_index("id")
    if not common.equals(original.loc[common.index]):
        raise ValueError("Common test texts or labels differ")
    rows = []
    for path in sorted((ROOT / "reports/finetune").glob("ft*/summary.json")):
        summary = json.loads(path.read_text())
        data_dir = summary["args"]["data_dir"]
        verify_data(data_dir, summary["data_sha256"])
        test = read_test(data_dir)
        dev = pd.read_csv(ROOT / data_dir / "dev.csv", keep_default_na=False)
        actual = test[EMOTIONS].to_numpy()
        mask = test["id"].isin(common.index)
        for seed_dir in sorted(path.parent.glob("seed_*")):
            run = json.loads((seed_dir / "metrics.json").read_text())
            proba = np.load(seed_dir / "test_proba.npy", allow_pickle=False)
            if proba.shape != actual.shape or not ((0 <= proba) & (proba <= 1)).all():
                raise ValueError(f"Invalid probabilities: {seed_dir}")
            thresholds = [run["thresholds"][e] for e in EMOTIONS]
            pred = (proba >= thresholds).astype(int)
            dev_proba = np.load(seed_dir / "dev_proba.npy", allow_pickle=False)
            dev_actual = dev[EMOTIONS].to_numpy()
            if dev_proba.shape != dev_actual.shape or not ((0 <= dev_proba) & (dev_proba <= 1)).all():
                raise ValueError(f"Invalid dev probabilities: {seed_dir}")
            dev_score = metrics(dev_actual, (dev_proba >= thresholds).astype(int))["macro_f1"]
            if not np.isclose(dev_score, run["dev_macro_f1_tuned"]):
                raise ValueError(f"Reported dev F1 differs: {seed_dir}")
            full = metrics(actual, pred)
            if not np.isclose(full["macro_f1"], run["test_macro_f1_tuned_on_dev"]):
                raise ValueError(f"Reported F1 differs: {seed_dir}")
            rows.append({"experiment": path.parent.name, "seed": run["seed"],
                         "dev_macro_f1": dev_score,
                         "full_test_macro_f1": full["macro_f1"],
                         **metrics(actual[mask], pred[mask])})

    from classifier import EmotionClassifier
    model = EmotionClassifier.load(ROOT / "models/logreg_mean.joblib")
    folder = ROOT / "features/bert-base-uncased_len128/mean"
    if json.loads((folder / "test_ids.json").read_text()) != list(original.index):
        raise ValueError("Baseline IDs do not match test")
    actual = original[EMOTIONS].to_numpy()
    pred = model.predict(np.load(folder / "test_X.npy", allow_pickle=False))
    dev = pd.read_csv(ROOT / "data/processed/dev.csv", keep_default_na=False)
    if json.loads((folder / "dev_ids.json").read_text()) != dev["id"].tolist():
        raise ValueError("Baseline IDs do not match dev")
    dev_score = metrics(dev[EMOTIONS].to_numpy(), model.predict(np.load(folder / "dev_X.npy", allow_pickle=False)))["macro_f1"]
    mask = original.index.isin(common.index)
    rows.append({"experiment": "frozen_bert_logreg", "seed": model.seed,
                 "dev_macro_f1": dev_score,
                 "full_test_macro_f1": metrics(actual, pred)["macro_f1"],
                 **metrics(actual[mask], pred[mask])})
    df = pd.DataFrame(rows)
    result = df.groupby("experiment").agg(
        macro_f1=("macro_f1", "mean"), macro_f1_std=("macro_f1", lambda x: x.std(ddof=0)),
        micro_f1=("micro_f1", "mean"), exact_match=("exact_match", "mean"),
        dev_macro_f1=("dev_macro_f1", "mean"), seeds=("seed", "count"))
    out = ROOT / "reports/evaluation_finetune"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_run.csv", index=False)
    result.to_csv(out / "comparison.csv")
    print(f"Comparison on {len(common)} identical English test examples:")
    print(result.to_string(float_format=lambda x: f"{x:.4f}"))
    print("Saved:", out)


def evaluate_checkpoint(checkpoint, data_dir, texts):
    import torch
    from transformers import AutoTokenizer
    from frozen_bert.finetune import EmotionBert, load_trainable_state, predict_proba

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionBert(saved["model"], saved["revision"], saved["pooling"], saved["dropout"])
    load_trainable_state(model, saved["state"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(saved["model"], revision=saved["revision"])
    df = None if texts else read_test(data_dir)
    inputs = texts or df["text"].tolist()
    scores = predict_proba(model, tokenizer, inputs, batch_size=32,
                           max_length=saved["max_length"], device=device)
    predicted = (scores >= np.asarray(saved["thresholds"])).astype(int)
    if texts:
        for text, values, labels in zip(inputs, scores, predicted):
            print("\nText:", text)
            print("Emotions:", ", ".join(e for e, label in zip(EMOTIONS, labels) if label) or "None")
            print("Scores:", ", ".join(f"{e}: {v:.3f}" for e, v in zip(EMOTIONS, values)))
    else:
        actual = df[EMOTIONS].to_numpy()
        print(json.dumps(evaluate(actual, scores, saved["thresholds"]), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--text", action="append")
    args = parser.parse_args()
    if args.text and not args.checkpoint:
        parser.error("--text requires --checkpoint")
    if args.checkpoint:
        evaluate_checkpoint(args.checkpoint, args.data_dir, args.text)
    else:
        evaluate_saved()


if __name__ == "__main__":
    main()
