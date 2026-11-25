"""Shared multi-label evaluation for any emotion classifier (BERT, RoBERTa, etc.)."""
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise"]


def evaluate(y_true, probabilities, thresholds=0.5, output_dir=None, verbose=True):
    """Arrays must have shape (N, 5), in EMOTIONS order and matching row order.

    Pass probabilities (after sigmoid) or binary predictions. Choose thresholds
    on dev, before evaluating test. A scalar, list or emotion dictionary is accepted.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if isinstance(thresholds, dict):
        thresholds = [thresholds[e] for e in EMOTIONS]
    thresholds = np.asarray(thresholds)
    if y_true.ndim != 2 or y_true.shape[1] != len(EMOTIONS) or not len(y_true) or y_true.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must have matching nonempty (N, 5) shapes")
    if not np.isin(y_true, [0, 1]).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("Expected binary labels and probabilities in [0, 1]")
    if thresholds.shape not in [(), (len(EMOTIONS),)] or not ((0 <= thresholds) & (thresholds <= 1)).all():
        raise ValueError("Expected one threshold or five thresholds in [0, 1]")

    predicted = (probabilities >= thresholds).astype(int)
    metrics = {
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, predicted, average="micro", zero_division=0)),
        "exact_match": float((y_true == predicted).all(axis=1).mean()),
        **dict(zip(EMOTIONS, f1_score(y_true, predicted, average=None, zero_division=0).tolist())),
    }
    if verbose:
        print(classification_report(y_true, predicted, target_names=EMOTIONS, digits=4, zero_division=0))
        print(f"Exact match: {metrics['exact_match']:.4f}")
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        report = classification_report(y_true, predicted, target_names=EMOTIONS, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(out / "classification_report.csv")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Test CSV in prediction order")
    parser.add_argument("--run", required=True, type=Path, help="Folder with test_proba.npy and metrics.json")
    parser.add_argument("--out", type=Path, help="Optional separate report folder")
    args = parser.parse_args()
    if args.out and args.out.resolve() == args.run.resolve():
        parser.error("Use a separate output folder to preserve the original run metrics")
    labels = pd.read_csv(args.data, keep_default_na=False)[EMOTIONS].to_numpy()
    scores = np.load(args.run / "test_proba.npy", allow_pickle=False)
    thresholds = json.loads((args.run / "metrics.json").read_text())["thresholds"]
    evaluate(labels, scores, thresholds, args.out)
