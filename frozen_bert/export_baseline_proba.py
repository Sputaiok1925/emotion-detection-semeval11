"""Export test probabilities of participant 3's logistic regression in the same folder
format as finetune.py runs, so it can be compared with bootstrap_compare.py.

Usage (from the repository root):
    python frozen_bert/export_baseline_proba.py models/logreg_mean.joblib \
        features/bert-base-uncased_len128/mean reports/finetune/baseline_logreg_mean

Result: <out_dir>/seed_0/test_proba.npy  (N x 5 probabilities, row order = test_ids.json).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np


def main(model_path, features_dir, out_dir):
    d = joblib.load(model_path)
    assert d["emotions"] == ["anger", "fear", "joy", "sadness", "surprise"], "emotion order mismatch"
    X = np.load(Path(features_dir) / "test_X.npy")
    ids = json.load(open(Path(features_dir) / "test_ids.json"))
    Xs = d["scaler"].transform(X)                       # scaler was fitted on train only
    proba = np.column_stack([m.predict_proba(Xs)[:, 1] for m in d["models"]]).astype("float32")
    assert proba.shape == (len(ids), 5)
    out = Path(out_dir) / "seed_0"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "test_proba.npy", proba)
    print(f"saved {out / 'test_proba.npy'} {proba.shape}; model C={d['C']}, "
          f"class_weight={d['class_weight']}, thresholds={d['thresholds']}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
