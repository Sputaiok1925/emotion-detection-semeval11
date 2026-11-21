"""Paired bootstrap on the test set: is config A really different from config B?

Usage (from the repository root):
    python bootstrap_compare.py reports/finetune/ft4_processed_mean data/processed \
                                reports/finetune/ft4_extended_mean  data/extended

Both runs are compared on the SAME test rows (matched by id), at a fixed threshold of 0.5,
averaging the macro-F1 over the available seeds in every bootstrap resample.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

E = ["anger", "fear", "joy", "sadness", "surprise"]
B, THR = 1000, 0.5


def load(run_dir, data_dir):
    test = pd.read_csv(Path(data_dir) / "test.csv", dtype={"id": str}, keep_default_na=False)
    probas = [np.load(p) for p in sorted(Path(run_dir).glob("seed_*/test_proba.npy"))]
    if not probas:
        sys.exit(f"no seed_*/test_proba.npy in {run_dir}")
    return test["id"].tolist(), test[E].to_numpy().astype(int), probas


def main(run_a, data_a, run_b, data_b):
    ids_a, y_a, pa = load(run_a, data_a)
    ids_b, y_b, pb = load(run_b, data_b)
    common = [i for i in ids_a if i in set(ids_b)]
    pos_a = {i: k for k, i in enumerate(ids_a)}
    pos_b = {i: k for k, i in enumerate(ids_b)}
    ia = np.array([pos_a[i] for i in common])
    ib = np.array([pos_b[i] for i in common])
    y = y_a[ia]
    assert (y == y_b[ib]).all(), "labels of common ids differ between the two test sets"
    pred_a = [(p[ia] >= THR).astype(int) for p in pa]
    pred_b = [(p[ib] >= THR).astype(int) for p in pb]
    print(f"common test rows: {len(common)} (A has {len(ids_a)}, B has {len(ids_b)}); "
          f"seeds: A={len(pa)}, B={len(pb)}")

    def score(preds, idx):
        return np.mean([f1_score(y[idx], pr[idx], average="macro", zero_division=0) for pr in preds])

    full = np.arange(len(y))
    sa, sb = score(pred_a, full), score(pred_b, full)
    rng = np.random.RandomState(0)
    diffs = []
    for _ in range(B):
        idx = rng.randint(0, len(y), len(y))
        diffs.append(score(pred_a, idx) - score(pred_b, idx))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"A macro-F1@0.5 = {sa:.4f}   B macro-F1@0.5 = {sb:.4f}   A - B = {sa - sb:+.4f}")
    print(f"95% bootstrap interval of (A - B): [{lo:+.4f}, {hi:+.4f}]")
    print("=> difference is " + ("NOT clearly different from zero (interval contains 0)"
                                 if lo <= 0 <= hi else "statistically distinguishable from zero"))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
