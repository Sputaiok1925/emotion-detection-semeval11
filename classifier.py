"""Classifier on frozen BERT features (participant 3)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise"]  # fixed order
FEATURES_ROOT = Path("features/bert-base-uncased_len128")
DATA_DIR = Path("data/processed")


def load_split(split, pooling="mean"):
    """Return X (N,768), y (N,5) and ids; ids are checked against the CSV."""
    folder = FEATURES_ROOT / pooling
    X = np.load(folder / f"{split}_X.npy")
    with open(folder / f"{split}_ids.json", encoding="utf-8") as f:
        ids = json.load(f)
    df = pd.read_csv(DATA_DIR / f"{split}.csv", dtype={"id": str})
    if df["id"].tolist() != [str(i) for i in ids]:
        raise ValueError(f"{split}: ids in features and CSV differ")
    return X, df[EMOTIONS].to_numpy(), df["id"].tolist()


def evaluate(y_true, y_pred):
    """Macro-F1 and per-emotion F1 (replace with participant 1's function if needed)."""
    per = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        **{f"f1_{e}": float(v) for e, v in zip(EMOTIONS, per)},
    }


class EmotionClassifier:
    """Five independent binary logistic regressions + per-emotion thresholds."""

    def __init__(self, C=1.0, class_weight=None, seed=42, thresholds=None):
        self.C, self.class_weight, self.seed = C, class_weight, seed
        self.thresholds = np.array(thresholds if thresholds is not None else [0.5] * 5)
        self.scaler, self.models = None, None

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)  # fitted on train only
        Xs = self.scaler.transform(X)
        self.models = [
            LogisticRegression(C=self.C, class_weight=self.class_weight,
                               max_iter=3000, random_state=self.seed).fit(Xs, y[:, i])
            for i in range(len(EMOTIONS))
        ]
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        return np.column_stack([m.predict_proba(Xs)[:, 1] for m in self.models])

    def predict(self, X):
        return (self.predict_proba(X) >= self.thresholds).astype(int)

    def save(self, path, meta=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"emotions": EMOTIONS, "C": self.C, "class_weight": self.class_weight,
                     "seed": self.seed, "thresholds": self.thresholds.tolist(),
                     "scaler": self.scaler, "models": self.models, "meta": meta or {}}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        assert d["emotions"] == EMOTIONS, "emotion order mismatch"
        obj = cls(d["C"], d["class_weight"], d["seed"], d["thresholds"])
        obj.scaler, obj.models = d["scaler"], d["models"]
        return obj