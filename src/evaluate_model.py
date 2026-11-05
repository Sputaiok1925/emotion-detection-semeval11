"""Evaluate the fixed mean-pooling model or predict emotions for your text."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, multilabel_confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from classifier import EmotionClassifier, EMOTIONS

FEATURES = ROOT / "features/bert-base-uncased_len128/mean"
MODEL = ROOT / "models/logreg_mean.joblib"


def evaluate_test(model):
    df = pd.read_csv(ROOT / "data/processed/test.csv")
    ids = json.loads((FEATURES / "test_ids.json").read_text())
    X = np.load(FEATURES / "test_X.npy")
    if ids != df["id"].tolist() or X.shape != (len(df), 768):
        raise ValueError("Test features and CSV do not match")
    probabilities = model.predict_proba(X)
    predicted = (probabilities >= model.thresholds).astype(int)
    actual = df[EMOTIONS].to_numpy()
    report = classification_report(actual, predicted, target_names=EMOTIONS,
                                   output_dict=True, zero_division=0)
    print(classification_report(actual, predicted, target_names=EMOTIONS,
                                digits=4, zero_division=0))
    metrics = {"macro_f1": f1_score(actual, predicted, average="macro"),
               "micro_f1": f1_score(actual, predicted, average="micro"),
               "exact_match": float((actual == predicted).all(axis=1).mean()),
               "model": str(MODEL.relative_to(ROOT)),
               "thresholds": model.thresholds.tolist()}
    out = ROOT / "reports/evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(report).T.to_csv(out / "classification_report.csv")
    rows = df[["id", "text"]].copy()
    for i, emotion in enumerate(EMOTIONS):
        rows[emotion + "_true"] = actual[:, i]
        rows[emotion + "_pred"] = predicted[:, i]
        rows[emotion + "_probability"] = probabilities[:, i]
    rows.to_csv(out / "test_predictions.csv", index=False)
    rows.loc[(actual != predicted).any(axis=1)].to_csv(out / "errors.csv", index=False)
    matrices = multilabel_confusion_matrix(actual, predicted)
    pd.DataFrame([dict(emotion=e, TN=int(m[0,0]), FP=int(m[0,1]),
                       FN=int(m[1,0]), TP=int(m[1,1]))
                  for e, m in zip(EMOTIONS, matrices)]).to_csv(out / "confusion_counts.csv", index=False)
    print("Reports saved to:", out)


def predict_texts(model, texts):
    from frozen_bert.encoder import EncoderConfig, FrozenBertEncoder
    config = json.loads((FEATURES / "config.json").read_text())["encoder"]
    encoder = FrozenBertEncoder(EncoderConfig(
        model_name=config["model_name"], revision=config["resolved_revision"],
        pooling="mean", max_length=config["max_length"],
        exclude_special_tokens=config["exclude_special_tokens"]))
    probabilities = model.predict_proba(encoder.encode(texts))
    for text, values in zip(texts, probabilities):
        emotions = [e for e, value, threshold in zip(EMOTIONS, values, model.thresholds)
                    if value >= threshold]
        print("\nText:", text)
        print("Emotions:", ", ".join(emotions) or "None detected")
        print("Scores:", ", ".join(f"{e}: {v:.3f}" for e, v in zip(EMOTIONS, values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", action="append", help="English text; can be repeated")
    parser.add_argument("--examples", action="store_true", help="Run a few illustrative examples")
    args = parser.parse_args()
    model = EmotionClassifier.load(MODEL)
    texts = args.text or []
    if args.examples:
        texts += ["I am so happy to see you!", "I am terrified and cannot stop crying.",
                  "I am furious that you lied to me!", "The meeting starts at ten."]
    if texts:
        predict_texts(model, texts)
    else:
        evaluate_test(model)


if __name__ == "__main__":
    main()
