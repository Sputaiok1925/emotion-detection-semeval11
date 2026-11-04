"""Train on train, tune on dev. Test only with --final."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from classifier import EMOTIONS, EmotionClassifier, evaluate, load_split
from tune_thresholds import tune_thresholds

SEED = 42
C_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 1.0]
CW_GRID = [None, "balanced"]


def log_run(record):
    Path("reports").mkdir(exist_ok=True)
    with open("reports/runs_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_predictions(ids, proba, pred, prefix):
    Path("outputs").mkdir(exist_ok=True)
    pd.DataFrame(proba, columns=EMOTIONS).assign(id=ids)[["id"] + EMOTIONS] \
        .to_csv(f"outputs/{prefix}_proba.csv", index=False)
    pd.DataFrame(pred, columns=EMOTIONS).assign(id=ids)[["id"] + EMOTIONS] \
        .to_csv(f"outputs/{prefix}_pred.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    ap.add_argument("--final", action="store_true", help="evaluate saved model on test")
    args = ap.parse_args()
    model_path = f"models/logreg_{args.pooling}.joblib"
    env = {"python": platform.python_version(), "sklearn": sklearn.__version__,
           "numpy": np.__version__, "pandas": pd.__version__}

    if args.final:  # run ONCE, after the model is fixed
        X_te, y_te, ids_te = load_split("test", args.pooling)
        clf = EmotionClassifier.load(model_path)
        proba, pred = clf.predict_proba(X_te), clf.predict(X_te)
        save_predictions(ids_te, proba, pred, f"test_{args.pooling}")
        res = evaluate(y_te, pred)
        print("TEST:", json.dumps(res, indent=2))
        log_run({"stage": "test", "pooling": args.pooling, "model": model_path, **res})
        return

    X_tr, y_tr, _ = load_split("train", args.pooling)
    X_dev, y_dev, ids_dev = load_split("dev", args.pooling)
    print(f"train {X_tr.shape}, dev {X_dev.shape}; positives in dev:",
          dict(zip(EMOTIONS, y_dev.sum(0).tolist())))

    # 1) baseline: C=1, no class weights, thresholds 0.5
    # 2) grid over C and class_weight, all with thresholds 0.5, scored on dev
    results = []
    for cw in CW_GRID:
        for C in C_GRID:
            t0 = time.time()
            clf = EmotionClassifier(C=C, class_weight=cw, seed=SEED).fit(X_tr, y_tr)
            res = evaluate(y_dev, clf.predict(X_dev))
            res.update(C=C, class_weight=cw, train_seconds=round(time.time() - t0, 1))
            results.append(res)
            print(f"C={C:<6} cw={str(cw):<9} dev macro-F1={res['macro_f1']:.4f}")
            log_run({"stage": "dev_grid", "pooling": args.pooling, "seed": SEED,
                     "thresholds": "0.5", "env": env, **res})
    best = max(results, key=lambda r: r["macro_f1"])
    print("BEST:", best["C"], best["class_weight"], f"{best['macro_f1']:.4f}")

    # 3) thresholds tuned on dev with the best settings
    clf = EmotionClassifier(C=best["C"], class_weight=best["class_weight"], seed=SEED).fit(X_tr, y_tr)
    proba_dev = clf.predict_proba(X_dev)
    base = evaluate(y_dev, (proba_dev >= 0.5).astype(int))
    clf.thresholds = tune_thresholds(y_dev, proba_dev)
    tuned = evaluate(y_dev, clf.predict(X_dev))
    print("thresholds:", dict(zip(EMOTIONS, clf.thresholds.round(2))))
    print(f"dev macro-F1 @0.5 = {base['macro_f1']:.4f} | tuned = {tuned['macro_f1']:.4f} "
          "(tuned is optimistic: thresholds were fitted on the same dev)")

    meta = {"pooling": args.pooling, "features": str(Path("features/bert-base-uncased_len128") / args.pooling),
            "bert": "bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594, max_len=128",
            "env": env}
    clf.save(model_path, meta)
    with open(f"models/thresholds_{args.pooling}.json", "w", encoding="utf-8") as f:
        json.dump(dict(zip(EMOTIONS, clf.thresholds.tolist())), f, indent=2)
    save_predictions(ids_dev, proba_dev, clf.predict(X_dev), f"dev_{args.pooling}")
    log_run({"stage": "dev_final", "pooling": args.pooling, "seed": SEED, "C": best["C"],
             "class_weight": best["class_weight"], "thresholds": clf.thresholds.tolist(),
             "dev_macro_f1_at_0.5": base["macro_f1"], "dev_macro_f1_tuned": tuned["macro_f1"], "env": env})
    print("saved:", model_path)


if __name__ == "__main__":
    main()