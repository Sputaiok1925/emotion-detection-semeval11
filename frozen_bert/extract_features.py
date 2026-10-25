"""Extract frozen-BERT features for train/dev/test.

Example:
    python extract_features.py --data-dir data --out-dir features \
        --model bert-base-uncased --max-length 128 --batch-size 32

Expects train.csv, dev.csv, test.csv (columns id,text,...) in --data-dir,
prepared by participant 1. Missing files are skipped (e.g. while test is not ready).

Output (a separate folder per pooling):
    features/<tag>/mean/{train,dev,test}_X.npy     N x D matrix, float32
    features/<tag>/mean/{train,dev,test}_ids.json  list of ids in the same row order
    features/<tag>/mean/config.json                BERT settings, statistics, data hashes
    features/<tag>/cls/...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from encoder import EncoderConfig, FrozenBertEncoder, POOLINGS

SPLITS = ("train", "dev", "test")


def sha256_file(path: Path) -> str:
    """File fingerprint: changes whenever the data file changes (used for cache checks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(path: Path) -> tuple[list[str], list[str]]:
    # id as string (so "001" stays "001"); keep_default_na=False keeps empty texts as ""
    df = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    for col in ("id", "text"):
        if col not in df.columns:
            sys.exit(f"{path}: missing column '{col}' (found: {list(df.columns)})")
    if df["id"].duplicated().any():
        sys.exit(f"{path}: ids are not unique")
    n_empty = int((df["text"].str.strip() == "").sum())
    if n_empty:
        print(f"WARNING: {path} contains {n_empty} empty texts", file=sys.stderr)
    return df["id"].tolist(), df["text"].tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="features")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--revision", default=None, help="model commit hash on the HF Hub (recommended)")
    ap.add_argument("--pooling", choices=[*POOLINGS, "both"], default="both")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--exclude-special-tokens", action="store_true",
                    help="mean pooling without [CLS] and [SEP]")
    ap.add_argument("--tag", default=None, help="subfolder name (default: built from model and max_length)")
    ap.add_argument("--force", action="store_true", help="recompute even if the cache is up to date")
    args = ap.parse_args()

    data_dir, out_root = Path(args.data_dir), Path(args.out_dir)
    # pooling here is irrelevant: encode_all() returns both mean and cls anyway
    cfg = EncoderConfig(model_name=args.model, revision=args.revision, pooling="mean",
                        max_length=args.max_length, batch_size=args.batch_size,
                        device=args.device, exclude_special_tokens=args.exclude_special_tokens)
    tag = args.tag or f"{args.model.replace('/', '_')}_len{args.max_length}" + \
        ("_nospecial" if args.exclude_special_tokens else "")
    poolings = POOLINGS if args.pooling == "both" else (args.pooling,)

    # Which split files exist, and their fingerprints
    split_files = {s: data_dir / f"{s}.csv" for s in SPLITS if (data_dir / f"{s}.csv").exists()}
    if not split_files:
        sys.exit(f"No train.csv / dev.csv / test.csv found in {data_dir}")
    data_hashes = {s: sha256_file(p) for s, p in split_files.items()}

    # Cache is valid only for the same data AND the same settings
    def cache_ok(pool: str) -> bool:
        cfg_path = out_root / tag / pool / "config.json"
        if not cfg_path.exists():
            return False
        old = json.loads(cfg_path.read_text())
        same_settings = all(old["encoder"].get(k) == getattr(cfg, k) for k in
                            ("model_name", "revision", "max_length", "exclude_special_tokens"))
        same_data = old.get("data_sha256") == data_hashes
        files = all((out_root / tag / pool / f"{s}_X.npy").exists() for s in split_files)
        return same_settings and same_data and files

    if not args.force and all(cache_ok(p) for p in poolings):
        print("Cache is up to date (same data and settings). Use --force to recompute.")
        return

    print(f"Loading model {cfg.model_name} ...")
    enc = FrozenBertEncoder(cfg)
    print(f"Device: {enc.device}, dimension D={enc.dim}, revision={enc.resolved_revision}")

    stats, results = {}, {p: {} for p in poolings}
    for split, path in split_files.items():
        ids, texts = load_split(path)
        print(f"[{split}] {len(texts)} texts")
        mats = enc.encode_all(texts, show_progress=True)
        stats[split] = dict(enc.last_stats)
        for p in poolings:
            X = mats[p]
            # Sanity checks: one row per text, no NaN/inf
            assert X.shape == (len(ids), enc.dim), (X.shape, len(ids), enc.dim)
            assert np.isfinite(X).all(), "matrix contains NaN/inf"
            results[p][split] = (ids, X)
        s = stats[split]
        print(f"  time {s['seconds']:.1f} s ({s['texts_per_sec']:.1f} texts/s), "
              f"truncated {s['n_truncated']} ({100 * s['truncated_frac']:.2f}%)")

    # Save matrices, ids (same row order) and a "passport" config.json
    for p in poolings:
        out = out_root / tag / p
        out.mkdir(parents=True, exist_ok=True)
        for split, (ids, X) in results[p].items():
            np.save(out / f"{split}_X.npy", X)
            (out / f"{split}_ids.json").write_text(json.dumps(ids, ensure_ascii=False))
        d = enc.describe()
        d["pooling"] = p
        meta = dict(
            encoder=d,
            dim=enc.dim,
            n_examples={s: len(results[p][s][0]) for s in results[p]},
            data_sha256=data_hashes,
            data_dir=str(data_dir),
            stats=stats,
        )
        (out / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
