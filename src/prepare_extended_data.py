"""Add translated German/Spanish train+test to English train; do not use dev."""
import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from data_preparation import (
    PROJECT_ROOT, DATASET_NAME, COLUMNS, EMOTIONS, prepare_split, validate_split,
    check_overlaps, clean_data, calculate_statistics, save_statistics, save_processed_data,
)
from translate_texts import translate_texts, MODELS

OUTPUT_DIR = PROJECT_ROOT / "data/extended"
REPORTS_DIR = PROJECT_ROOT / "reports/extended"
CACHE_DIR = PROJECT_ROOT / "data/translation_cache"
SEED = 42


def remove_invalid_rows(df, source):
    """Keep complete examples with text, ID and five binary labels."""
    df = df.copy()
    valid_text = df["text"].map(lambda x: isinstance(x, str) and bool(x.strip())
                              and x.strip().lower() not in {"<unset>", "#name?"})
    valid_id = df["id"].map(lambda x: isinstance(x, str) and bool(x.strip()))
    valid = valid_text & valid_id & df[EMOTIONS].isin([0, 1]).all(axis=1)
    removed = df.loc[~valid].copy()
    removed["source"] = source
    removed["reason"] = "invalid_text_id_or_labels"
    df = df.loc[valid].copy()
    df["text"] = df["text"].str.strip()
    df[EMOTIONS] = df[EMOTIONS].astype(int)
    # Conflicting IDs must not silently associate one example with another.
    if df["id"].duplicated().any():
        raise ValueError(f"{source}: duplicate IDs; inspect the source before continuing")
    print(f"{source}: removed {len(removed)} invalid rows; kept {len(df)}", flush=True)
    return df, removed


def load_foreign_data(language):
    parts = []
    for split_name in ["train", "test"]:
        split = load_dataset(DATASET_NAME, language, split=split_name)
        df = prepare_split(split)
        df["source_id"] = df["id"]
        df["source_split"] = split_name
        df["language"] = language
        df["id"] = language + "_" + split_name + "_" + df["id"].astype(str)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def select_test_size(train, test, fraction=0.30):
    # T / (N + T) = fraction -> T = N * fraction / (1 - fraction).
    target = max(1, round(len(train) * fraction / (1 - fraction)))
    count = min(len(test), target)
    selected = test if count == len(test) else test.sample(n=count, random_state=SEED).sort_index()
    excluded = test.loc[~test.index.isin(selected.index)].copy()
    # Excluded test rows are never moved into train.
    return selected.reset_index(drop=True), excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Use the existing cleaned English files; original baseline outputs stay intact.
    frames, invalid_rows, source_counts = {}, [], []
    for name in ["train", "test"]:
        df = pd.read_csv(PROJECT_ROOT / "data/processed" / f"{name}.csv", keep_default_na=False)
        df, removed = remove_invalid_rows(df[COLUMNS], "eng_" + name)
        frames[name] = df.reset_index(drop=True)
        invalid_rows.append(removed)
        source_counts.append({"source": "eng_" + name, "valid_rows": len(df)})

    additions = []
    for language in MODELS:
        df, removed = remove_invalid_rows(load_foreign_data(language), language + "_original")
        invalid_rows.append(removed)
        if df.empty:
            raise ValueError(f"{language}: no valid examples with all five emotion labels")
        df["original_text"] = df["text"]
        df["text"] = translate_texts(df["text"].tolist(), language, CACHE_DIR,
                                     args.batch_size, args.device)
        df, removed = remove_invalid_rows(df, language + "_translated")
        invalid_rows.append(removed)
        source_counts.append({"source": language + "_translated_train_and_test", "valid_rows": len(df)})
        additions.append(df)

    translations = pd.concat(additions, ignore_index=True)
    frames["train"] = pd.concat([frames["train"], translations[COLUMNS]], ignore_index=True)
    # Deduplicate before sizing test. All original test texts are protected.
    frames, summary = clean_data(frames, REPORTS_DIR)
    frames["test"], excluded = select_test_size(frames["train"], frames["test"])
    overlaps = check_overlaps(frames)
    for name, df in frames.items():
        validate_split(df, name)
        if df["text"].duplicated().any():
            raise ValueError(f"{name}: repeated texts remain")
    if overlaps["common_texts"].sum():
        raise ValueError("Train and test contain shared texts")

    statistics = {name: calculate_statistics(df, name) for name, df in frames.items()}
    save_statistics(statistics, overlaps, REPORTS_DIR)
    save_processed_data(frames, OUTPUT_DIR)
    pd.concat(invalid_rows, ignore_index=True).to_csv(REPORTS_DIR / "invalid_rows.csv", index=False)
    pd.DataFrame(source_counts).to_csv(REPORTS_DIR / "source_counts.csv", index=False)
    excluded.to_csv(REPORTS_DIR / "excluded_test_rows.csv", index=False)
    translations["used_in_train"] = translations["id"].isin(frames["train"]["id"])
    translations.to_csv(REPORTS_DIR / "translations.csv", index=False)
    sizes = {name: len(df) for name, df in frames.items()}
    sizes["test_fraction"] = len(frames["test"]) / sum(len(df) for df in frames.values())
    sizes["test_rows_excluded_for_size"] = len(excluded)
    (REPORTS_DIR / "final_sizes.json").write_text(json.dumps(sizes, indent=2), encoding="utf-8")
    print("\nExtended dataset completed.")
    print(summary.to_string(index=False))
    print(json.dumps(sizes, indent=2))
    print("Saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
