from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATASET_NAME = "brighter-dataset/BRIGHTER-emotion-categories"
SPLITS = ["train", "dev", "test"]
EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise"]
COLUMNS = ["id", "text"] + EMOTIONS


def load_data():
    return load_dataset(DATASET_NAME, "eng")


def show_dataset_info(dataset):
    print("Dataset info:")
    for split_name, split in dataset.items():
        print(split_name.upper())
        df = split.to_pandas()
        print(df.shape)
        print(df.columns.tolist())
        print(df.dtypes)
        print(df.head())
        print(df.isna().sum())
        print(f"\n{'=' * 50}\n")


def prepare_split(split):
    df = split.to_pandas()
    # Не изменяем исходные тексты и порядок строк.
    return df[COLUMNS].copy()


def validate_split(df, split_name):
    # Только основные проверки, без сообщений при нормальных данных.
    if df.empty or df.isna().any().any():
        raise ValueError(f"{split_name}: empty dataset or missing values")
    if df["id"].duplicated().any() or df["id"].str.strip().eq("").any():
        raise ValueError(f"{split_name}: duplicate or blank IDs")
    if df["text"].str.strip().eq("").any():
        raise ValueError(f"{split_name}: blank text")
    if not df[EMOTIONS].isin([0, 1]).all().all():
        raise ValueError(f"{split_name}: labels must be 0 or 1")


def check_overlaps(frames):
    rows = []
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:
            ids = set(frames[first]["id"]) & set(frames[second]["id"])
            texts = set(frames[first]["text"]) & set(frames[second]["text"])
            if ids:
                raise ValueError(f"Shared IDs between {first} and {second}")
            rows.append({"split_a": first, "split_b": second,
                         "common_ids": len(ids), "common_texts": len(texts)})
    return pd.DataFrame(rows)


def calculate_statistics(df, split_name):
    labels = df[EMOTIONS].sum(axis=1)
    # Длина в словах, не в токенах BERT.
    lengths = df["text"].str.split().str.len()
    summary = {
        "split": split_name, "rows": len(df),
        "no_emotion": int(labels.eq(0).sum()),
        "multiple_emotions": int(labels.ge(2).sum()),
        "repeated_text_rows": int(df["text"].duplicated().sum()),
        "duplicate_example_rows": int(df.duplicated(subset=["text"] + EMOTIONS).sum()),
        "words_min": int(lengths.min()),
        "words_mean": round(float(lengths.mean()), 2),
        "words_median": float(lengths.median()), "words_max": int(lengths.max()),
    }
    counts = []
    for emotion in EMOTIONS:
        count = int(df[emotion].sum())
        counts.append({"split": split_name, "emotion": emotion, "count": count,
                       "percent": round(count / len(df) * 100, 2)})
    distribution = []
    for count in range(6):
        distribution.append({"split": split_name, "number_of_emotions": count,
                             "texts": int(labels.eq(count).sum())})
    return {"summary": summary, "counts": counts,
            "distribution": distribution, "lengths": lengths}


def save_statistics(statistics, overlaps, reports_dir):
    reports_dir = Path(reports_dir)
    summaries, counts, distributions = [], [], []
    for name in SPLITS:
        summaries.append(statistics[name]["summary"])
        counts.extend(statistics[name]["counts"])
        distributions.extend(statistics[name]["distribution"])
    tables = {"dataset_summary.csv": pd.DataFrame(summaries),
              "emotion_counts.csv": pd.DataFrame(counts),
              "label_distribution.csv": pd.DataFrame(distributions),
              "split_overlaps.csv": overlaps}
    for filename, table in tables.items():
        table.to_csv(reports_dir / filename, index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(EMOTIONS, [row["count"] for row in statistics["train"]["counts"]])
    ax.set(title="Emotion frequencies in train", xlabel="Emotion", ylabel="Number of texts")
    fig.tight_layout()
    fig.savefig(reports_dir / "train_emotions.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(statistics["train"]["lengths"], bins=30, edgecolor="white")
    ax.set(title="Text lengths in train", xlabel="Number of whitespace-separated words",
           ylabel="Number of texts")
    fig.tight_layout()
    fig.savefig(reports_dir / "train_text_lengths.png", dpi=150)
    plt.close(fig)


def save_processed_data(frames, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        df = frames[name]
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8")


def clean_data(frames):
    """Удаляем повторы полных текстов внутри выборок и между ними."""
    cleaned = {}
    summaries = []
    removed = []
    protected_texts = set()

    # Сохраняем примеры test, затем dev, затем train.
    for name in ["test", "dev", "train"]:
        original = frames[name]
        # Оставляем первый пример с таким текстом, независимо от ID и меток.
        df = original.drop_duplicates(subset=["text"], keep="first").copy()
        duplicate_count = len(original) - len(df)
        duplicates = original.loc[~original.index.isin(df.index)].copy()
        duplicates["split"] = name
        duplicates["reason"] = "duplicate_text"
        removed.append(duplicates)

        # Сравниваем весь текст, а не отдельные слова или подстроки.
        overlap = df["text"].isin(protected_texts)
        overlap_count = int(overlap.sum())
        excluded = df.loc[overlap].copy()
        excluded["split"] = name
        excluded["reason"] = "full_text_in_held_out_split"
        removed.append(excluded)
        df = df.loc[~overlap].copy()

        protected_texts.update(df["text"])
        cleaned[name] = df.reset_index(drop=True)
        summaries.append({"split": name, "before": len(original),
                          "duplicates_removed": duplicate_count,
                          "overlaps_removed": overlap_count, "after": len(df)})
        assert not df["text"].duplicated().any()

    summary = pd.DataFrame(summaries).set_index("split").loc[SPLITS].reset_index()
    summary.to_csv(REPORTS_DIR / "cleaning_summary.csv", index=False)
    pd.concat(removed, ignore_index=True).to_csv(REPORTS_DIR / "removed_rows.csv", index=False)
    return cleaned, summary


def main():
    dataset = load_data()
    show_dataset_info(dataset)
    frames = {}
    for name in SPLITS:
        df = prepare_split(dataset[name])
        validate_split(df, name)
        frames[name] = df
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frames, summary = clean_data(frames)
    overlaps = check_overlaps(frames)
    assert overlaps["common_texts"].sum() == 0
    statistics = {}
    for name, df in frames.items():
        statistics[name] = calculate_statistics(df, name)
    save_statistics(statistics, overlaps, REPORTS_DIR)
    save_processed_data(frames, OUTPUT_DIR)
    print("\nData preparation completed successfully.")
    print(summary.to_string(index=False))
    print("Repeated full texts within splits remaining: 0")
    print("Full-text overlaps between splits remaining: 0")
    print(f"CSV files saved to: {OUTPUT_DIR}")
    print(f"Reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
