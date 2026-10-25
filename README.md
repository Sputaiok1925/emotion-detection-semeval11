# emotion-detection-semeval11
Bridging the Gap in Text-Based Emotion Detection.

## Data preparation (Track A, English)

Install the dependencies and run from the project folder:

```bash
python -m pip install -r requirements.txt
python src/data_preparation.py
```

The script downloads the English splits from
[BRIGHTER emotion categories](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories).
The first download requires an internet connection; Hugging Face caches the dataset locally.

Each output CSV contains `id`, `text`, and five binary emotion labels:
`anger`, `fear`, `joy`, `sadness`, `surprise`. A text can have multiple emotions.

Cleaning keeps the first occurrence of each identical full text within a split,
even when labels differ. Texts shared between splits are retained in priority order:
`test`, then `dev`, then `train`. Individual repeated words are not removed,
and the text itself is not modified.

Outputs:
- `data/processed/train.csv`, `dev.csv`, `test.csv`: cleaned data for the team.
- `reports/`: statistics, plots, cleaning counts, and `removed_rows.csv`, which records removed examples.

Current cleaned split sizes: train **2755**, dev **115**, test **2762**.
These are modified splits; scores on the cleaned test set are not directly comparable
with scores on the original official test set. Running the script again overwrites the generated outputs.
