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

## Model evaluation and custom texts

We evaluated the saved `models/logreg_mean.joblib` classifier on all **2762 cleaned test examples**. The pipeline uses frozen `bert-base-uncased` (maximum length 128, mean pooling including special tokens but excluding padding), followed by a StandardScaler and five independent logistic regressions. Training uses train; hyperparameters and thresholds were selected on dev. Mean pooling was selected based on dev results. No fitting or threshold tuning is performed by the evaluation script.

For test evaluation, the script loads precomputed BERT vectors and checks their shape and ID order against `data/processed/test.csv`. It compares predicted binary labels with the five reference labels. The saved thresholds, in emotion order, are **0.60, 0.30, 0.60, 0.50, 0.60**.

Run from the project root with your configured Python interpreter:

```bash
python src/evaluate_model.py
python src/evaluate_model.py --text "I am happy but also scared."
python src/evaluate_model.py --examples
```

Test evaluation requires `classifier.py`, `models/`, `features/bert-base-uncased_len128/mean/`, and the processed CSV files. These components come from the data preparation, frozen-BERT and classifier branches and must be present together. Install dependencies with `python -m pip install -r requirements.txt`. For a new text, the script runs BERT and then the classifier; the first run downloads BERT to the Hugging Face cache. Text examples are illustrative checks, not a substitute for the labelled test set.

### Test results

| Emotion | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| anger | 0.4892 | 0.5621 | 0.5231 | 322 |
| fear | 0.6627 | 0.9689 | 0.7870 | 1541 |
| joy | 0.6810 | 0.6403 | 0.6600 | 670 |
| sadness | 0.6380 | 0.7503 | 0.6896 | 881 |
| surprise | 0.6900 | 0.6165 | 0.6512 | 798 |

- **Macro-F1: 0.6622** — mean F1 across the five emotions, each with equal weight.
- **Micro-F1: 0.7068** — computed from decisions pooled across all emotions.
- **Exact match: 0.3566** — all five predicted labels match the reference labels for 35.66% of texts.
- **Support** is the number of texts actually labelled with an emotion, not the number of correct predictions. Supports sum to 4212 because a text can have several emotions.

Fear has the highest F1 and recall, but its lower precision indicates false positives. Anger has the lowest F1 and the fewest positive test examples. These results show useful but imperfect predictions; F1 is not the percentage of completely correct texts. The tuned dev macro-F1 was approximately 0.7393; the lower test score shows that performance does not fully carry over to unseen data. Dev contains only 115 examples, so model and threshold selection can be sensitive to this small sample.

Reports are saved in `reports/evaluation/`:
- `metrics.json`: overall metrics, model path and thresholds.
- `classification_report.csv`: precision, recall, F1 and support.
- `test_predictions.csv`: texts, true labels, predicted labels and scores.
- `errors.csv`: texts with at least one incorrect label, for manual error analysis.
- `confusion_counts.csv`: TN, FP, FN and TP for each emotion.

Repeated evaluation overwrites these reports. Keep test results for final reporting; do not tune the model against this test set. Since the official splits were cleaned, these scores are not directly comparable with the official competition leaderboard.
