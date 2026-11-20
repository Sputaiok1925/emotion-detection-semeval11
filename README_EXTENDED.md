# Extended English dataset

Run from the project folder using the same Python interpreter as PyCharm:

```powershell
& "C:\Users\sputa\AppData\Local\Programs\Python\Python314\python.exe" -m pip install -r requirements-translation.txt
& "C:\Users\sputa\AppData\Local\Programs\Python\Python314\python.exe" src/prepare_extended_data.py
```

The first run downloads German/Spanish BRIGHTER data and two local OPUS-MT translation models. It can take a long time on CPU. `--batch-size 4` reduces memory use. GPU is selected automatically only if the installed PyTorch supports CUDA; otherwise CPU is used.

## Data flow

1. Read the existing English `data/processed/train.csv` and `test.csv`.
2. Download German (`deu`) and Spanish (`esp`) train and test. No dev split is used in the output or training data (the dataset library may cache other splits internally).
3. Remove empty/invalid text or ID, literal `<unset>` / `#name?` placeholders, and examples without all five binary emotion labels. Strip whitespace at text boundaries; preserve case and internal punctuation. Duplicate IDs cause an error for inspection.
4. Translate the foreign texts into English with `Helsinki-NLP/opus-mt-de-en` and `Helsinki-NLP/opus-mt-es-en`. Labels are inherited without changes. Long source texts are split into chunks instead of truncated.
5. Append the translations to English train. Remove identical full texts, keeping the first occurrence. English train is placed first. English test takes priority over train in overlaps, even for test rows later excluded by the size rule. Matching is exact after stripping boundary whitespace; it does not detect paraphrases or all cross-language equivalents.
6. Keep at most `round(train_size * 0.30 / 0.70)` examples from the cleaned English test. This targets 30% of the FINAL total, subject to whole-row rounding. If test is smaller, keep all of it. Subsampling is deterministic (`random_state=42`); excluded test rows are never added to train.

## Outputs

- `data/extended/train.csv`, `test.csv`: seven columns in the original format; no dev file.
- `data/translation_cache/`: resumable translations, saved after each batch. Repeated source texts are translated once. Restart the same command to continue. To deliberately regenerate translations, use a new cache or remove the corresponding cache file.
- `reports/extended/`: dataset statistics and plots, cleaning counts, removed duplicates/overlaps, invalid examples, excluded test examples, final sizes, and translations with original text, source language, source split and `used_in_train`.

`cleaning_summary.csv` describes deduplication BEFORE test subsampling. `final_sizes.json` and `dataset_summary.csv` describe the final output.

The original English processed files and baseline reports are not overwritten. `data_preparation.py` retains its original entry point; its shared functions now also accept datasets without dev and a separate report directory. BERT, classifiers and evaluation scripts are not changed by this preparation step. Existing BERT feature files cannot be reused for the enlarged training set.

Automatic checks do not establish translation quality: inspect a sample of `translations.csv` for meaning and preserved emotions. Using foreign test labels as training data makes this an expanded-data experiment, not an evaluation on those foreign test sets. For future training, hold out validation data from TRAIN before tuning; do not tune using English test. Keep translations/duplicates of the same source together when making that validation split.
