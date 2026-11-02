# Frozen BERT features (participant 2)

Turns every text into a 768-dimensional vector using a **frozen** `bert-base-uncased`
(weights are never trained). The vectors are the input of the classifier (participant 3)
and of the app (participant 4).

Pipeline position:

```
text -> frozen BERT (this folder) -> vector (768) -> classifier -> 5 emotions
```

## Ready-made features (no need to run anything)

Precomputed features are committed to this branch in:

```
features/bert-base-uncased_len128/
    mean/   train_X.npy  dev_X.npy  test_X.npy  + *_ids.json + config.json   <- use this one
    cls/    same files for CLS pooling (for the pooling experiment)
```

| Split | Shape of `X` | Truncated texts | Time on CPU |
| --- | --- | --- | --- |
| train | 2755 x 768 | 0.00% | 19.6 s |
| dev | 115 x 768 | 0.00% | 1.0 s |
| test | 2762 x 768 | 0.00% | 19.2 s |

Settings: `bert-base-uncased`, revision `86b5e0934494bd15c9632b12f734a8a67f723594`,
max length 128 tokens, mean pooling includes `[CLS]` and `[SEP]` (padding excluded).
They were computed from the cleaned data of participant 1 (`data/processed/*.csv`).
`config.json` stores SHA-256 fingerprints of those CSV files, so you can check that
your data is the same version.

### Load features and labels

Row `i` of `X` belongs to `ids[i]`. Always match by id, never by assumption:

```python
import json
import numpy as np
import pandas as pd

F = "features/bert-base-uncased_len128/mean"
EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise"]   # fixed label order

def load(split):
    X = np.load(f"{F}/{split}_X.npy")                         # (N, 768) float32
    ids = json.load(open(f"{F}/{split}_ids.json"))
    df = pd.read_csv(f"data/processed/{split}.csv", dtype={"id": str})
    assert df["id"].tolist() == ids, "ids differ: data version does not match the features"
    return X, df[EMOTIONS].to_numpy()                         # y: (N, 5) of 0/1

X_train, y_train = load("train")
X_dev, y_dev = load("dev")
# X_test, y_test = load("test")   # use only once, for the final evaluation
```

Rules: tune only on `dev`; touch `test` once at the very end.

## Recompute the features (only if data or settings change)

From the repository root:

```bash
python -m pip install -r frozen_bert/requirements.txt

python frozen_bert/extract_features.py --data-dir data/processed --out-dir features \
    --model bert-base-uncased \
    --revision 86b5e0934494bd15c9632b12f734a8a67f723594 \
    --max-length 128 --batch-size 32
```

`data/processed/` comes from the branch of participant 1 (it is not on this branch):

```bash
git fetch origin
git archive origin/Data_preparation,model_evaluation data/processed | tar -x
```

The first run downloads `bert-base-uncased` (about 440 MB). On a laptop CPU the whole
dataset takes roughly one minute. If the data and settings did not change, the script
reuses the existing features; add `--force` to recompute anyway.

Options: `--pooling mean|cls|both` (default both), `--exclude-special-tokens`,
`--device cpu|cuda`, `--tag`.

The message `UNEXPECTED ... cls.predictions...` while loading the model is harmless:
those layers belong to BERT's pretraining and are not used here.

## Use in code (app, participant 4)

```python
from encoder import FrozenBertEncoder, EncoderConfig

enc = FrozenBertEncoder(EncoderConfig(
    revision="86b5e0934494bd15c9632b12f734a8a67f723594",
    pooling="mean", max_length=128))
X = enc.encode(["I am so happy!"])        # shape (1, 768)
```

`encode` is the single function used for both training features and the app, so the
processing is identical. Pass a **list** of texts, not a single string.

## Tests

```bash
cd frozen_bert
pytest -q test_encoder.py                # real model, 18 checks
python make_tiny_bert.py /tmp/tiny_bert  # optional offline check with a tiny random model
EMO_BERT_MODEL=/tmp/tiny_bert pytest -q test_encoder.py
```

## Files

| File | Purpose |
| --- | --- |
| `encoder.py` | `FrozenBertEncoder`: text -> vectors (padding, attention mask, pooling, frozen weights) |
| `extract_features.py` | Reads train/dev/test CSV, saves `X.npy`, `ids.json`, `config.json` |
| `test_encoder.py` | 18 automated checks (single = batch, order kept, weights unchanged, ...) |
| `make_tiny_bert.py` | Tiny random BERT for offline tests of the code logic only |

## Notes

- Results on GPU and CPU may differ around the 5th-6th decimal; this does not change classification.
- Do not mix features from different settings: the folder name contains the model and
  max length, and `config.json` contains everything else.
- `data/` is intentionally not committed on this branch; `features/` is.
