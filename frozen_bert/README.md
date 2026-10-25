# Participant 2: Frozen BERT

Turns texts into feature vectors with a frozen `bert-base-uncased`. BERT weights are never trained.

## Install and check
```bash
pip install -r requirements.txt
pytest -q test_encoder.py     # downloads bert-base-uncased (~440 MB), 18 checks
```
Offline logic check with a tiny random model:
```bash
python make_tiny_bert.py /tmp/tiny_bert
EMO_BERT_MODEL=/tmp/tiny_bert pytest -q test_encoder.py
```

## Extract features
Needs `data/train.csv`, `data/dev.csv`, `data/test.csv` (columns `id,text,...`) from participant 1.
```bash
python extract_features.py --data-dir data --out-dir features \
    --model bert-base-uncased --revision <commit-hash> --max-length 128 --batch-size 32
```
Options: `--pooling mean|cls|both` (default both), `--exclude-special-tokens`,
`--device cpu|cuda`, `--force`, `--tag`.

Output: `features/<tag>/{mean,cls}/{train,dev,test}_X.npy`, `*_ids.json`, `config.json`.
Row i of `X.npy` belongs to id i of `ids.json`.

## Use in code (participants 3 and 4)
```python
from encoder import FrozenBertEncoder, EncoderConfig
enc = FrozenBertEncoder(EncoderConfig(pooling="mean", max_length=128))
X = enc.encode(["I am so happy!"])      # shape (1, 768)
```

## What to hand over
- Participant 3: `features/.../*_X.npy` and `*_ids.json`.
- Participant 4: `encoder.py` and the `config.json` of the chosen pooling.
- Participant 1: processing time and truncated share from `config.json` -> `stats`.

## Notes
- Mean pooling averages every non-padding token, including `[CLS]` and `[SEP]`.
  Use `--exclude-special-tokens` to leave them out.
- GPU and CPU results may differ around the 5th-6th decimal; this is normal.
- Pin the model with `--revision` (commit hash from the HF Hub cache folder name).
