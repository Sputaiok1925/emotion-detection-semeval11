# Pretrained RoBERTa emotion detection

Model: [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions).
It was fine-tuned on GoEmotions with 28 independent outputs. We select the five
BRIGHTER emotions by their names and apply sigmoid. No model weights are updated.

Run from the project root with the Python interpreter used in PyCharm:

```powershell
$py = "C:\Users\sputa\AppData\Local\Programs\Python\Python314\python.exe"
& $py roberta/evaluate_pretrained.py
```

The first run downloads about 500 MB to the Hugging Face cache. GPU is used
automatically if CUDA is available; otherwise CPU is used. Reduce memory usage
with `--batch-size 8` if needed. The default maximum input length is 128 tokens,
matching the previous BERT experiments.

By default the script reads `data/extended/dev.csv` (115 original English texts)
and `data/extended/test.csv` (2,761 original English texts). It does not read train.
These are the common test examples used by the previous model comparison.
For the original 2,762-example test, specify `--data-dir data/processed` and a
different `--out` directory.

The output directory `reports/roberta_final` contains the dev/test scores
and IDs, thresholds selected on dev, and the final metrics/classification report.
The test is never used to select thresholds.

Check a custom text after evaluation (using saved dev thresholds):

```powershell
& $py roberta/evaluate_pretrained.py --text "I am happy to see you, but worried about tomorrow."
```

Earlier experiments compared direct labels with max, arithmetic mean and noisy-OR
aggregation using the [GoEmotions Ekman mapping](https://github.com/google-research/google-research/blob/master/goemotions/data/ekman_mapping.json).
Direct labels achieved the highest dev macro-F1 (0.6319); the selected method
achieved test macro-F1 0.5510. The threshold search includes values below 0.05
because the pretrained model assigns very low scores on BRIGHTER.
Historical results, including the fixed 0.5 baseline and aggregation comparison,
remain in `reports/roberta_pretrained`, `reports/roberta_recalibrated`,
`reports/roberta_grouped` and `reports/roberta_aggregation`.

GoEmotions and BRIGHTER have different annotation guidelines and domains, so a
good score on GoEmotions does not guarantee a good BRIGHTER score.

## Fine-tuning base RoBERTa

`finetune.py` uses the original [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base),
not the GoEmotions checkpoint. It reuses the encoder, pooling, freezing,
optimizer, prediction and threshold functions from `frozen_bert/finetune.py`.
The full encoder and a new five-output linear head are trained together on
`data/extended/train.csv`. The English dev chooses the best epoch at threshold
0.5 and then the five final thresholds; the English test is evaluated with
`evaluation.py` after training. Class balancing uses only the train labels.

For Windows with an RTX 5060 and Python 3.14, a CUDA environment has been configured
in the project directory `.venv`. On another machine, create the environment and
install the appropriate PyTorch build for its GPU. The tested setup was:

```powershell
& $py -m venv --system-site-packages .venv
& .\.venv\Scripts\python.exe -m pip install "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130
```

Use that interpreter to check GPU availability and start the long run:

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available())"
& $py -u roberta/finetune.py --device cuda
```

The default settings are 12 unfrozen layers, 4 epochs, 128 tokens, batch size 8,
gradient accumulation 4 (effective batch size 32), AdamW and mixed precision
on CUDA. If GPU memory is insufficient, try `--batch-size 4 --accumulation 8`.
If CUDA is unavailable, `--device cpu` works but full training can be much slower.
The initial checkpoint downloads automatically into the Hugging Face cache.

The best checkpoint is saved after each improving epoch to
`runs/roberta_extended_ft12_seed42/model.pt` (ignored by Git). If the program
is stopped after a checkpoint is saved, evaluate that checkpoint without retraining:

```powershell
& $py roberta/finetune.py --eval-only
```

Use `--train-only` to postpone test evaluation while comparing settings on dev.
To try three more epochs from the best saved checkpoint:

```powershell
& $py -u roberta/finetune.py --device cuda --continue-from runs/roberta_extended_ft12_seed42/model.pt --epochs 3
```

This starts a new training stage with fresh AdamW optimizer/scheduler state and
lower default learning rates (`1e-5` encoder, `2e-4` head), because the original
checkpoint contains the model weights but no optimizer state. Epoch numbering
continues from the saved best epoch. The initial checkpoint is included in the
dev comparison, so a worse additional epoch cannot replace it. New weights and
reports go to separate `*_continued` directories; original results are retained.
This is additional fine-tuning, not an exact resume of an interrupted epoch.
Reports (history, dev/test probabilities and IDs, metrics and classification report)
are saved under `reports/roberta_finetune/roberta_extended_ft12_seed42/`.
To test your own text after training, use the existing checkpoint evaluator:

```powershell
& $py src/evaluate_finetune.py --checkpoint runs/roberta_extended_ft12_seed42/model.pt --text "I am so happy but worried about tomorrow."
```

For comparison on the original English train, specify `--data-dir data/processed`;
this uses a separate run name and the original 2,762-example test. Compare scores
on matching IDs when comparing with the 2,761-example common test.
