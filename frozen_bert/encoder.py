"""Frozen BERT: text -> vector (participant 2).

Usage (the same call is used for training and in the app of participant 4):

    from encoder import FrozenBertEncoder, EncoderConfig
    enc = FrozenBertEncoder(EncoderConfig(pooling="mean"))
    X = enc.encode(["I am happy you came!", "Leave me alone."])   # shape (2, 768)

BERT weights are never updated: the model is in eval mode, every parameter has
requires_grad=False, and all computations run under torch.inference_mode().
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

# Supported ways to turn per-token vectors into one vector per text
POOLINGS = ("mean", "cls")


@dataclass
class EncoderConfig:
    """All settings that influence the resulting vectors."""

    model_name: str = "bert-base-uncased"
    revision: Optional[str] = None      # commit hash on the HF Hub; pin it for reproducibility
    pooling: str = "mean"               # "mean" | "cls"
    max_length: int = 128               # max tokens per text; longer texts are truncated
    batch_size: int = 32                # texts processed at once
    device: str = "auto"                # "auto" | "cpu" | "cuda"
    # Rule for special tokens in mean pooling:
    #   False -> average every token except padding (includes [CLS] and [SEP])
    #   True  -> average only ordinary tokens (no [CLS], [SEP], padding)
    exclude_special_tokens: bool = False

    def __post_init__(self):
        # Fail early on typos instead of producing strange results later
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {self.pooling!r}")
        if self.max_length < 3:  # room for [CLS], at least one token, [SEP]
            raise ValueError("max_length must be >= 3")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


class FrozenBertEncoder:
    def __init__(self, config: Optional[EncoderConfig] = None):
        self.config = config or EncoderConfig()
        c = self.config

        # Pick the device: GPU if available, otherwise CPU
        if c.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(c.device)

        # The tokenizer must match the model, so both are loaded by the same name
        self.tokenizer = AutoTokenizer.from_pretrained(c.model_name, revision=c.revision)
        self.model = AutoModel.from_pretrained(c.model_name, revision=c.revision)
        self.model.to(self.device)
        self.model.eval()                       # disables dropout -> deterministic output
        for p in self.model.parameters():       # freeze: no gradients, weights never change
            p.requires_grad_(False)

        self.dim: int = int(self.model.config.hidden_size)   # 768 for bert-base

        # Exact model version, recorded in config.json.
        # Newer transformers versions may not expose _commit_hash, so fall back
        # to the user-supplied revision, then to the HF cache folder name.
        rev = getattr(self.model.config, "_commit_hash", None) or c.revision
        if rev is None:
            try:
                from huggingface_hub import try_to_load_from_cache
                p = try_to_load_from_cache(c.model_name, "config.json")
                if isinstance(p, str):
                    rev = Path(p).parent.name   # cache path is .../snapshots/<hash>/config.json
            except Exception:
                pass
        self.resolved_revision: Optional[str] = rev

        # Statistics of the last encode / encode_all call
        self.last_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def describe(self) -> dict:
        """Everything needed to reproduce the features and to link a classifier to them."""
        d = asdict(self.config)
        d.update(
            resolved_revision=self.resolved_revision,
            dim=self.dim,
            device_used=str(self.device),
            torch=torch.__version__,
            transformers=transformers.__version__,
            do_lower_case=getattr(self.tokenizer, "do_lower_case", None),
            tokenizer_class=type(self.tokenizer).__name__,
        )
        return d

    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_texts(texts) -> List[str]:
        # A bare string would be silently split into characters, so reject it
        if isinstance(texts, str):
            raise TypeError("encode() expects a list of texts, not a single string")
        texts = list(texts)
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"text #{i} is {type(t).__name__}, expected str")
        return texts

    def _pool_batch(self, hidden, attention_mask, special_mask) -> Dict[str, torch.Tensor]:
        """hidden: (B, L, D) per-token vectors -> one (B, D) vector per text, for both poolings."""
        out = {}
        # CLS pooling: vector of the first token ([CLS]) from the last layer
        out["cls"] = hidden[:, 0, :]

        # Mean pooling: average over real tokens only (padding is masked out)
        mask = attention_mask.clone()
        if self.config.exclude_special_tokens:
            mask = mask * (1 - special_mask)    # also drop [CLS] and [SEP]
        m = mask.unsqueeze(-1).to(hidden.dtype)             # (B, L) -> (B, L, 1)
        summed = (hidden * m).sum(dim=1)                    # padding contributes zeros
        counts = m.sum(dim=1).clamp(min=1.0)                # avoid division by zero
        out["mean"] = summed / counts
        return out

    @torch.inference_mode()
    def encode_all(self, texts: Sequence[str], show_progress: bool = False) -> Dict[str, np.ndarray]:
        """One BERT pass -> both vector types: {"mean": N x D, "cls": N x D}.
        Row order always equals the order of the input texts."""
        c = self.config
        texts = self._check_texts(texts)
        n = len(texts)
        res = {p: np.zeros((n, self.dim), dtype=np.float32) for p in POOLINGS}
        if n == 0:
            self.last_stats = dict(n_texts=0, n_truncated=0, truncated_frac=0.0,
                                   seconds=0.0, texts_per_sec=0.0)
            return res

        t0 = time.perf_counter()

        # Token lengths WITHOUT truncation -> share of texts that will be cut
        full_ids = self.tokenizer(texts, add_special_tokens=True, truncation=False,
                                  padding=False)["input_ids"]
        lengths = np.array([len(x) for x in full_ids])
        n_truncated = int((lengths > c.max_length).sum())

        # Sort by length so a batch contains similar lengths (less padding, faster).
        # Results are written back to the original positions, so order is preserved.
        order = np.argsort(lengths, kind="stable")

        for start in range(0, n, c.batch_size):
            idx = order[start:start + c.batch_size]         # original indices of this batch
            batch = [texts[i] for i in idx]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=c.max_length, return_tensors="pt",
                                 return_special_tokens_mask=True)
            # The special-token map is only for pooling; the model does not accept it
            special = enc.pop("special_tokens_mask").to(self.device)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            hidden = self.model(**enc).last_hidden_state    # (B, L, D)
            pooled = self._pool_batch(hidden, enc["attention_mask"], special)
            for p in POOLINGS:
                res[p][idx] = pooled[p].float().cpu().numpy()
            if show_progress:
                print(f"\r  {min(start + c.batch_size, n)}/{n}", end="", flush=True)
        if show_progress:
            print()

        dt = time.perf_counter() - t0
        self.last_stats = dict(
            n_texts=n,
            n_truncated=n_truncated,
            truncated_frac=n_truncated / n,
            seconds=dt,
            texts_per_sec=n / dt if dt > 0 else float("inf"),
        )
        return res

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """List of texts -> matrix N x D (pooling taken from config.pooling).
        This is the ONLY function used by both training and the app, so the
        preprocessing is guaranteed to be identical."""
        return self.encode_all(texts, show_progress)[self.config.pooling]
