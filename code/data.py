"""Original window sampling, loss reduction and cosine schedule; portable imports.

Corpus intentionally accepts only train/dev. Holdout loading lives in evaluate.py.
"""
import math
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import digest, write_json

class Corpus:
    def __init__(self, root, split, context):
        if split not in ("train", "dev"):
            raise ValueError("migration trainer does not access holdout data")
        self.root, self.split, self.context = Path(root), split, context
        self.tokens = np.load(self.root / f"tokens_{split}.npy", mmap_mode="r", allow_pickle=False)
        self.docs = np.load(self.root / f"docs_{split}.npy", allow_pickle=False)
        if self.docs.ndim != 2 or self.docs.shape[1] != 2:
            raise ValueError("expected Nx2 document boundaries")
        if len(self.docs) == 0 or self.docs.min() < 0 or self.docs.max() > len(self.tokens):
            raise ValueError("invalid document boundaries")
        self.lengths = self.docs[:, 1] - self.docs[:, 0]
        if (self.lengths <= 0).any():
            raise ValueError("nonpositive document length")
        self.eligible = np.flatnonzero(self.lengths > context)
        self.windows = self.lengths[self.eligible] - context
        self.cumulative = np.cumsum(self.windows)
        if not len(self.eligible):
            raise ValueError("no sufficiently long documents")

    def schedule(self, seed, steps, batch):
        rng = np.random.default_rng(202609210000 + seed)
        flat = rng.integers(0, self.cumulative[-1], size=(steps, batch), dtype=np.int64)
        ids = np.searchsorted(self.cumulative, flat, side="right")
        previous = np.concatenate(([0], self.cumulative[:-1]))[ids]
        return self.docs[self.eligible[ids], 0] + flat - previous

    def evaluation_schedule(self, max_tokens):
        offsets, ids = [], []
        limit = max_tokens // self.context
        if limit < 1:
            raise ValueError("evaluation budget must include one complete context")
        order = np.random.default_rng(20260921).permutation(self.eligible)
        for block_index in range(16):
            for index in order:
                start, end = self.docs[index]
                offset = int(start) + block_index * self.context
                if offset + self.context >= end:
                    continue
                offsets.append(offset)
                ids.append(int(index))
                if len(offsets) >= limit:
                    return np.asarray(offsets, dtype=np.int64), np.asarray(ids, dtype=np.int64)
        return np.asarray(offsets, dtype=np.int64), np.asarray(ids, dtype=np.int64)

    def batch(self, offsets, device):
        positions = offsets[:, None] + np.arange(self.context + 1)[None, :]
        sample = torch.from_numpy(np.asarray(self.tokens[positions], dtype=np.int64)).to(device)
        return sample[:, :-1], sample[:, 1:]


@torch.no_grad()
def evaluate(model, corpus, schedule, batch, check=None):
    device = next(model.parameters()).device  # Preserve cuda:0, not merely "cuda".
    offsets, ids = schedule
    sums = []
    model.eval()
    for begin in range(0, len(offsets), batch):
        if check is not None:
            check()
        x, y = corpus.batch(offsets[begin:begin + batch], device)
        logits = model(x)
        losses = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none")
        sums.extend(losses.reshape(-1, corpus.context).sum(1).cpu().double().tolist())
    sums = np.asarray(sums, dtype=np.float64)
    counts = np.full(len(sums), corpus.context, dtype=np.int64)
    if not len(sums) or not np.isfinite(sums).all():
        raise FloatingPointError("empty or nonfinite evaluation")
    return {"nll": float(sums.sum() / counts.sum()), "tokens": int(counts.sum()),
            "blocks": len(sums), "documents": len(set(ids.tolist())),
            "loss_sums": sums, "counts": counts, "doc_ids": ids, "offsets": offsets}


def save_eval(directory, label, result):
    path = Path(directory) / f"{label}.npz"
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    summary = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    summary["arrays_sha256"] = digest(path)
    write_json(Path(directory) / f"{label}.json", summary)
    return summary


def learning_rate(step, total, base, warmup, schedule="cosine"):
    if schedule == "constant":
        return base
    if schedule != "cosine":
        raise ValueError("unknown learning-rate schedule")
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return base * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
