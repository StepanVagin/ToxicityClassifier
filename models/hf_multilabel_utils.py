"""
Shared helpers for HuggingFace multi-label toxicity training.

Includes per-label BCE pos_weight, optional label-decoupling penalty, dynamic
padding collate, layer-wise LR decay (LLRD), and scheduler step accounting for
gradient accumulation.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def compute_bce_pos_weights(
    y: np.ndarray,
    eps: float = 1.0,
    max_weight: Optional[float] = None,
) -> torch.Tensor:
    """
    Per-label positive weights N_neg / N_pos for BCEWithLogitsLoss.

    PyTorch expects ``pos_weight`` of shape ``(num_labels,)``.
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D; got shape {y.shape}")
    yf = y.astype(np.float64)
    n = yf.shape[0]
    pos = yf.sum(axis=0)
    neg = n - pos
    w = neg / np.maximum(pos, eps)
    if max_weight is not None:
        w = np.minimum(w, float(max_weight))
    return torch.tensor(w, dtype=torch.float32)


def label_correlation_matrix(y: np.ndarray) -> np.ndarray:
    """Pearson correlation between label columns (handles constant columns)."""
    yf = y.astype(np.float64)
    if yf.shape[0] < 2:
        return np.eye(yf.shape[1], dtype=np.float64)
    c = np.corrcoef(yf, rowvar=False)
    if np.ndim(c) == 0:
        return np.ones((1, 1), dtype=np.float64)
    c = np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=-1.0)
    np.fill_diagonal(c, 1.0)
    return c


def decoupling_weight_matrix(
    R: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Symmetric weights w_ij = max(0, threshold - R_ij) for i != j.

    Penalises joint predicted probability on pairs that are weakly correlated
    in the training data (e.g. threat vs toxic when r is small).
    """
    r = np.asarray(R, dtype=np.float64)
    n = r.shape[0]
    w = np.maximum(0.0, float(threshold) - r)
    np.fill_diagonal(w, 0.0)
    # Symmetrise in case of numerical drift
    w = 0.5 * (w + w.T)
    return w.astype(np.float32)


def decoupling_penalty(probs: torch.Tensor, w_upper: torch.Tensor) -> torch.Tensor:
    """
    Scalar mean over batch of sum_{i<j} w_ij * p_i * p_j.

    Parameters
    ----------
    probs : (batch, num_labels)
    w_upper : (num_labels, num_labels) on same device as probs; upper triangle
        can be zero if w is symmetric and we double-count carefully — here w is
        full symmetric with zero diagonal; we sum i<j using upper triangle only.
    """
    # Use upper triangle indices
    n = probs.shape[1]
    if w_upper.shape != (n, n):
        raise ValueError(f"w shape {w_upper.shape} != ({n}, {n})")
    iu = torch.triu_indices(n, n, offset=1)
    pi = probs[:, iu[0]]
    pj = probs[:, iu[1]]
    wij = w_upper[iu[0], iu[1]]
    per_sample = (pi * pj * wij).sum(dim=1)
    return per_sample.mean()


class ToxicityTextDataset(Dataset):
    """Stores raw text + labels; tokenisation happens in the collate function."""

    def __init__(self, texts: List[str], labels: np.ndarray) -> None:
        if labels.ndim != 2 or labels.shape[1] != 6:
            raise ValueError(f"labels must have shape (n_samples, 6); got {labels.shape}")
        self.texts = texts
        self.labels_tensor = torch.tensor(labels.astype(np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"text": self.texts[idx], "labels": self.labels_tensor[idx]}


def make_dynamic_padding_collate_fn(tokenizer: Any, max_length: int) -> Callable:
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out: Dict[str, torch.Tensor] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }
        return out

    return collate


def _encoder_layer_index_from_name(name: str) -> Optional[int]:
    m = re.search(r"\.transformer\.layer\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    m = re.search(r"\.encoder\.layer\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    m = re.search(r"albert_layers\.(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def _infer_transformer_layer_indices(model: nn.Module) -> List[int]:
    idxs: List[int] = []
    for n, _ in model.named_parameters():
        li = _encoder_layer_index_from_name(n)
        if li is not None:
            idxs.append(li)
    return sorted(set(idxs))


def build_llrd_parameter_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    gamma: float,
    head_lr_multiplier: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Layer-wise learning rate decay for HF sequence classification models.

    Higher layers (closer to the classifier) receive larger learning rates:
    ``lr(layer) = base_lr * gamma ** (max_depth - depth)``.

    DistilBERT / BERT-style models use one depth per transformer block index.
    ALBERT-base shares weights across inner repetitions; when only one physical
    ``albert_layers`` index exists, we use a 3-tier schedule (embeddings /
    encoder / head).
    """
    layer_ids = _infer_transformer_layer_indices(model)
    is_albert = any("albert" in n for n, _ in model.named_parameters())

    if not layer_ids:
        return [{"params": list(model.parameters()), "lr": base_lr, "weight_decay": weight_decay}]

    if is_albert and len(layer_ids) == 1 and layer_ids[0] == 0:
        embed_params: List[nn.Parameter] = []
        enc_params: List[nn.Parameter] = []
        head_params: List[nn.Parameter] = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "classifier" in n:
                head_params.append(p)
            elif "embeddings" in n:
                embed_params.append(p)
            else:
                enc_params.append(p)
        groups: List[Dict[str, Any]] = []
        if embed_params:
            groups.append(
                {
                    "params": embed_params,
                    "lr": base_lr * (gamma**2),
                    "weight_decay": weight_decay,
                }
            )
        if enc_params:
            groups.append(
                {
                    "params": enc_params,
                    "lr": base_lr * gamma,
                    "weight_decay": weight_decay,
                }
            )
        if head_params:
            groups.append(
                {
                    "params": head_params,
                    "lr": base_lr * head_lr_multiplier,
                    "weight_decay": weight_decay,
                }
            )
        return groups if groups else [{"params": list(model.parameters()), "lr": base_lr, "weight_decay": weight_decay}]

    L = max(layer_ids) + 1
    # depth: embeddings=0, transformer layer i => i+1, head => L+1
    max_depth = L + 1

    buckets: Dict[int, List[nn.Parameter]] = {}
    for d in range(max_depth + 1):
        buckets[d] = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "classifier" in n or "pre_classifier" in n:
            buckets[max_depth].append(p)
            continue
        li = _encoder_layer_index_from_name(n)
        if li is not None:
            buckets[li + 1].append(p)
            continue
        if "embeddings" in n:
            buckets[0].append(p)
            continue
        # RoBERTa pooler, etc.
        buckets[1].append(p)

    groups_out: List[Dict[str, Any]] = []
    for d in range(max_depth + 1):
        params = buckets[d]
        if not params:
            continue
        lr_d = base_lr * (gamma ** (max_depth - d))
        if d == max_depth:
            lr_d *= head_lr_multiplier
        groups_out.append({"params": params, "lr": lr_d, "weight_decay": weight_decay})

    if not groups_out:
        return [{"params": list(model.parameters()), "lr": base_lr, "weight_decay": weight_decay}]
    return groups_out


def count_optimizer_steps_per_epoch(num_batches: int, grad_accum_steps: int) -> int:
    """How many ``optimizer.step`` calls per epoch with HF-style accumulation."""
    if num_batches <= 0:
        return 0
    g = max(int(grad_accum_steps), 1)
    return (num_batches + g - 1) // g
