#!/usr/bin/env python3
"""
Tensor decomposition for transformer Linear layers.

Replaces selected ``nn.Linear(in_features -> out_features)`` layers with a
low-rank factorization ``Linear(in -> r) -> Linear(r -> out)`` via truncated
SVD of the original weight matrix. Bias is preserved on the second Linear.

Typical targets
---------------
- DistilBERT: ``ffn.lin1``, ``ffn.lin2``, ``attention.q_lin/k_lin/v_lin/out_lin``
- ALBERT:     ``ffn``, ``ffn_output``, ``attention.query/key/value/dense``

Config schema
-------------
``config["tensor_decomposition"]``::

    {
        "enabled":        true,
        "method":         "svd",
        "rank_ratio":     0.5,                          # r = max(1, rank_ratio * min(in, out))
        "rank":           null,                         # explicit r; overrides rank_ratio
        "energy":         null,                         # e.g. 0.9 — pick smallest r whose cum. singular-value energy >= this; overrides rank_ratio
        "target_patterns": ["ffn.lin1", "ffn.lin2"],    # substring match on fully-qualified module name
        "exclude_patterns": [],
        "min_params":     0                             # skip layers with in*out below this
    }

Save/load
---------
``save_td_config(save_dir, td_applied)`` writes ``td_config.json`` alongside
the HF model directory so :func:`apply_tensor_decomposition` can be replayed
before ``from_pretrained`` on reload.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


TD_CONFIG_FILENAME = "td_config.json"


class LowRankLinear(nn.Module):
    """Two-Linear low-rank replacement: ``x -> A -> B(+bias)``.

    Equivalent semantics to ``nn.Linear(in_features, out_features, bias)`` with
    weight approximated by ``B.weight @ A.weight`` (rank r).
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x))


def _resolve_rank(
    in_features: int,
    out_features: int,
    singular_values: torch.Tensor,
    td_cfg: Dict[str, Any],
) -> int:
    """Pick rank from explicit ``rank``, ``energy`` threshold, or ``rank_ratio``."""
    full = min(in_features, out_features)
    explicit = td_cfg.get("rank")
    if explicit is not None:
        return max(1, min(int(explicit), full))

    energy = td_cfg.get("energy")
    if energy is not None:
        sq = singular_values.pow(2)
        total = float(sq.sum().item())
        if total <= 0:
            return full
        cum = torch.cumsum(sq, dim=0) / total
        target = float(energy)
        idx = int(torch.searchsorted(cum, torch.tensor(target)).item()) + 1
        return max(1, min(idx, full))

    ratio = float(td_cfg.get("rank_ratio", 0.5))
    return max(1, min(int(round(ratio * full)), full))


def _svd_factorize(
    linear: nn.Linear, td_cfg: Dict[str, Any]
) -> Tuple[LowRankLinear, int]:
    """Truncated-SVD factorize a ``nn.Linear`` into ``LowRankLinear``."""
    W = linear.weight.data  # shape (out, in)
    out_features, in_features = W.shape

    W_cpu = W.detach().to(dtype=torch.float32, device="cpu")
    U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)

    rank = _resolve_rank(in_features, out_features, S, td_cfg)

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    # W ≈ (U_r * sqrt(S_r)) @ (sqrt(S_r) * Vh_r)
    s_sqrt = torch.sqrt(S_r)
    B_weight = U_r * s_sqrt.unsqueeze(0)           # (out, r)
    A_weight = Vh_r * s_sqrt.unsqueeze(1)          # (r, in)

    has_bias = linear.bias is not None
    lr = LowRankLinear(in_features, out_features, rank, bias=has_bias)
    dtype = linear.weight.dtype
    device = linear.weight.device
    lr.A.weight.data = A_weight.to(device=device, dtype=dtype).contiguous()
    lr.B.weight.data = B_weight.to(device=device, dtype=dtype).contiguous()
    if has_bias:
        lr.B.bias.data = linear.bias.data.detach().clone()
    return lr, rank


def _match(name: str, patterns: List[str]) -> bool:
    return any(p in name for p in patterns) if patterns else False


def _iter_linear_targets(
    model: nn.Module, td_cfg: Dict[str, Any]
) -> List[Tuple[str, nn.Linear]]:
    targets = td_cfg.get("target_patterns") or []
    excludes = td_cfg.get("exclude_patterns") or []
    min_params = int(td_cfg.get("min_params", 0) or 0)

    out: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _match(name, targets):
            continue
        if _match(name, excludes):
            continue
        n_params = module.in_features * module.out_features
        if n_params < min_params:
            continue
        out.append((name, module))
    return out


def _set_submodule(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    """Replace ``root.a.b.c`` with ``new_module``."""
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def apply_tensor_decomposition(
    model: nn.Module, td_cfg: Dict[str, Any], verbose: bool = True
) -> Dict[str, Any]:
    """Apply tensor decomposition in-place to ``model``.

    Parameters
    ----------
    model : nn.Module
        The (loaded) transformer model. Mutated in place.
    td_cfg : dict
        ``config["tensor_decomposition"]`` block.
    verbose : bool
        Print a per-layer summary and overall parameter savings.

    Returns
    -------
    dict
        Report with keys ``method``, ``replacements`` (list of per-layer info),
        ``params_before`` / ``params_after`` (targeted layers only).
    """
    if not td_cfg or not td_cfg.get("enabled", False):
        return {"method": None, "replacements": [], "params_before": 0, "params_after": 0}

    method = str(td_cfg.get("method", "svd")).lower()
    if method != "svd":
        raise NotImplementedError(
            f"tensor_decomposition.method={method!r} not supported (only 'svd')."
        )

    targets = _iter_linear_targets(model, td_cfg)
    if not targets:
        if verbose:
            print("apply_tensor_decomposition: no layers matched target_patterns; skipping.")
        return {"method": method, "replacements": [], "params_before": 0, "params_after": 0}

    replacements: List[Dict[str, Any]] = []
    params_before = 0
    params_after = 0

    for name, linear in targets:
        in_f, out_f = linear.in_features, linear.out_features
        orig = in_f * out_f
        lr, rank = _svd_factorize(linear, td_cfg)
        _set_submodule(model, name, lr)
        new = rank * (in_f + out_f)
        params_before += orig
        params_after += new
        replacements.append(
            {"name": name, "in_features": in_f, "out_features": out_f,
             "rank": rank, "params_before": orig, "params_after": new}
        )
        if verbose:
            pct = (1.0 - new / orig) * 100.0 if orig else 0.0
            print(f"  [TD] {name:<60} {in_f}x{out_f} -> rank {rank}  "
                  f"(-{pct:.1f}% params)")

    if verbose and params_before:
        pct = (1.0 - params_after / params_before) * 100.0
        print(f"  [TD] targeted layers: {params_before:,} -> {params_after:,} params "
              f"(-{pct:.1f}%)")

    return {
        "method": method,
        "replacements": replacements,
        "params_before": params_before,
        "params_after": params_after,
    }


def save_td_config(save_dir: str, td_cfg: Dict[str, Any]) -> None:
    """Persist the decomposition config next to the HF model directory.

    Only written if decomposition was actually applied (``enabled=True``).
    """
    if not td_cfg or not td_cfg.get("enabled", False):
        return
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, TD_CONFIG_FILENAME), "w", encoding="utf-8") as f:
        json.dump(td_cfg, f, indent=2)


def load_hf_state_dict(load_dir: str) -> Dict[str, torch.Tensor]:
    """Load a HuggingFace-saved state dict from ``model.safetensors`` or ``pytorch_model.bin``."""
    safetensors_path = os.path.join(load_dir, "model.safetensors")
    if os.path.isfile(safetensors_path):
        from safetensors.torch import load_file
        return load_file(safetensors_path)
    bin_path = os.path.join(load_dir, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(
        f"No model weights found in {load_dir!r} "
        f"(looked for model.safetensors and pytorch_model.bin)."
    )


def load_td_config(load_dir: str) -> Optional[Dict[str, Any]]:
    """Return the decomposition config if one was saved, else ``None``."""
    path = os.path.join(load_dir, TD_CONFIG_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
