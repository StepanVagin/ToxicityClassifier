#!/usr/bin/env python3
"""
Resource-efficiency benchmarking for toxicity classification models.

Measures, for each (model, device) pair:
  - Inference throughput (samples/sec) at batch sizes 1, 16, 64
  - Disk footprint (serialized model size in MB)
  - Peak memory during inference (CUDA peak alloc, or process RSS on CPU)
  - Training wall-clock + GPU-hours per epoch (read from {model}_metadata.json)
  - Energy proxy (GPU-hours x TDP, in Wh)

Outputs JSON + a LaTeX snippet ready to drop into the report.

Usage:
    # All models, default devices (LR=cpu, ALBERT/DistilBERT=cuda+cpu)
    python scripts/benchmark.py

    # Single model on cpu only
    python scripts/benchmark.py --models albert --devices cpu

    # Allow HF backbones to fall back to fresh pretrained weights when no
    # fine-tuned checkpoint is on disk (timing is identical to a trained model).
    python scripts/benchmark.py --use-pretrained
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from models import create_model  # noqa: E402


MODELS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "config": "logistic_regression.json",
        "default_devices": ["cpu"],
        # CPU TDP for the energy proxy when LR runs on CPU.
        "tdp_w": {"cpu": 65, "cuda": 170},
    },
    "albert": {
        "config": "albert.json",
        "default_devices": ["cuda", "cpu"],
        "tdp_w": {"cpu": 65, "cuda": 170},
    },
    "distilbert": {
        "config": "distilbert.json",
        "default_devices": ["cuda", "cpu"],
        "tdp_w": {"cpu": 65, "cuda": 170},
    },
}


def load_merged_config(model_type: str) -> dict:
    base = PROJECT_ROOT / "configs" / "base_config.json"
    model = PROJECT_ROOT / "configs" / MODELS[model_type]["config"]
    with open(base) as f:
        cfg = json.load(f)
    with open(model) as f:
        cfg.update(json.load(f))
    cfg["model_type"] = model_type
    return cfg


def load_sample_texts(n: int) -> List[str]:
    """Pull ``n`` real comments from train.csv if present, else synthesize."""
    csv = PROJECT_ROOT / "data" / "processed" / "train.csv"
    if csv.exists():
        try:
            import pandas as pd

            df = pd.read_csv(csv, usecols=["comment_text"], nrows=max(n, 1))
            texts = [str(t) for t in df["comment_text"].tolist()]
            if len(texts) >= n:
                return texts[:n]
            # Repeat to fill if dataset is smaller than requested
            out = []
            while len(out) < n:
                out.extend(texts)
            return out[:n]
        except Exception as e:
            print(f"  (could not read train.csv: {e}; falling back to synthetic)")

    return [
        f"sample comment number {i} used only for benchmarking inference latency."
        for i in range(n)
    ]


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def synchronize(device: str) -> None:
    if device == "cuda" and cuda_available():
        import torch

        torch.cuda.synchronize()


def reset_peak_memory(device: str) -> None:
    if device == "cuda" and cuda_available():
        import torch

        torch.cuda.reset_peak_memory_stats()
    # CPU peak isn't trivially resettable; we just sample RSS at the end.


def peak_memory_mb(device: str) -> Optional[float]:
    if device == "cuda" and cuda_available():
        import torch

        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 ** 2))
    except ImportError:
        return None


def benchmark_throughput(
    model,
    texts: List[str],
    batch_size: int,
    device: str,
    n_warmup: int,
    min_seconds: float,
) -> Dict[str, Any]:
    """Run inference repeatedly and report samples/sec for one batch size."""
    n = len(texts)
    if n == 0 or batch_size <= 0:
        return {"batch_size": batch_size, "error": "no samples or invalid batch size"}

    batches = [texts[i : i + batch_size] for i in range(0, n, batch_size)]

    # Warmup (not timed)
    for i in range(min(n_warmup, len(batches))):
        _ = model.predict_proba(batches[i])
    synchronize(device)

    # Measurement loop: keep iterating until min_seconds elapsed
    start = time.perf_counter()
    n_processed = 0
    n_batches = 0
    while True:
        for batch in batches:
            _ = model.predict_proba(batch)
            n_processed += len(batch)
            n_batches += 1
            if time.perf_counter() - start >= min_seconds:
                break
        if time.perf_counter() - start >= min_seconds:
            break
    synchronize(device)
    elapsed = time.perf_counter() - start

    return {
        "batch_size": batch_size,
        "samples_per_sec": n_processed / elapsed if elapsed > 0 else float("nan"),
        "ms_per_batch": (elapsed / n_batches) * 1000.0 if n_batches else float("nan"),
        "n_samples_processed": n_processed,
        "elapsed_seconds": elapsed,
    }


def load_training_metadata(model_type: str) -> Dict[str, Any]:
    p = PROJECT_ROOT / "models" / "saved" / f"{model_type}_metadata.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def saved_artifact_exists(model_type: str) -> bool:
    pkl = PROJECT_ROOT / "models" / "saved" / f"{model_type}.pkl"
    hf_dir = PROJECT_ROOT / "models" / "saved" / model_type
    return pkl.is_file() or hf_dir.is_dir()


def benchmark_one(
    model_type: str,
    device: str,
    batch_sizes: List[int],
    num_samples: int,
    n_warmup: int,
    min_seconds: float,
    use_pretrained: bool,
) -> Dict[str, Any]:
    print(f"\n=== Benchmarking {model_type} on {device} ===")

    if device == "cuda" and not cuda_available():
        msg = "CUDA requested but torch.cuda.is_available() is False."
        print(f"  Skipping: {msg}")
        return {"model_type": model_type, "device": device, "error": msg}

    cfg = load_merged_config(model_type)
    cfg.setdefault("training", {})["device"] = device
    cfg.setdefault("training", {})["show_progress"] = False

    has_saved = saved_artifact_exists(model_type)
    if not has_saved:
        if model_type == "logistic_regression":
            msg = (
                "No saved logistic regression checkpoint found at "
                "models/saved/logistic_regression.pkl. Train it first."
            )
            print(f"  Skipping: {msg}")
            return {"model_type": model_type, "device": device, "error": msg}
        if not use_pretrained:
            msg = (
                f"No saved checkpoint at models/saved/{model_type}/. "
                "Pass --use-pretrained to time the architecture using fresh HF base weights."
            )
            print(f"  Skipping: {msg}")
            return {"model_type": model_type, "device": device, "error": msg}

    # Build / load
    try:
        model = create_model(cfg)
        if has_saved:
            pkl = str(PROJECT_ROOT / "models" / "saved" / f"{model_type}.pkl")
            model.load(pkl)
    except Exception as e:
        msg = f"Failed to instantiate/load model: {e}"
        print(f"  Skipping: {msg}")
        return {"model_type": model_type, "device": device, "error": msg}

    texts = load_sample_texts(num_samples)
    reset_peak_memory(device)

    # Throughput sweep
    throughput: List[Dict[str, Any]] = []
    for bs in batch_sizes:
        try:
            r = benchmark_throughput(model, texts, bs, device, n_warmup, min_seconds)
            print(
                f"  bs={bs:>3}: {r.get('samples_per_sec', float('nan')):8.1f} samples/s  "
                f"({r.get('ms_per_batch', float('nan')):7.2f} ms/batch)"
            )
            throughput.append(r)
        except Exception as e:
            print(f"  bs={bs}: error {e}")
            throughput.append({"batch_size": bs, "error": str(e)})

    peak_mem = peak_memory_mb(device)

    try:
        disk_mb = float(model.get_model_size())
    except Exception as e:
        print(f"  could not measure disk size: {e}")
        disk_mb = float("nan")

    # Training cost from saved metadata
    meta = load_training_metadata(model_type)
    training_time = meta.get("training_time_seconds")
    history = (meta.get("training_history") or {})
    epochs_run = len(history.get("epoch", []))
    if not epochs_run:
        epochs_run = int(cfg.get("hyperparameters", {}).get("epochs", 1) or 1)

    tdp_w = MODELS[model_type]["tdp_w"].get(device, 170)
    gpu_hours_total = None
    gpu_hours_per_epoch = None
    energy_wh = None
    if isinstance(training_time, (int, float)) and training_time > 0:
        gpu_hours_total = training_time / 3600.0
        gpu_hours_per_epoch = gpu_hours_total / max(epochs_run, 1)
        energy_wh = gpu_hours_total * tdp_w

    # Cleanup so the next run isn't poisoned
    del model
    gc.collect()
    if cuda_available():
        import torch

        torch.cuda.empty_cache()

    return {
        "model_type": model_type,
        "device": device,
        "num_samples": num_samples,
        "weights_source": "saved_checkpoint" if has_saved else "huggingface_base",
        "throughput": throughput,
        "disk_size_mb": disk_mb,
        "peak_memory_mb": peak_mem,
        "training": {
            "training_time_seconds": training_time,
            "epochs_run": epochs_run,
            "gpu_hours_total": gpu_hours_total,
            "gpu_hours_per_epoch": gpu_hours_per_epoch,
            "tdp_w": tdp_w,
            "energy_proxy_wh": energy_wh,
        },
    }


def _fmt(v: Any, prec: int = 2) -> str:
    if v is None:
        return "{--}"
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "{--}"
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}"
    return str(v)


def write_latex_snippet(results: List[Dict[str, Any]], out_path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for r in results:
        if "error" in r:
            continue
        bs_to_sps = {
            entry["batch_size"]: entry.get("samples_per_sec", float("nan"))
            for entry in r["throughput"]
            if isinstance(entry.get("batch_size"), int)
        }
        train = r.get("training") or {}
        rows.append(
            {
                "name": f"{r['model_type']} ({r['device']})",
                "sps1": bs_to_sps.get(1, float("nan")),
                "sps16": bs_to_sps.get(16, float("nan")),
                "sps64": bs_to_sps.get(64, float("nan")),
                "disk_mb": r.get("disk_size_mb", float("nan")),
                "peak_mb": r.get("peak_memory_mb"),
                "train_s": train.get("training_time_seconds"),
                "gh_per_epoch": train.get("gpu_hours_per_epoch"),
                "energy_wh": train.get("energy_proxy_wh"),
            }
        )

    lines: List[str] = []
    lines.append(r"\subsection*{Resource-Efficiency Benchmarking}")
    lines.append("")
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Inference throughput (samples/sec) at batch sizes 1, 16, 64.}"
    )
    lines.append(r"  \label{tab:throughput}")
    lines.append(r"  \begin{tabular}{l r r r}")
    lines.append(r"    \toprule")
    lines.append(r"    Model (device) & bs=1 & bs=16 & bs=64 \\")
    lines.append(r"    \midrule")
    for row in rows:
        lines.append(
            f"    {row['name']} & {_fmt(row['sps1'], 1)} & "
            f"{_fmt(row['sps16'], 1)} & {_fmt(row['sps64'], 1)} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Model footprint, training cost, and energy proxy "
        r"(GPU-hours $\times$ TDP).}"
    )
    lines.append(r"  \label{tab:footprint}")
    lines.append(r"  \begin{tabular}{l r r r r r}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Model (device) & Disk (MB) & Peak Mem (MB) & Train (s) & GPU-h/epoch & Energy (Wh) \\"
    )
    lines.append(r"    \midrule")
    for row in rows:
        lines.append(
            f"    {row['name']} & {_fmt(row['disk_mb'], 1)} & "
            f"{_fmt(row['peak_mb'], 1)} & {_fmt(row['train_s'], 1)} & "
            f"{_fmt(row['gh_per_epoch'], 4)} & {_fmt(row['energy_wh'], 2)} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nLaTeX snippet written to: {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Resource-efficiency benchmark for trained toxicity models."
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
    )
    p.add_argument(
        "--devices",
        nargs="+",
        default=None,
        choices=["cpu", "cuda"],
        help="Override the device list. Default: per-model natural set "
        "(LR=cpu, ALBERT/DistilBERT=cuda+cpu).",
    )
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 64])
    p.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="Number of texts drawn from train.csv for the throughput loop.",
    )
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument(
        "--min-seconds",
        type=float,
        default=3.0,
        help="Minimum measurement window per (model, device, batch-size).",
    )
    p.add_argument(
        "--use-pretrained",
        action="store_true",
        help="When a fine-tuned checkpoint is missing for an HF model, fall back "
        "to fresh pretrained base weights for timing (architecture-only).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "benchmarks"),
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for model_type in args.models:
        device_list = args.devices or MODELS[model_type]["default_devices"]
        for device in device_list:
            r = benchmark_one(
                model_type=model_type,
                device=device,
                batch_sizes=args.batch_sizes,
                num_samples=args.num_samples,
                n_warmup=args.n_warmup,
                min_seconds=args.min_seconds,
                use_pretrained=args.use_pretrained,
            )
            results.append(r)

    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to: {json_path}")

    write_latex_snippet(results, out_dir / "benchmark_snippet.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
