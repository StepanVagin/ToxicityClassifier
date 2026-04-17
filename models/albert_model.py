#!/usr/bin/env python3
"""
ALBERT-base-v2 model for multilabel toxicity classification.

Implements ModelABC interface for compatibility with trainer.py and run_pipeline.py.
Uses HuggingFace transformers with BCEWithLogitsLoss for multi-label classification.

Checkpoint loading (e.g. ``albert-base-v2``)
--------------------------------------------
Pretrained ALBERT weights are from **masked language modeling** (MLM). When Hugging Face loads
them into ``AlbertForSequenceClassification``, you may see a load report like:

- **UNEXPECTED** ``predictions.*`` — the old MLM head; it is not used for classification and
  can be ignored.
- **MISSING** ``classifier.*`` — the classification head did not exist in the checkpoint; it is
  **initialized at random** and must be learned during fine-tuning. This is normal before/while
  you train on your labels.

We pass ``ignore_mismatched_sizes=True`` so partial weight loading is explicit and quieter.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .abc_model import ModelABC
from .tensor_decomposition import (
    apply_tensor_decomposition,
    load_hf_state_dict as _load_hf_state_dict,
    load_td_config,
    save_td_config,
)
from .hf_multilabel_utils import (
    ToxicityTextDataset,
    build_llrd_parameter_groups,
    compute_bce_pos_weights,
    count_optimizer_steps_per_epoch,
    decoupling_penalty,
    decoupling_weight_matrix,
    label_correlation_matrix,
    make_dynamic_padding_collate_fn,
)


def _resolve_torch_device(train_cfg: dict) -> torch.device:
    """
    Pick compute device from config ``training.device``.

    - ``auto`` (default): CUDA if ``torch.cuda.is_available()`` else CPU.
    - ``cuda`` / ``gpu``: require CUDA or raise (fails fast instead of silent CPU).
    - ``cpu``: CPU only.
    - Any other string is passed to ``torch.device`` (e.g. ``cuda:0``).
    """
    spec = str(train_cfg.get("device", "auto")).strip().lower()
    if spec in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "training.device is 'cuda' but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch (see https://pytorch.org/get-started/locally/)."
            )
        return torch.device("cuda")
    if spec == "cpu":
        return torch.device("cpu")
    return torch.device(spec)


class ToxicityDataset(Dataset):
    """PyTorch Dataset for toxicity classification with tokenized texts."""

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = 128,
    ) -> None:
        """
        Parameters
        ----------
        texts : List[str]
            Cleaned text samples.
        labels : np.ndarray
            Binary labels of shape (n_samples, 6).
        tokenizer
            HuggingFace tokenizer.
        max_length : int
            Maximum sequence length for truncation/padding.
        """
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
        )

        self.input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(encodings["attention_mask"], dtype=torch.long)
        self.labels_tensor = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels_tensor[idx],
        }


class ALBERTModel(ModelABC):
    """
    ALBERT-base-v2 for multilabel toxicity classification.
    Implements ModelABC for plug-and-play with trainer and experiment runner.
    """

    def __init__(self, config: dict) -> None:
        """
        Parameters
        ----------
        config : dict
            Merged config (base_config + albert.json).
        """
        self.config = config
        self.labels = config.get("data", {}).get("labels", [])

        prep_cfg = config.get("preprocessing", {})
        self.max_length = prep_cfg.get("max_length", 128)

        hp = config.get("hyperparameters", {})
        self.learning_rate = hp.get("learning_rate", 2e-5)
        self.batch_size = hp.get("batch_size", 32)
        self.epochs = hp.get("epochs", 3)
        self.warmup_steps = hp.get("warmup_steps", 500)
        self.weight_decay = hp.get("weight_decay", 0.01)

        train_cfg = config.get("training", {})
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        self.fp16 = train_cfg.get("fp16", False)
        self.early_stopping_patience = train_cfg.get("early_stopping_patience", 2)
        self.show_progress = train_cfg.get("show_progress", True)

        model_name = config.get("model_name", "albert-base-v2")

        self.device = _resolve_torch_device(train_cfg)
        print(f"ALBERTModel: using device {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(model_name)
        model_config.num_labels = 6
        model_config.problem_type = "multi_label_classification"

        # Backbone from MLM checkpoint; classifier head is new (see module docstring).
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=model_config,
            ignore_mismatched_sizes=True,
        )

        self._tensor_decomposition_config: Dict[str, Any] = dict(
            config.get("tensor_decomposition") or {}
        )
        if self._tensor_decomposition_config.get("enabled", False):
            apply_tensor_decomposition(
                self.model, self._tensor_decomposition_config, verbose=True
            )

        self.model.to(self.device)

        self._last_save_path: Optional[str] = None
        self.training_history: Dict[str, List] = {}

    @property
    def tensor_decomposition_config(self) -> Dict[str, Any]:
        """Active tensor-decomposition config (empty if disabled)."""
        return dict(self._tensor_decomposition_config)

    def train(self, train_data: Tuple, val_data: Optional[Tuple] = None) -> None:
        """
        Train ALBERT on train_data with optional validation and early stopping.

        Parameters
        ----------
        train_data : (X_train, y_train)
            X_train: List[str] of cleaned texts
            y_train: np.ndarray of shape (n_samples, 6) binary labels
        val_data : (X_val, y_val) or None
            Same format as train_data. Used for validation and early stopping.
        """
        X_train, y_train = train_data
        if not isinstance(y_train, np.ndarray):
            y_train = np.asarray(y_train, dtype=np.float32)
        else:
            y_train = y_train.astype(np.float32)

        train_cfg = self.config.get("training", {})
        dynamic_padding = bool(train_cfg.get("dynamic_padding", False))
        grad_accum = max(int(train_cfg.get("gradient_accumulation_steps", 1)), 1)
        use_pos_weight = bool(train_cfg.get("use_pos_weight", False))
        pos_weight_max = train_cfg.get("pos_weight_max")
        use_llrd = bool(train_cfg.get("use_llrd", False))
        llrd_gamma = float(train_cfg.get("llrd_decay", 0.9))
        head_lr_mult = float(train_cfg.get("head_lr_multiplier", 1.0))
        decouple_w = float(train_cfg.get("label_decouple_weight", 0.0) or 0.0)
        decouple_tau = float(train_cfg.get("label_decouple_correlation_threshold", 0.2))

        if dynamic_padding:
            train_dataset: Dataset = ToxicityTextDataset(X_train, y_train)
            collate_fn = make_dynamic_padding_collate_fn(self.tokenizer, self.max_length)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=(self.device.type == "cuda"),
                collate_fn=collate_fn,
            )
        else:
            train_dataset = ToxicityDataset(
                X_train, y_train, self.tokenizer, max_length=self.max_length
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=(self.device.type == "cuda"),
            )

        val_loader = None
        if val_data is not None:
            X_val, y_val = val_data
            if not isinstance(y_val, np.ndarray):
                y_val = np.asarray(y_val, dtype=np.float32)
            else:
                y_val = y_val.astype(np.float32)
            val_dataset = ToxicityDataset(
                X_val, y_val, self.tokenizer, max_length=self.max_length
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )

        if use_llrd:
            param_groups = build_llrd_parameter_groups(
                self.model,
                base_lr=self.learning_rate,
                weight_decay=self.weight_decay,
                gamma=llrd_gamma,
                head_lr_multiplier=head_lr_mult,
            )
            optimizer = torch.optim.AdamW(param_groups)
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        n_train_batches = len(train_loader)
        steps_per_epoch = count_optimizer_steps_per_epoch(n_train_batches, grad_accum)
        total_steps = max(steps_per_epoch * self.epochs, 1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        if use_pos_weight:
            pw = compute_bce_pos_weights(
                y_train,
                max_weight=float(pos_weight_max) if pos_weight_max is not None else None,
            ).to(self.device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            criterion = torch.nn.BCEWithLogitsLoss()

        decouple_matrix: Optional[torch.Tensor] = None
        if decouple_w > 0.0:
            r = label_correlation_matrix(y_train)
            w_np = decoupling_weight_matrix(r, decouple_tau)
            decouple_matrix = torch.tensor(w_np, dtype=torch.float32, device=self.device)

        scaler = torch.amp.GradScaler("cuda") if self.fp16 and self.device.type == "cuda" else None

        self.training_history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_f1_macro": [],
            "val_f1_micro": [],
            "val_f1_weighted": [],
            "learning_rate": [],
            "batch_step": [],
            "batch_train_loss": [],
            "epoch_boundaries": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0
        global_batch_step = 0

        try:
            from tqdm.auto import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            pbar = None
            train_iter = train_loader
            if self.show_progress and _tqdm is not None:
                pbar = _tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{self.epochs}",
                    leave=True,
                    unit="batch",
                )
                train_iter = pbar

            for batch_idx, batch in enumerate(train_iter, start=1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                if self.fp16 and scaler is not None:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        logits = outputs.logits
                        bce = criterion(logits, labels)
                        if decouple_matrix is not None:
                            extra = decoupling_penalty(
                                torch.sigmoid(logits), decouple_matrix
                            )
                        else:
                            extra = logits.new_tensor(0.0)
                        loss = (bce + decouple_w * extra) / grad_accum
                    scaler.scale(loss).backward()
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = outputs.logits
                    bce = criterion(logits, labels)
                    if decouple_matrix is not None:
                        extra = decoupling_penalty(
                            torch.sigmoid(logits), decouple_matrix
                        )
                    else:
                        extra = logits.new_tensor(0.0)
                    loss = (bce + decouple_w * extra) / grad_accum
                    loss.backward()

                lv = float((bce + decouple_w * extra).detach().item())
                epoch_loss += lv
                global_batch_step += 1
                self.training_history["batch_step"].append(global_batch_step)
                self.training_history["batch_train_loss"].append(lv)

                should_step = (batch_idx % grad_accum == 0) or (batch_idx == n_train_batches)
                if should_step:
                    if self.fp16 and scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if pbar is not None:
                    pbar.set_postfix(loss=f"{epoch_loss / batch_idx:.4f}")

            avg_loss = epoch_loss / max(len(train_loader), 1)
            print(f"Epoch {epoch + 1}/{self.epochs} - Train loss: {avg_loss:.4f}")

            ep = epoch + 1
            self.training_history["epoch"].append(ep)
            self.training_history["train_loss"].append(float(avg_loss))
            self.training_history["epoch_boundaries"].append(global_batch_step)
            self.training_history["learning_rate"].append(
                float(scheduler.get_last_lr()[0])
            )

            if val_loader is not None:
                val_loss, val_f1_macro, val_f1_micro, val_f1_weighted = self._evaluate(
                    val_loader, criterion
                )
                print(
                    f"  Val loss: {val_loss:.4f} - Val F1 macro/micro/weighted: "
                    f"{val_f1_macro:.4f} / {val_f1_micro:.4f} / {val_f1_weighted:.4f}"
                )
                self.training_history["val_loss"].append(float(val_loss))
                self.training_history["val_f1_macro"].append(val_f1_macro)
                self.training_history["val_f1_micro"].append(val_f1_micro)
                self.training_history["val_f1_weighted"].append(val_f1_weighted)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

    def _evaluate(
        self, data_loader: DataLoader, criterion: torch.nn.Module
    ) -> Tuple[float, float, float, float]:
        """Compute validation loss and F1 (macro / micro / weighted) at threshold 0.5."""
        from sklearn.metrics import f1_score

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                loss = criterion(logits, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(np.int32)
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        self.model.train()

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        avg_loss = total_loss / len(data_loader)
        f1_macro = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        f1_micro = float(f1_score(all_labels, all_preds, average="micro", zero_division=0))
        f1_weighted = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))

        return avg_loss, f1_macro, f1_micro, f1_weighted

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities for each label.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 6) with probabilities in [0, 1].
        """
        if not texts:
            return np.zeros((0, 6), dtype=np.float32)

        self.model.eval()
        all_probs = []

        batch_size = self.batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs).astype(np.float32)

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels for given texts.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 6) with {0, 1} predictions.
        """
        proba = self.predict_proba(texts)
        return (proba >= threshold).astype(np.int32)

    def save(self, path: str) -> None:
        """Save model and tokenizer to disk (HuggingFace format).

        If tensor decomposition was applied, a ``td_config.json`` is written
        alongside so ``load()`` can rebuild the decomposed architecture.
        """
        save_dir = path[:-4] if path.endswith(".pkl") else path
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        save_td_config(save_dir, self._tensor_decomposition_config)
        self._last_save_path = save_dir

    def load(self, path: str) -> None:
        """Load model and tokenizer from disk.

        Detects a saved ``td_config.json``; if present, rebuilds the decomposed
        architecture before loading weights.
        """
        load_dir = path[:-4] if path.endswith(".pkl") else path

        td_cfg = load_td_config(load_dir)
        if td_cfg and td_cfg.get("enabled", False):
            from transformers import AutoConfig

            model_config = AutoConfig.from_pretrained(load_dir)
            self.model = AutoModelForSequenceClassification.from_config(model_config)
            apply_tensor_decomposition(self.model, td_cfg, verbose=False)
            state_dict = _load_hf_state_dict(load_dir)
            self.model.load_state_dict(state_dict, strict=False)
            self._tensor_decomposition_config = dict(td_cfg)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                load_dir,
                ignore_mismatched_sizes=True,
            )
            self._tensor_decomposition_config = {}

        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        self.model.to(self.device)
        self._last_save_path = load_dir

    def get_model_size(self) -> float:
        """Return total size of saved model directory in MB."""
        if self._last_save_path and os.path.isdir(self._last_save_path):
            total = 0
            for dirpath, _, filenames in os.walk(self._last_save_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total += os.path.getsize(fp)
            return round(total / (1024 * 1024), 2)

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            self.model.save_pretrained(tmpdir)
            self.tokenizer.save_pretrained(tmpdir)
            total = 0
            for dirpath, _, filenames in os.walk(tmpdir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total += os.path.getsize(fp)
            return round(total / (1024 * 1024), 2)
