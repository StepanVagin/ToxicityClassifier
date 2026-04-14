# ToxicityClassifier

Multi-label text classification for the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) challenge: six toxicity labels (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).

The project downloads data via the Kaggle API, trains a model (logistic regression with TF–IDF, or transformer backbones such as ALBERT / DistilBERT), tunes per-label decision thresholds, and writes metrics and figures under `outputs/`.

## Requirements

- Python 3.10+ (tested in development with recent Python versions)
- A Kaggle account with access to the competition data (see below)

Install dependencies from the project root:

```bash
cd ToxicityClassifier
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

GPU training for transformer models requires a CUDA-capable PyTorch install matching your system; see [PyTorch install instructions](https://pytorch.org/get-started/locally/).

---

## Kaggle API authentication

Downloads use the official [`kaggle`](https://github.com/Kaggle/kaggle-api) Python package (`KaggleApi().authenticate()`). You must authenticate **before** running anything that fetches the dataset.

### 1. Join the competition

1. Open [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
2. Sign in and click **Join Competition** (or **Late Submission** if the competition is closed), and accept the rules. Without this, API downloads fail with a permissions error.

### 2. Create API credentials on Kaggle

1. Go to **Kaggle → Account** (click your avatar → **Settings**).
2. Scroll to **API** and click **Create New Token**. This downloads `kaggle.json` (contains `username` and `key`).

### 3. Place credentials where the CLI expects them

**Recommended (file-based):**

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

On Windows, put `kaggle.json` in `%USERPROFILE%\.kaggle\`.

**Alternative (environment variables):** the Kaggle client also reads:

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_api_key"
```

Use the same `username` and `key` values as in `kaggle.json`. This project’s downloader can also pass credentials programmatically, but the default path is the file or these env vars.

### 4. Verify

```bash
kaggle competitions list -s jigsaw
```

If authentication works, you should see the competition (or no auth error). Then you can run the pipeline or download step below.

**Troubleshooting:** If download fails, confirm you joined the competition, accepted rules, and that `~/.kaggle/kaggle.json` exists with correct permissions. Errors mentioning “Late Submission” usually mean you need to accept the competition rules on the website.

---

## How to run the project

All commands below assume your virtual environment is active and your working directory is the project root (`ToxicityClassifier/`).

### End-to-end training and evaluation

```bash
python run_pipeline.py
```

Defaults merge `configs/base_config.json` with `configs/logistic_regression.json`, download data if missing, train, tune thresholds, and evaluate on the held-out split.

Useful options:

| Flag | Meaning |
|------|---------|
| `--config configs/albert.json` | Train/eval with ALBERT (example GPU-oriented config). |
| `--config configs/distilbert.json` | DistilBERT variant. |
| `--skip-download` | Use existing `data/processed/train.csv`, `test.csv`, `test_labels.csv` (no Kaggle call). |
| `--no-tune` | Skip threshold search; use 0.5 for all labels. |
| `--eval-only` | Load a saved model from `models/saved/`, skip training (requires a prior train). |
| `--retune-thresholds` | With `--eval-only`, re-run CV threshold tuning instead of loading saved JSON. |

Outputs go to `outputs/` (metrics, confusion matrices, ROC/PR/calibration plots, etc.). Models and thresholds are saved under `models/saved/`.

### Download data only

```bash
python data/download_dataset.py
# or
python -m data.download_dataset
```

Writes zips under `data/raw/` and CSVs under `data/processed/`.

### Inference on new text

After training (or with a compatible saved model):

```bash
python inference.py --text "Example comment"
python inference.py --input comments.txt --format json
```

See `python inference.py --help` for flags.

### Sanity checks (pipeline behavior)

```bash
python sanity_check.py
python sanity_check.py --skip-download   # if data is already present
```

### Tests

```bash
pytest tests/
```

### Optional scripts

- `python scripts/check_cuda.py` — quick CUDA visibility check.
- `python scripts/benchmark.py` — resource/latency benchmarking for saved models.
- `python scripts/combine_figures.py` — combine existing figures (see script for paths).

---

## Project layout (short)

| Path | Role |
|------|------|
| `configs/` | `base_config.json` plus model-specific JSON (merged at runtime). |
| `data/` | Download, preprocessing, loading. |
| `models/` | Model implementations and saved weights under `models/saved/`. |
| `training/` | Training loop. |
| `evaluation/` | Metrics, threshold tuning, plots. |
| `outputs/` | Generated metrics and figures from `run_pipeline.py`. |
| `reports/` | LaTeX reports (not produced automatically by the pipeline). |

---

## Data notice

Competition data is subject to [Kaggle’s rules](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/rules). Do not redistribute downloaded files; keep `kaggle.json` private.
