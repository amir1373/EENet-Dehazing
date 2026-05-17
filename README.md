# EENet Dehazing

Python implementation workspace for EENet-style image dehazing experiments.

## What This Repository Contains

- `model/` - model implementation files.
- `scripts/` - training, evaluation, or utility scripts.
- `utils/` - helper functions.
- `data/` - dataset-related structure or examples.
- `assets/` - images and documentation assets.
- `results/` - output examples or experiment artifacts.
- `requirements.txt` - Python dependencies.
- `.github/workflows/python-compile.yml` - lightweight Python compile check.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Notes

For reproducible experiments, document the dataset split, checkpoint initialization, training schedule, hyperparameters, and evaluation metrics used for each result.