# Freight Rate Prediction

This project predicts total freight prices for future loads. The main model uses histogram gradient boosting and the observed market index. A separate CatBoost model produces the December chart from the smaller set of inputs available for that scenario. Both models use absolute-error loss and exclude `quote_signal` because its relationship with price changes across months.

## Setup

Use Python 3.11. From the repository root:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\python -m pip install -r requirements-solution.txt
.venv\Scripts\python -m unittest discover -s tests -v
```

On macOS or Linux, use `.venv/bin/python` for the same commands. The tests generate their own sample data.

## Reproduce the assessment outputs

Place your copy of the supplied assessment files in the repository root. The runner checks their hashes before using them.

- `train-test.csv`
- `validation.csv`
- `validation-predictions-template.csv`
- `december-chart-inputs.csv`
- `score.py`
- `requirements.txt`
- `readme.txt`
- `freight-rate-ml-assessment-1-.pdf`

Run `python run_project.py` with the virtual environment's Python. It runs the tests and backtests, fits both models, writes `validation_predictions.csv` and `december_predictions.csv`, and uses the provided scorer to create the chart for the PDF report. The predictions and assessment report are provided separately with the submission.

## Code map

- [`freight_rate/pipeline.py`](freight_rate/pipeline.py) defines the feature transforms, training, and inference for both models.
- [`evaluate.py`](evaluate.py) runs chronological backtests and a held-out city test. The small reference files under `analysis/` contain comparison metrics and input checksums.
- [`train_predict.py`](train_predict.py) fits both models and writes the predictions in template order.
- [`reports/build_report.py`](reports/build_report.py) builds the report from the measured results and the scorer's chart.
- [`validation/check_submission.py`](validation/check_submission.py) checks the output schemas and confirms that reloaded models reproduce the predictions.
