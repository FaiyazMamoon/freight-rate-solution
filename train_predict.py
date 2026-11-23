"""Fit final estimators on labeled rows and write the two requested CSV files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from freight_rate.pipeline import (ChartRateModel, MainRateModel, align_predictions,
    load_models, save_models, source_hashes, runtime_versions)


def summarize(values) -> dict:
    values = np.asarray(values)
    return dict(rows=len(values), minimum=float(values.min()), maximum=float(values.max()),
        mean=float(values.mean()), finite=bool(np.isfinite(values).all()), positive=bool((values > 0).all()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--load-models", action="store_true", help="Predict from trusted existing artifacts")
    args = parser.parse_args()
    hashes = source_hashes(args.input_dir)
    validation = pd.read_csv(args.input_dir / "validation.csv")
    template = pd.read_csv(args.input_dir / "validation-predictions-template.csv")
    december = pd.read_csv(args.input_dir / "december-chart-inputs.csv")
    if args.load_models:
        main_model, chart_model = load_models(args.model_dir)
    else:
        train = pd.read_csv(args.input_dir / "train-test.csv")
        main_model, chart_model = MainRateModel().fit(train), ChartRateModel().fit(train)
        save_models(main_model, chart_model, args.model_dir, hashes)
    main_pred, chart_pred = main_model.predict(validation), chart_model.predict(december)
    output = align_predictions(validation, template, main_pred)
    december_output = december.copy()
    december_output["predicted_rate"] = chart_pred
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "validation_predictions.csv", index=False, float_format="%.10f")
    december_output.to_csv(args.output_dir / "december_predictions.csv", index=False, float_format="%.10f")
    if source_hashes(args.input_dir) != hashes:
        raise RuntimeError("Original source files changed during prediction")
    run = dict(loaded_models=args.load_models, source_sha256=hashes,
        training=main_model.training, runtime_versions=runtime_versions(),
        validation=summarize(main_pred), december=summarize(chart_pred),
        template_order_preserved=bool(output.load_id.equals(template.load_id)),
        original_sources_unchanged=True,
        outputs=["validation_predictions.csv", "december_predictions.csv"])
    (args.output_dir / "prediction_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
