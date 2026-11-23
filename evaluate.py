"""Recheck final production policies on historical data without hidden labels."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from freight_rate.pipeline import (ChartRateModel, MainRateModel, SEED,
    runtime_versions, source_hashes)


def metrics(y, predictions) -> dict:
    y, p = np.asarray(y), np.asarray(predictions)
    return dict(n=len(y), mae=float(mean_absolute_error(y, p)),
        rmse=float(np.sqrt(mean_squared_error(y, p))),
        r2=float(r2_score(y, p)) if len(y) > 1 else None,
        mape_pct=float(100 * np.mean(abs(y - p) / y)),
        smape_pct=float(200 * np.mean(abs(y - p) / (abs(y) + abs(p)))),
        mean_error_pred_minus_true=float(np.mean(p - y)),
        min_prediction=float(p.min()), max_prediction=float(p.max()),
        nonpositive_predictions=int((p <= 0).sum()))


def error_slices(train, test, pred):
    routes = set(zip(train.pickup, train.delivery))
    cities = set(train.pickup) | set(train.delivery)
    dimensions = dict(month=test.date.str[:7], equipment=test.equipment,
        distance=pd.cut(test.distance, [0, 300, 600, 1000, 1500, 2000, np.inf], right=False).astype(str),
        route=pd.Series(["seen" if route in routes else "unseen" for route in zip(test.pickup, test.delivery)], index=test.index),
        city=pd.Series(["both_seen" if p in cities and d in cities else "at_least_one_unseen" for p, d in zip(test.pickup, test.delivery)], index=test.index),
        target=pd.cut(test.posted_rate, [0, 500, 1000, 2000, 4000, 8000, np.inf], right=False).astype(str),
        weight=pd.Series(np.where(test.weight.isna(), "missing_original", np.where(test.weight.lt(0), "negative_original", "positive_original")), index=test.index))
    rows = []
    for dimension, groups in dimensions.items():
        for group in groups.unique():
            mask = (groups == group).to_numpy()
            rows.append(dict(slice=dimension, group=str(group), **metrics(test.posted_rate.to_numpy()[mask], pred[mask])))
    return rows


def load_reference(directory: Path) -> dict:
    """Read recorded numeric results only; production never imports audit code."""
    reference = {}
    if (directory / "robustness_results.json").exists():
        source = json.loads((directory / "robustness_results.json").read_text(encoding="utf-8"))
        for row in source["metrics"]:
            if row["experiment"] == "abs_weight_flag_market_only":
                reference[("chronological_" + row["fold"], "hgb_market_only")] = row
    if (directory / "hgb_primary_results.json").exists():
        source = json.loads((directory / "hgb_primary_results.json").read_text(encoding="utf-8"))
        for row in source["metrics"]:
            if row["model"] == "hgb_robust_market_only":
                reference[(row["split"], "hgb_market_only")] = row
    if (directory / "hgb_stress_results.json").exists():
        source = json.loads((directory / "hgb_stress_results.json").read_text(encoding="utf-8"))
        for row in source["metrics"]:
            if row["configuration"] == "abs_weight_flag_market_only":
                reference[(row["split"], "hgb_market_only")] = row
    if (directory / "benchmark_metrics.csv").exists():
        for row in pd.read_csv(directory / "benchmark_metrics.csv").to_dict("records"):
            if row["model"] == "catboost_universal_mae":
                reference[(row["split"], "catboost_universal")] = row
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--reference-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--metric-atol", type=float, default=2., help="Absolute $ tolerance for audited MAE/RMSE reproduction")
    args = parser.parse_args()
    tick = time.monotonic()
    hashes = source_hashes(args.input_dir)
    # Hidden and chart rows are never parsed or used; source bytes are hashed only.
    df = pd.read_csv(args.input_dir / "train-test.csv").sort_values(["date", "load_id"]).reset_index(drop=True)
    indices = np.arange(len(df))
    random_train, random_test = train_test_split(indices, test_size=.2, random_state=SEED)
    held_cities = np.random.default_rng(SEED).choice(sorted(set(df.pickup) | set(df.delivery)), size=4, replace=False)
    held = df.pickup.isin(held_cities) | df.delivery.isin(held_cities)
    splits = [
        ("chronological_jul_aug", indices[df.date < "2025-07-01"], indices[(df.date >= "2025-07-01") & (df.date < "2025-09-01")], True),
        ("chronological_sep_oct", indices[df.date < "2025-09-01"], indices[df.date >= "2025-09-01"], True),
        ("random_80_20", random_train, random_test, False),
        ("unseen_cities_temporal", indices[(df.date < "2025-09-01") & ~held], indices[(df.date >= "2025-09-01") & held], False),
    ]
    result = dict(seed=SEED, metrics=[], split_metadata={}, reference_comparisons=[],
        runtime_versions=runtime_versions(), source_sha256=hashes,
        checks=dict(hidden_labels_used=False, hidden_predictors_used=False, hidden_files_hashed_only=True, fitted_artifacts_saved=False,
            target_rows_deleted=False, target_values_modified=False, holdout_early_stopping=False),
        notes=["The temporal test months are disjoint, but expanding training windows overlap; these are model-selection checks rather than independent final estimates.",
               "Random splitting is an optimism diagnostic. City holdout is a proxy for unseen future cities.",
               "MAE/RMSE reproduction tolerance is absolute dollars; runtime versions are recorded.",
               "All observed targets and extreme values are retained. No clipping, price signals forecast or target encodings are applied."])
    slices = []
    reference = load_reference(args.reference_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, trix, teix, include_chart in splits:
        train, test = df.iloc[trix].copy(), df.iloc[teix].copy()
        result["split_metadata"][split] = dict(train_n=len(train), test_n=len(test),
            train_dates=[str(train.date.min()), str(train.date.max())], test_dates=[str(test.date.min()), str(test.date.max())],
            held_cities=held_cities.tolist() if split == "unseen_cities_temporal" else [])
        models = [("hgb_market_only", MainRateModel)]
        if include_chart:
            models.append(("catboost_universal", ChartRateModel))
        for name, estimator_class in models:
            started = time.monotonic()
            model = estimator_class().fit(train)
            pred = model.predict(test.drop(columns="posted_rate"))
            row = dict(split=split, model=name, train_n=len(train), feature_count=len(model.feature_columns),
                fit_predict_seconds=time.monotonic() - started, **metrics(test.posted_rate, pred))
            result["metrics"].append(row)
            slices.extend(dict(split=split, model=name, **slice_row) for slice_row in error_slices(train, test, pred))
            expected = reference.get((split, name))
            if expected:
                deltas = {metric: row[metric] - expected[metric] for metric in ["mae", "rmse"]}
                passed = all(abs(delta) <= args.metric_atol for delta in deltas.values())
                result["reference_comparisons"].append(dict(split=split, model=name,
                    expected={metric: expected[metric] for metric in ["mae", "rmse"]},
                    deltas=deltas, absolute_tolerance=args.metric_atol, passed=passed))
            print(json.dumps(row), flush=True)
            (args.output_dir / "production_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["runtime_seconds"] = time.monotonic() - tick
    result["checks"]["original_sources_unchanged"] = source_hashes(args.input_dir) == hashes
    result["checks"]["audit_metrics_reproduced"] = bool(result["reference_comparisons"]) and all(item["passed"] for item in result["reference_comparisons"])
    (args.output_dir / "production_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(result["metrics"]).to_csv(args.output_dir / "production_metrics.csv", index=False)
    pd.DataFrame(slices).to_csv(args.output_dir / "production_error_slices.csv", index=False)
    if not result["checks"]["original_sources_unchanged"]:
        raise RuntimeError("Original files changed during evaluation")
    if result["reference_comparisons"] and not result["checks"]["audit_metrics_reproduced"]:
        raise SystemExit("Production metrics differ from audit tolerance; inspect evaluation/production_results.json")


if __name__ == "__main__":
    main()
