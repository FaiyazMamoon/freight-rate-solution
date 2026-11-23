"""Independently check the delivered files; never fit a model or change inputs.

Run from any directory with the solution environment:
    python validation/check_submission.py

The supplied score.py establishes file conformance, not predictive accuracy.
This checker adds source preservation, alignment, serialization, and PDF checks.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    require(len(set(header)) == len(header), f"Duplicate CSV headers in {path.name}")
    require(all(len(row) == len(header) for row in rows), f"Ragged CSV rows in {path.name}")
    return header, rows


def positive_rates(rows: list[list[str]], index: int) -> dict:
    values = [float(row[index]) for row in rows]
    require(all(math.isfinite(v) and v > 0 for v in values), "Rates must be finite and strictly positive")
    return {"rows": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def check_originals(root: Path) -> dict:
    manifest = json.loads((root / "analysis/source_manifest.json").read_text(encoding="utf-8"))["sha256"]
    require(len(manifest) == 8, "Expected the eight-file original assessment manifest")
    actual = {name: sha256(root / name) for name in manifest}
    changed = [name for name in manifest if actual[name] != manifest[name]]
    require(not changed, f"Original assessment files changed: {changed}")
    return {"files_checked": len(actual), "sha256": actual}


def check_prediction_files(root: Path, main_path: Path, december_path: Path) -> dict:
    header, rows = csv_rows(main_path)
    require(header == ["load_id", "predicted_rate"], "Incorrect main prediction schema/order")
    _, template = csv_rows(root / "validation-predictions-template.csv")
    input_header, hidden_rows = csv_rows(root / "validation.csv")
    ids = [row[0] for row in rows]
    template_ids = [row[0] for row in template]
    hidden_ids = [row[input_header.index("load_id")] for row in hidden_rows]
    require(len(rows) == 12_000, "Main predictions must contain exactly 12,000 rows")
    require(len(set(ids)) == 12_000 and all(ids), "Missing or duplicate main IDs")
    require(ids == template_ids == hidden_ids, "Prediction IDs/order differ from the template or inputs")
    require(set(ids) == {f"TE-{i:06d}" for i in range(1, 12_001)}, "Unexpected validation ID set")
    main_stats = positive_rates(rows, 1)

    december_header, december_rows = csv_rows(december_path)
    expected_header, expected_rows = csv_rows(root / "december-chart-inputs.csv")
    require(december_header == expected_header == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"], "Incorrect December schema/order")
    require(len(december_rows) == 31, "December predictions must contain exactly 31 rows")
    require([row[:6] for row in december_rows] == [row[:6] for row in expected_rows], "December input values or row order changed")
    expected_dates = [(date(2025, 12, 1) + timedelta(days=i)).isoformat() for i in range(31)]
    require([row[5] for row in december_rows] == expected_dates, "December dates/order differ from December 2025")
    require(all(row[:5] == ["Lexington", "Fort Wayne", "360", "Dry Van", "32000"] for row in december_rows), "Incorrect fixed December inputs")
    december_stats = positive_rates(december_rows, 6)
    return {"main": main_stats, "december": december_stats, "input_and_template_order_preserved": True, "december_inputs_preserved_exactly": True}


def check_provided_validators(root: Path, main_path: Path, december_path: Path) -> dict:
    import pandas as pd

    spec = importlib.util.spec_from_file_location("assessment_score", root / "score.py")
    require(spec is not None and spec.loader is not None, "Unable to load supplied scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_predictions(pd.read_csv(main_path))
    module.validate_december(pd.read_csv(december_path))
    return {"supplied_main_validator": "passed", "supplied_december_validator": "passed", "accuracy_measured_by_scorer": False}


def check_model_metadata(root: Path, model_dir: Path) -> dict:
    import pandas as pd

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    source = json.loads((root / "analysis/source_manifest.json").read_text(encoding="utf-8"))["sha256"]
    require(metadata["format_version"] == 1 and metadata["seed"] == 1729, "Unexpected model artifact format/seed")
    require(metadata["source_sha256"] == source, "Model input hashes differ from original assessment inputs")
    for artifact in ("main_hgb.joblib", "december_catboost.cbm"):
        path = model_dir / artifact
        require(path.is_file() and path.stat().st_size > 1000, f"Missing or empty model artifact: {artifact}")
    expected_training = {"source": "train-test.csv", "rows": 48_000, "date_min": "2025-01-01", "date_max": "2025-10-31", "target": "posted_rate", "validation_used_for_training": False}
    for branch in ("main", "chart"):
        state = metadata[branch]
        require(state["training"] == expected_training, f"Unexpected documented training provenance: {branch}")
        columns = state["feature_columns"]
        require(len(set(columns)) == len(columns), f"Duplicate model features: {branch}")
        require(not any("quote" in column or column in {"load_id", "posted_rate", "predicted_rate"} for column in columns), f"Excluded pricing/key/target field appears in {branch} features")
        require(set(state["numeric_dtypes"]).issubset(columns), f"Numeric dtype metadata has unexpected features: {branch}")
    main_columns = metadata["main"]["feature_columns"]
    require("market_index" in main_columns and "market_index_distance" in main_columns, "Main model must retain observed market inputs")
    reference = json.loads((root / "analysis/hgb_primary_results.json").read_text(encoding="utf-8"))
    require(main_columns == reference["feature_contracts"]["hgb_robust_market_only"]["columns"], "Main features differ from the evaluated feature contract")
    require(metadata["main"]["parameters"] == reference["parameters"], "Main parameters differ from the evaluated configuration")
    chart = metadata["chart"]
    require(not any("market" in column for column in chart["feature_columns"]), "December model must exclude all market features")
    require(set(chart["numeric_medians"]) == set(chart["numeric_dtypes"]), "December median state is incomplete")
    require(all(math.isfinite(float(value)) for value in chart["numeric_medians"].values()), "Nonfinite December imputation medians")
    expected_chart_parameters = {"loss_function": "MAE", "iterations": 500, "depth": 7, "learning_rate": .07, "l2_leaf_reg": 8, "random_seed": 1729, "thread_count": 4, "verbose": False, "allow_writing_files": False}
    require(chart["parameters"] == expected_chart_parameters, "December estimator differs from evaluated settings")
    train = pd.read_csv(root / "train-test.csv")
    expected_lookup = {}
    for side in ("pickup", "delivery"):
        for city, lat, lon in train[[side, f"{side}_lat", f"{side}_lon"]].itertuples(index=False, name=None):
            coordinates = [float(lat), float(lon)]
            require(city not in expected_lookup or expected_lookup[city] == coordinates, f"Inconsistent training coordinates: {city}")
            expected_lookup[city] = coordinates
    require(chart["city_coordinate_lookup"] == expected_lookup, "Saved city lookup does not match the labeled training data")
    for branch in ("main", "chart"):
        vocabularies = metadata[branch]["category_vocabularies"]
        for column in ("pickup", "delivery", "equipment"):
            require(set(vocabularies[column]) == set(train[column]), f"{branch} category vocabulary differs from training {column}")
    for package, version in metadata["runtime_versions"]["packages"].items():
        import importlib.metadata

        require(importlib.metadata.version(package) == version, f"Runtime version differs from model artifact for {package}")
    return {"main_features": len(main_columns), "december_features": len(chart["feature_columns"]), "training_rows_documented": 48_000, "city_lookup_entries": len(expected_lookup), "features_exclude_ids_targets_and_quote": True, "december_excludes_market": True, "source_hashes_match": True, "runtime_versions_match_artifacts": True}


def check_serialized_predictions(root: Path, model_dir: Path, main_path: Path, december_path: Path) -> dict:
    import numpy as np
    import pandas as pd
    from freight_rate.pipeline import load_models

    main, chart = load_models(model_dir)
    hidden = pd.read_csv(root / "validation.csv")
    original_december = pd.read_csv(root / "december-chart-inputs.csv")
    main_values = main.predict(hidden)
    december_values = chart.predict(original_december)
    main_csv = pd.read_csv(main_path).set_index("load_id")
    saved_main = main_csv.loc[hidden.load_id, "predicted_rate"].to_numpy()
    saved_december = pd.read_csv(december_path).predicted_rate.to_numpy()
    require(np.allclose(main_values, saved_main, atol=1e-8, rtol=0), "Saved main model does not reproduce exported predictions")
    require(np.allclose(december_values, saved_december, atol=1e-8, rtol=0), "Saved December model does not reproduce exported predictions")
    altered_quote = hidden.copy()
    altered_quote["quote_signal"] = np.arange(len(hidden), dtype=float) + 1_000_000
    require(np.array_equal(main_values, main.predict(altered_quote)), "Main predictions unexpectedly depend on excluded quote_signal")
    train = pd.read_csv(root / "train-test.csv", usecols=["pickup", "delivery"])
    cities = set(train.pickup) | set(train.delivery)
    unseen = (~hidden.pickup.isin(cities)) | (~hidden.delivery.isin(cities))
    require(int(unseen.sum()) > 0, "Expected supplied unseen-city rows were not exercised")
    return {"main_rows_reproduced": len(main_values), "december_rows_reproduced": len(december_values), "absolute_tolerance": 1e-8, "main_max_export_difference": float(np.max(np.abs(main_values - saved_main))), "december_max_export_difference": float(np.max(np.abs(december_values - saved_december))), "excluded_quote_behaviorally_invariant": True, "unseen_city_rows_exercised": int(unseen.sum()), "missing_weight_rows_exercised": int(hidden.weight.isna().sum()), "missing_market_rows_exercised": int(hidden.market_index.isna().sum())}


def check_production_evaluation(root: Path) -> dict:
    result = json.loads((root / "evaluation/production_results.json").read_text(encoding="utf-8"))
    expected = {("chronological_jul_aug", "hgb_market_only"), ("chronological_sep_oct", "hgb_market_only"), ("random_80_20", "hgb_market_only"), ("unseen_cities_temporal", "hgb_market_only"), ("chronological_jul_aug", "catboost_universal"), ("chronological_sep_oct", "catboost_universal")}
    records = result["metrics"]
    require(len(records) == len(expected) and {(row["split"], row["model"]) for row in records} == expected, "Production evaluation is incomplete")
    expected_counts = {"chronological_jul_aug": 9671, "chronological_sep_oct": 9523, "random_80_20": 9600, "unseen_cities_temporal": 1271}
    for row in records:
        require(row["n"] == expected_counts[row["split"]], f"Unexpected evaluation row count: {row['split']}")
        require(all(math.isfinite(float(row[metric])) for metric in ("mae", "rmse", "r2", "mape_pct", "smape_pct")), "Nonfinite evaluation metric")
        require(row["nonpositive_predictions"] == 0, "Evaluation produced nonpositive predictions")
    for split, metadata in result["split_metadata"].items():
        if split != "random_80_20":
            require(metadata["train_dates"][1] < metadata["test_dates"][0], f"Training dates overlap a future holdout: {split}")
    comparisons = result["reference_comparisons"]
    require(len(comparisons) == len(expected) and {(row["split"], row["model"]) for row in comparisons} == expected, "Audited reference comparisons are incomplete")
    for row in comparisons:
        require(row["passed"] and all(abs(delta) <= row["absolute_tolerance"] for delta in row["deltas"].values()), "A production backtest failed audited reference tolerance")
    require(result["checks"]["audit_metrics_reproduced"] and result["checks"]["original_sources_unchanged"], "Production evaluation checks failed")
    require(not result["checks"]["hidden_labels_used"] and not result["checks"]["hidden_predictors_used"], "Evaluation metadata documents hidden-data modeling use")
    return {"metric_records": len(records), "reference_comparisons": len(comparisons), "all_reference_tolerances_met": True, "chronological_training_precedes_holdout": True}


def check_report(root: Path, pdf_path: Path, chart_path: Path) -> dict:
    from PIL import Image
    from pypdf import PdfReader

    require(chart_path.is_file(), "Scorer chart is missing")
    with Image.open(chart_path) as image:
        image.load()
        require(image.width >= 1000 and image.height >= 500, "Scorer chart is unexpectedly small")
        chart = image.convert("RGB")
        chart_size = chart.size
        chart_pixels = chart.tobytes()
    reader = PdfReader(pdf_path)
    require(not reader.is_encrypted, "Report PDF must be accessible without a password")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    require(len(reader.pages) >= 2 and len(text.strip()) >= 2000, "Report PDF lacks readable content")
    normalized = text.lower()
    for term in ("validation", "september", "october", "december", "mae"):
        require(term in normalized, f"Report does not describe {term}")
    embedded_images = 0
    matched_chart = False
    for page in reader.pages:
        for image_object in page.images:
            embedded_images += 1
            embedded = image_object.image.convert("RGB")
            if embedded.size == chart_size and embedded.tobytes() == chart_pixels:
                matched_chart = True
    require(matched_chart, "Report PDF does not embed the actual scorer chart pixels")
    require((root / "reports/build_report.py").is_file(), "Report source is missing")
    return {"pages": len(reader.pages), "readable_characters": len(text), "embedded_images": embedded_images, "actual_scorer_chart_embedded": matched_chart, "chart_dimensions": list(chart_size)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--predictions", type=Path, default=Path("validation_predictions.csv"))
    parser.add_argument("--december-predictions", type=Path, default=Path("december_predictions.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/Freight_Rate_Assessment_Report.pdf"))
    parser.add_argument("--chart", type=Path, default=Path("scorer_results/candidate_december.png"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("validation/submission_checks.json"))
    args = parser.parse_args()
    root = args.root.resolve()

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    predictions, december, report, chart = map(resolved, [args.predictions, args.december_predictions, args.report, args.chart])
    results: dict = {"checks": {}, "limitations": ["File conformance and serialized prediction reproducibility do not establish hidden accuracy.", "Metadata records training provenance; this checker cannot prove the absence of all possible leakage."]}

    def run(label: str, function: Callable[[], dict]) -> None:
        try:
            details = function()
        except (Exception, SystemExit) as exc:
            results["checks"][label] = {"passed": False, "error": str(exc)}
            print(f"FAIL {label}: {exc}")
        else:
            results["checks"][label] = {"passed": True, **details}
            print(f"PASS {label}")

    run("original_assessment_files", lambda: check_originals(root))
    run("prediction_files_and_alignment", lambda: check_prediction_files(root, predictions, december))
    run("provided_scorer_validators", lambda: check_provided_validators(root, predictions, december))
    run("model_metadata_and_training_provenance", lambda: check_model_metadata(root, resolved(args.models)))
    run("serialized_model_prediction_reproduction", lambda: check_serialized_predictions(root, resolved(args.models), predictions, december))
    run("production_evaluation", lambda: check_production_evaluation(root))
    run("pdf_and_scorer_chart", lambda: check_report(root, report, chart))
    results["passed"] = all(check["passed"] for check in results["checks"].values())
    artifact_candidates = [predictions, december, report, chart, root / "reports/build_report.py", Path(__file__).resolve()]
    artifact_candidates += [root / name for name in ("train_predict.py", "evaluate.py", "run_project.py", "prediction_run.json", "requirements-solution.txt", "README.md")]
    artifact_candidates += sorted((root / "models").glob("*"))
    artifact_candidates += sorted((root / "freight_rate").rglob("*.py"))
    artifact_candidates += sorted((root / "evaluation").glob("*.json"))
    results["artifact_sha256"] = {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in artifact_candidates if path.is_file() and path.is_relative_to(root)}
    output = resolved(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(root) if output.is_relative_to(root) else output}")
    if not results["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
