"""Deployable feature policies selected by the historical assessment audits.

Only labeled training rows determine vocabularies, imputations and city metadata.
No target, load identifier or quote signal enters either feature matrix.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

SEED = 1729
CATEGORICAL = ["pickup", "delivery", "equipment"]
COORDINATES = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
BASE_COLUMNS = ["pickup", "delivery", *COORDINATES, "distance", "equipment", "weight"]
TRAIN_COLUMNS = ["load_id", *BASE_COLUMNS, "date", "market_index", "quote_signal", "posted_rate"]
MAIN_COLUMNS = TRAIN_COLUMNS[:-1]
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
MAIN_PARAMETERS = dict(loss="absolute_error", max_iter=150, learning_rate=.08,
    max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=10,
    early_stopping=False, categorical_features="from_dtype", random_state=SEED)
CHART_PARAMETERS = dict(iterations=500, depth=7, learning_rate=.07,
    loss_function="MAE", l2_leaf_reg=8, random_seed=SEED,
    thread_count=4, verbose=False, allow_writing_files=False)
SOURCE_NAMES = ["train-test.csv", "validation.csv", "validation-predictions-template.csv",
    "december-chart-inputs.csv", "score.py", "readme.txt", "requirements.txt",
    "freight-rate-ml-assessment-1-.pdf"]


def source_hashes(directory: Path) -> dict:
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in SOURCE_NAMES if (directory / name).is_file()}


def runtime_versions() -> dict:
    names = ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib", "threadpoolctl"]
    return {"python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in names}}


def _require_columns(frame: pd.DataFrame, expected: list, label: str) -> None:
    if frame.columns.duplicated().any():
        raise ValueError(f"{label}: duplicate column names")
    missing, extra = set(expected) - set(frame), set(frame) - set(expected)
    if missing or extra:
        raise ValueError(f"{label}: missing columns={sorted(missing)}, unexpected columns={sorted(extra)}")
    if frame.empty:
        raise ValueError(f"{label}: no rows")


def _validate_values(frame: pd.DataFrame, label: str) -> None:
    for column in CATEGORICAL:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{label}: missing {column}")
    date = pd.to_datetime(frame["date"], errors="coerce", format="%Y-%m-%d")
    if date.isna().any():
        raise ValueError(f"{label}: invalid date")
    for column in [*COORDINATES, "distance", "weight", "market_index", "quote_signal"]:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(values).any():
            raise ValueError(f"{label}: infinite {column}")
    distance = pd.to_numeric(frame["distance"], errors="raise")
    if distance.isna().any() or distance.le(0).any():
        raise ValueError(f"{label}: distance must be finite and positive")
    if "load_id" in frame:
        if frame.load_id.isna().any() or frame.load_id.duplicated().any():
            raise ValueError(f"{label}: missing or duplicate load_id")


def validate_training(frame: pd.DataFrame) -> None:
    _require_columns(frame, TRAIN_COLUMNS, "training")
    _validate_values(frame, "training")
    target = pd.to_numeric(frame.posted_rate, errors="raise")
    if not np.isfinite(target).all() or target.le(0).any():
        raise ValueError("training: posted_rate must be finite and positive")


def validate_main(frame: pd.DataFrame) -> None:
    _require_columns(frame, MAIN_COLUMNS, "main inference")
    _validate_values(frame, "main inference")


def city_lookup(train: pd.DataFrame) -> dict:
    lookup = {}
    for side in ("pickup", "delivery"):
        for city, lat, lon in train[[side, side + "_lat", side + "_lon"]].itertuples(index=False, name=None):
            values = [float(lat), float(lon)]
            if not np.isfinite(values).all():
                raise ValueError(f"City coordinates must be finite: {city}")
            if city in lookup and lookup[city] != values:
                raise ValueError(f"Inconsistent city coordinates: {city}")
            lookup[city] = values
    return lookup


def main_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact audited abs-weight/flags/observed-market policy (25 features)."""
    x = frame[[*BASE_COLUMNS, "market_index"]].copy()
    dates = pd.to_datetime(frame.date)
    x["elapsed_days"] = (dates - pd.Timestamp("2025-01-01")).dt.days
    x["month"], x["day"], x["weekday"] = dates.dt.month, dates.dt.day, dates.dt.weekday
    x["month_sin"] = np.sin(2 * np.pi * (dates.dt.month - 1) / 12)
    x["month_cos"] = np.cos(2 * np.pi * (dates.dt.month - 1) / 12)
    x["weight_negative"] = x.weight.lt(0).astype(int)
    x["weight"] = x.weight.abs()
    x["weight_missing_original"] = frame.weight.isna().astype(int)
    x["weight_distance"] = x.weight * x.distance / 10000
    x["weight_per_mile"] = x.weight / x.distance
    x["log_distance"] = np.log(x.distance)
    x["market_index_distance"] = x.market_index * x.distance
    for equipment in ["Dry Van", "Reefer", "Flatbed"]:
        x["distance_" + equipment.replace(" ", "_")] = x.distance * x.equipment.eq(equipment)
    return x


def chart_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact benchmark universal policy; raw weight and no pricing signals."""
    x = frame[BASE_COLUMNS].copy()
    dates = pd.to_datetime(frame.date)
    x["elapsed_days"] = (dates - pd.Timestamp("2025-01-01")).dt.days
    x["month"], x["day"], x["weekday"] = dates.dt.month, dates.dt.day, dates.dt.weekday
    x["week"] = dates.dt.isocalendar().week.astype(int)
    x["month_sin"] = np.sin(2 * np.pi * (dates.dt.month - 1) / 12)
    x["month_cos"] = np.cos(2 * np.pi * (dates.dt.month - 1) / 12)
    x["weekday_sin"] = np.sin(2 * np.pi * dates.dt.weekday / 7)
    x["weekday_cos"] = np.cos(2 * np.pi * dates.dt.weekday / 7)
    x["route"] = x.pickup.astype(str) + " -> " + x.delivery.astype(str)
    x["log_distance"] = np.log(x.distance)
    x["weight_per_mile"] = x.weight / x.distance
    p1, p2 = np.radians(x.pickup_lat), np.radians(x.delivery_lat)
    dp, dl = p2 - p1, np.radians(x.delivery_lon - x.pickup_lon)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    x["haversine_miles"] = 3958.7613 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    x["distance_ratio"] = x.distance / x.haversine_miles.clip(lower=1)
    x["latitude_delta"] = x.delivery_lat - x.pickup_lat
    x["longitude_delta"] = x.delivery_lon - x.pickup_lon
    bearing = np.arctan2(np.sin(dl) * np.cos(p2),
        np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))
    x["bearing_sin"], x["bearing_cos"] = np.sin(bearing), np.cos(bearing)
    for equipment in ["Dry Van", "Reefer", "Flatbed"]:
        x["distance_" + equipment.replace(" ", "_")] = x.distance * (x.equipment == equipment)
    x["weight_times_distance"] = x.weight * x.distance / 1e4
    return x


def _positive_predictions(values) -> np.ndarray:
    predictions = np.asarray(values, dtype=float)
    if not np.isfinite(predictions).all() or (predictions <= 0).any():
        raise ValueError("Model produced nonfinite or nonpositive prices; outputs were not written")
    return predictions


def _numeric_cast(x: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    for column, dtype in dtypes.items():
        x[column] = x[column].astype(dtype)
    return x


class MainRateModel:
    """Observed-market HGB with native numeric and unknown-category missingness."""

    def fit(self, train: pd.DataFrame) -> "MainRateModel":
        validate_training(train)
        x = main_features(train)
        self.categories = {column: list(x[column].unique()) for column in CATEGORICAL}
        self.numeric_dtypes = {column: str(x[column].dtype) for column in x if column not in CATEGORICAL}
        self.feature_columns = list(x)
        x = self._prepare(x)
        self.estimator = HistGradientBoostingRegressor(**MAIN_PARAMETERS)
        with threadpool_limits(limits=1):
            self.estimator.fit(x, train.posted_rate)
        self.training = _training_summary(train)
        return self

    def _prepare(self, x: pd.DataFrame) -> pd.DataFrame:
        x = _numeric_cast(x[self.feature_columns].copy(), self.numeric_dtypes)
        for column, categories in self.categories.items():
            x[column] = pd.Categorical(x[column], categories=categories)
        return x

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        validate_main(frame)
        return self._prepare(main_features(frame))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        with threadpool_limits(limits=1):
            return _positive_predictions(self.estimator.predict(self.transform(frame)))

    def metadata(self) -> dict:
        return dict(model="HistGradientBoostingRegressor", parameters=MAIN_PARAMETERS,
            inference_columns=MAIN_COLUMNS,
            feature_columns=self.feature_columns, numeric_dtypes=self.numeric_dtypes,
            category_vocabularies=self.categories, training=self.training,
            missing_policy="native numeric NaNs; unknown categories mapped to native missing",
            weight_policy="absolute weight plus original-negative and original-missing flags",
            pricing_policy="observed market_index retained; quote_signal excluded", threads=1)


class ChartRateModel:
    """Universal CatBoost for reduced December inputs and full-schema backtests."""

    def fit(self, train: pd.DataFrame) -> "ChartRateModel":
        validate_training(train)
        self.lookup = city_lookup(train)
        x = chart_features(train)
        self.feature_columns = list(x)
        self.categorical_columns = [*CATEGORICAL, "route"]
        self.numeric_dtypes = {column: str(x[column].dtype) for column in x if column not in self.categorical_columns}
        self.medians = x[list(self.numeric_dtypes)].median().to_dict()
        if not np.isfinite(list(self.medians.values())).all():
            raise ValueError("Training medians must be finite")
        self.categories = {column: list(x[column].unique()) for column in self.categorical_columns}
        self.estimator = CatBoostRegressor(**CHART_PARAMETERS)
        self.estimator.fit(self._prepare(x), train.posted_rate, cat_features=self.categorical_columns)
        self.training = _training_summary(train)
        return self

    def _prepare(self, x: pd.DataFrame) -> pd.DataFrame:
        x = _numeric_cast(x[self.feature_columns].copy(), self.numeric_dtypes)
        for column in self.categorical_columns:
            x[column] = x[column].astype(str)
        x[list(self.numeric_dtypes)] = x[list(self.numeric_dtypes)].fillna(self.medians)
        return x

    def adapt(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Restore only static coordinates; never synthesize absent price signals."""
        if set(frame) == set(MAIN_COLUMNS):
            validate_main(frame)
            return frame
        _require_columns(frame, DECEMBER_COLUMNS, "reduced chart inference")
        _validate_values(frame, "reduced chart inference")
        x = frame.drop(columns="predicted_rate").copy()
        for side in ("pickup", "delivery"):
            unknown = set(x[side]) - set(self.lookup)
            if unknown:
                raise ValueError(f"Unknown chart cities have no training coordinates: {sorted(unknown)}")
            x[side + "_lat"] = x[side].map(lambda city: self.lookup[city][0])
            x[side + "_lon"] = x[side].map(lambda city: self.lookup[city][1])
        return x

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._prepare(chart_features(self.adapt(frame)))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return _positive_predictions(self.estimator.predict(self.transform(frame)))

    def metadata(self) -> dict:
        return dict(model="CatBoostRegressor", parameters=CHART_PARAMETERS,
            inference_columns=DECEMBER_COLUMNS, backtest_columns=MAIN_COLUMNS,
            feature_columns=self.feature_columns, numeric_dtypes=self.numeric_dtypes,
            category_vocabularies=self.categories, categorical_columns=self.categorical_columns,
            numeric_medians=self.medians, city_coordinate_lookup=self.lookup, training=self.training,
            missing_policy="numeric medians fitted on training rows only",
            weight_policy="raw signed weight, matching evaluated universal CatBoost policy",
            pricing_policy="market_index and quote_signal and all their products excluded")


def _training_summary(train: pd.DataFrame) -> dict:
    return dict(source="train-test.csv", rows=len(train), date_min=str(train.date.min()),
        date_max=str(train.date.max()), target="posted_rate", validation_used_for_training=False)


def save_models(main: MainRateModel, chart: ChartRateModel, directory: Path, hashes: dict | None = None) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(main, directory / "main_hgb.joblib")
    chart.estimator.save_model(str(directory / "december_catboost.cbm"))
    metadata = dict(format_version=1, seed=SEED, runtime_versions=runtime_versions(),
        source_sha256=hashes or {}, main=main.metadata(), chart=chart.metadata())
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_models(directory: Path) -> tuple[MainRateModel, ChartRateModel]:
    """Load only trusted locally generated model artifacts (joblib uses pickle)."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["format_version"] != 1:
        raise ValueError("Unsupported artifact format")
    main = joblib.load(directory / "main_hgb.joblib")
    chart = ChartRateModel()
    state = metadata["chart"]
    chart.feature_columns, chart.numeric_dtypes = state["feature_columns"], state["numeric_dtypes"]
    chart.categorical_columns, chart.categories = state["categorical_columns"], state["category_vocabularies"]
    chart.medians, chart.lookup, chart.training = state["numeric_medians"], state["city_coordinate_lookup"], state["training"]
    chart.estimator = CatBoostRegressor()
    chart.estimator.load_model(str(directory / "december_catboost.cbm"))
    if main.metadata() != metadata["main"]:
        raise ValueError("Main model and feature metadata do not match")
    return main, chart


def align_predictions(validation: pd.DataFrame, template: pd.DataFrame, predictions) -> pd.DataFrame:
    """One-to-one identifier join preserves template order without positional guesses."""
    _require_columns(template, ["load_id", "predicted_rate"], "prediction template")
    for label, frame in [("validation", validation), ("template", template)]:
        if frame.load_id.isna().any() or frame.load_id.duplicated().any():
            raise ValueError(f"{label}: missing or duplicate IDs")
    if set(validation.load_id) != set(template.load_id):
        raise ValueError("Validation IDs and template IDs differ")
    values = _positive_predictions(predictions)
    if len(values) != len(validation):
        raise ValueError("Prediction count does not match validation")
    keyed = pd.DataFrame({"load_id": validation.load_id.to_numpy(), "predicted_rate": values})
    return template[["load_id"]].merge(keyed, on="load_id", how="left", validate="one_to_one", sort=False)
