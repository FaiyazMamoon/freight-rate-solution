"""Behavioral checks for schema boundaries, fold-local state and output identity."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from freight_rate.pipeline import (ChartRateModel, MainRateModel, align_predictions,
    chart_features, load_models, main_features, save_models)


def fixture():
    rng = np.random.default_rng(1729)
    n = 240
    cities = ["A", "B", "C"]
    lookup = {"A": [35., -80.], "B": [38., -85.], "C": [40., -90.]}
    pickup = np.array(cities)[np.arange(n) % 3]
    delivery = np.array(cities)[(np.arange(n) + 1) % 3]
    distance = rng.uniform(150, 600, n)
    frame = pd.DataFrame(dict(load_id=[f"TR-{i}" for i in range(n)], pickup=pickup,
        delivery=delivery, pickup_lat=[lookup[c][0] for c in pickup],
        pickup_lon=[lookup[c][1] for c in pickup], delivery_lat=[lookup[c][0] for c in delivery],
        delivery_lon=[lookup[c][1] for c in delivery], distance=distance,
        equipment=np.array(["Dry Van", "Reefer", "Flatbed"])[np.arange(n) % 3],
        weight=rng.uniform(15000, 40000, n), date=pd.date_range("2025-01-01", periods=n).strftime("%Y-%m-%d"),
        market_index=rng.uniform(.8, 1.2, n), quote_signal=rng.uniform(1., 3., n)))
    frame["posted_rate"] = 150 + 2 * distance + rng.normal(0, 8, n)
    frame.loc[0, "weight"], frame.loc[1, "weight"] = -32000., np.nan
    return frame


class PipelineBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = fixture()
        cls.main = MainRateModel().fit(cls.train)
        cls.chart = ChartRateModel().fit(cls.train)
        cls.inference = cls.train.drop(columns="posted_rate").iloc[:8].copy()

    def test_schema_rejects_target_and_missing_market(self):
        with self.assertRaisesRegex(ValueError, "unexpected columns"):
            self.main.predict(self.train.iloc[:8])
        with self.assertRaisesRegex(ValueError, "market_index"):
            self.main.predict(self.inference.drop(columns="market_index"))

    def test_weight_policies_preserve_distinct_evaluated_behavior(self):
        main = main_features(self.train)
        chart = chart_features(self.train)
        self.assertEqual(main.loc[0, "weight"], 32000.)
        self.assertEqual(main.loc[0, "weight_negative"], 1)
        self.assertEqual(main.loc[1, "weight_missing_original"], 1)
        self.assertTrue(np.isnan(main.loc[1, "weight"]))
        self.assertEqual(chart.loc[0, "weight"], -32000.)

    def test_quote_is_excluded_and_universal_ignores_market(self):
        changed = self.inference.copy()
        changed["quote_signal"] = 99999.
        np.testing.assert_array_equal(self.main.predict(changed), self.main.predict(self.inference))
        changed["market_index"] = 99999.
        pd.testing.assert_frame_equal(self.chart.transform(changed), self.chart.transform(self.inference))
        self.assertFalse(any("quote" in column for column in self.main.feature_columns))
        self.assertFalse(any("quote" in column or "market" in column for column in self.chart.feature_columns))

    def test_unknown_cities_use_native_missing_without_growing_vocabulary(self):
        incoming = self.inference.copy()
        incoming["pickup"] = "Unseen future city"
        before = list(self.main.categories["pickup"])
        transformed = self.main.transform(incoming)
        self.assertTrue(transformed.pickup.isna().all())
        self.assertEqual(before, self.main.categories["pickup"])
        self.assertTrue(np.isfinite(self.main.predict(incoming)).all())

    def test_december_adaptation_only_recovers_static_coordinates(self):
        reduced = self.inference[["pickup", "delivery", "distance", "equipment", "weight", "date"]].copy()
        reduced["predicted_rate"] = np.nan
        adapted = self.chart.adapt(reduced)
        self.assertTrue(set(["market_index", "quote_signal", "load_id"]).isdisjoint(adapted))
        self.assertEqual(self.chart.transform(reduced).shape[1], 31)
        np.testing.assert_allclose(self.chart.predict(reduced), self.chart.predict(self.inference))
        reduced.loc[reduced.index[0], "pickup"] = "Unknown city"
        with self.assertRaisesRegex(ValueError, "no training coordinates"):
            self.chart.predict(reduced)

    def test_id_join_tracks_ids_and_rejects_duplicates(self):
        frame = self.inference.copy()
        template = pd.DataFrame(dict(load_id=list(reversed(frame.load_id)), predicted_rate=np.nan))
        predictions = np.arange(1, len(frame) + 1, dtype=float)
        aligned = align_predictions(frame, template, predictions)
        self.assertEqual(aligned.load_id.tolist(), template.load_id.tolist())
        self.assertEqual(aligned.predicted_rate.tolist(), list(reversed(predictions)))
        template.loc[0, "load_id"] = template.loc[1, "load_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            align_predictions(frame, template, predictions)

    def test_saved_artifacts_reload_identically(self):
        with tempfile.TemporaryDirectory() as folder:
            save_models(self.main, self.chart, Path(folder))
            main, chart = load_models(Path(folder))
            np.testing.assert_array_equal(main.predict(self.inference), self.main.predict(self.inference))
            np.testing.assert_array_equal(chart.predict(self.inference), self.chart.predict(self.inference))


if __name__ == "__main__":
    unittest.main()
