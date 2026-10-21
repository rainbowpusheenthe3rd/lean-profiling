"""Tests. Run with ``python -m unittest discover tests`` (standard library only).

Expected values for the statistics were produced by scipy, which this package replaces.
"""

import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from lean_profiling import ProfileReport, stats
from lean_profiling.config import Settings
from lean_profiling.svg import max_n_ticks, png_rgb, scalar_format
from lean_profiling.typeset import infer


class TestStats(unittest.TestCase):
    def test_chi2_sf_matches_scipy(self):
        self.assertAlmostEqual(stats.chi2_sf(3.5, 4), 0.477878344488724, places=12)
        self.assertAlmostEqual(stats.chi2_sf(0.2, 1), 0.6547208460185768, places=12)
        self.assertAlmostEqual(stats.chi2_sf(120.0, 49) / 6.876961427062773e-08, 1.0, places=9)

    def test_chisquare_matches_scipy(self):
        result = stats.chisquare(np.array([10, 12, 9, 30, 11]))
        self.assertAlmostEqual(result["statistic"], 21.47222222222222, places=10)
        self.assertAlmostEqual(result["pvalue"], 0.00025520571749432106, places=14)

    def test_contingency_matches_scipy(self):
        self.assertAlmostEqual(stats.chi2_contingency_statistic(np.array([[10, 20], [30, 5]])), 16.57848214285714)
        self.assertAlmostEqual(
            stats.chi2_contingency_statistic(np.array([[10, 20, 5], [30, 5, 7]])), 18.852777777777785
        )

    def test_entropy_and_spearman_match_scipy(self):
        self.assertAlmostEqual(stats.entropy_base2(np.array([5, 3, 2])), 1.4854752972273346)
        x = pd.Series([1, 2, 3, 4, 5, np.nan, 7])
        y = pd.Series([2, 1, 4, 3, 7, 8, np.nan])
        self.assertAlmostEqual(stats.spearman(x, y), 0.8)


class TestTypeInference(unittest.TestCase):
    def setUp(self):
        self.config = Settings.build()

    def check(self, values, expected):
        self.assertEqual(infer(pd.Series(values), self.config)[0], expected)

    def test_types(self):
        self.check(np.arange(100, dtype=float), "Numeric")
        self.check([0, 1, 2] * 30, "Categorical")
        self.check(["yes", "no"] * 30, "Boolean")
        self.check([f"{i}.5" for i in range(100)], "Numeric")
        self.check(pd.date_range("2020-01-01", periods=50), "DateTime")
        self.check([f"word {i}" for i in range(100)], "Text")
        self.check(["a", "b", "c"] * 30, "Categorical")
        self.check([np.nan] * 10, "Unsupported")
        self.check([[1, 2]] * 10, "Unsupported")


class TestTicks(unittest.TestCase):
    def test_matplotlib_tick_choices(self):
        # Cases checked against matplotlib's MaxNLocator / ScalarFormatter.
        np.testing.assert_allclose(max_n_ticks(-29.95, 628.95, 9)[1:-1], [0, 100, 200, 300, 400, 500, 600])
        np.testing.assert_allclose(max_n_ticks(-29.95, 628.95, 6)[1:-1], [0, 200, 400, 600])
        ticks = max_n_ticks(0, 17.85, 8)
        labels, offset = scalar_format(ticks, (0, 17.85))
        self.assertEqual(labels[:3], ["0.0", "2.5", "5.0"])
        self.assertEqual(offset, "")
        labels, offset = scalar_format(np.array([4.7e7, 4.8e7, 4.9e7]), (4.65e7, 4.95e7))
        self.assertEqual((labels, offset), (["4.7", "4.8", "4.9"], "1e7"))


class TestPng(unittest.TestCase):
    def test_png_is_well_formed(self):
        data = png_rgb(np.zeros((3, 2, 3), dtype=np.uint8))
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        length, kind = struct.unpack(">I4s", data[8:16])
        self.assertEqual(kind, b"IHDR")
        self.assertEqual(struct.unpack(">II", data[16:24]), (2, 3))
        crc = struct.unpack(">I", data[16 + length : 20 + length])[0]
        self.assertEqual(crc, zlib.crc32(data[12 : 16 + length]) & 0xFFFFFFFF)


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame(
            {
                "x": rng.normal(size=n),
                "y": rng.integers(0, 3, n),
                "flag": rng.choice(["yes", "no"], n),
                "when": pd.date_range("2021-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
                "note": [f"note number {i}" for i in range(n)],
            }
        )
        df.loc[::7, "x"] = np.nan
        cls.report = ProfileReport(pd.concat([df, df.head(3)], ignore_index=True), title="Test report")

    def test_json_layout(self):
        data = json.loads(self.report.to_json())
        self.assertEqual(
            list(data),
            ["analysis", "time_index_analysis", "table", "variables", "scatter", "correlations", "missing",
             "alerts", "package", "sample", "duplicates"],
        )
        self.assertEqual(data["table"]["n_duplicates"], 3)
        self.assertEqual({k: v["type"] for k, v in data["variables"].items()},
                         {"x": "Numeric", "y": "Categorical", "flag": "Boolean", "when": "DateTime", "note": "Text"})
        self.assertIn("[x] 44 (14.5%) missing values", data["alerts"])

    def test_html_sections(self):
        html = self.report.to_html()
        for section in ["Overview", "Variables", "Interactions", "Correlations", "Missing values", "Sample",
                        "Duplicate rows"]:
            self.assertIn(f'<h1 class="section-name">{section}</h1>', html)
        self.assertNotIn("matplotlib", html)

    def test_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.report.to_file(Path(tmp) / "profiling.json")
            self.report.to_file(Path(tmp) / "report.html")
            self.assertTrue((Path(tmp) / "report.html").read_text(encoding="utf-8").startswith("<!doctype html>"))
            json.loads((Path(tmp) / "profiling.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
