"""Numerical regressions for the GazeLNN metric conventions.
   Use to check if all the evaluation metrics are working as intended.
Run with: python -m unittest -v test_metrics
"""

import unittest
from unittest.mock import patch

import numpy as np

from metrics import evaluate_metrics, scanmatch_score, scanpath_to_string, time_delay_embedding


class MetricsTests(unittest.TestCase):
    def test_identical_paths(self):
        path = [[16, 16], [112, 80], [240, 144], [368, 240]]
        scores = evaluate_metrics(path, path)
        self.assertEqual(set(scores), {"Levenshtein", "ScanMatch", "Hausdorff", "Frechet", "FastDTW", "TDE"})
        for name, score in scores.items():
            self.assertAlmostEqual(score, 1.0 if name == "ScanMatch" else 0.0)

    def test_two_character_levenshtein_convention(self):
        # One cell change in both axes costs two character substitutions.
        self.assertEqual(scanpath_to_string([[16, 240]])[0], "aA")
        self.assertEqual(scanpath_to_string([[48, 208]])[0], "bB")
        self.assertEqual(evaluate_metrics([[16, 240]], [[48, 208]])["Levenshtein"], 2)
        # Inserting one fixation adds two characters.
        self.assertEqual(evaluate_metrics([[16, 240]], [[16, 240], [48, 208]])["Levenshtein"], 2)

    def test_uniform_grid_at_non_divisible_resolution(self):
        # True cell width is 10.5; integer division would place x=10 in column 1.
        text, cells = scanpath_to_string([[10, 95], [11, 95]], height=100, width=126)
        self.assertEqual((text, cells), ("aAaB", [0, 1]))
        # True cell height is 12.5; integer division would put y=88 in row 1.
        self.assertEqual(scanpath_to_string([[0, 88]], height=100, width=126), ("aA", [0]))

    def test_grid_edges_are_clipped(self):
        text, cells = scanpath_to_string([[0, 0], [384, 256], [-1, -1], [400, 300]])
        self.assertEqual((text, cells), ("hAaLhAaL", [84, 11, 84, 11]))
        # Clipping the symbolic grid must not erase continuous spatial errors.
        scores = evaluate_metrics([[400, 300]], [[384, 256]])
        self.assertEqual(scores["Levenshtein"], 0)
        self.assertAlmostEqual(scores["Hausdorff"], np.hypot(16, 44))

    def test_scanmatch_spatial_costs_and_length_normalization(self):
        self.assertEqual(scanmatch_score([0, 1, 12], [0, 1, 12]), 1)
        self.assertAlmostEqual(scanmatch_score([0], [1]), 2.5 / 3.5)
        self.assertAlmostEqual(scanmatch_score([0], [13]), (3.5 - np.sqrt(2)) / 3.5)
        self.assertEqual(scanmatch_score([0], [95]), 0)
        self.assertEqual(scanmatch_score([0], [0, 0]), 0.5)
        self.assertEqual(scanmatch_score([], [0]), 0)
        # With zero gaps, a shared fixation can align across an intervening point.
        self.assertAlmostEqual(scanmatch_score([0, 95, 1], [0, 1]), 2 / 3)

    def test_scanmatch_custom_grid(self):
        # Cells 0 and 4 are vertically adjacent on a 4-column grid.
        self.assertAlmostEqual(scanmatch_score([0], [4], Xbins=4, Ybins=2), 2.5 / 3.5)

    def test_tde_direction_and_window_normalization(self):
        gt = [[0, 0], [0, 0], [0, 0]]
        pred = [[0, 0], [0, 0], [0, 0], [30, 0]]
        # Predicted windows cost 0 and 30/3. In the reverse direction cost is 0.
        self.assertEqual(time_delay_embedding(gt, pred), 5)
        self.assertEqual(time_delay_embedding(pred, gt), 0)
        self.assertEqual(time_delay_embedding(gt, pred, distance_mode="Hausdorff"), 10)
        self.assertEqual(evaluate_metrics(pred, gt)["TDE"], 5)
        # Flattened-window norm divided by k, rather than mean per-point norm.
        self.assertAlmostEqual(time_delay_embedding(gt, [[3, 4]] * 3), 5 / np.sqrt(3))

    def test_tde_unsigned_coordinates_do_not_wrap(self):
        gt = np.zeros((3, 2), dtype=np.uint8)
        pred = np.full((3, 2), [3, 4], dtype=np.uint8)
        self.assertAlmostEqual(time_delay_embedding(pred, gt), 5 / np.sqrt(3))

    def test_short_paths_have_unavailable_tde(self):
        scores = evaluate_metrics([[0, 0]], [[3, 4]])
        for name in ("Hausdorff", "Frechet", "FastDTW"):
            self.assertEqual(scores[name], 5)
        self.assertTrue(np.isnan(scores["TDE"]))
        with self.assertRaises(ValueError):
            time_delay_embedding([[0, 0]], [[3, 4]])

    def test_repeated_points_work_for_frechet(self):
        scores = evaluate_metrics([[0, 0]] * 3, [[3, 4]] * 3)
        self.assertEqual(scores["Frechet"], 5)
        self.assertEqual(scores["Hausdorff"], 5)
        self.assertEqual(scores["FastDTW"], 15)

    def test_order_matters_to_frechet_and_dtw(self):
        path = [[0, 0], [30, 0], [60, 0]]
        scores = evaluate_metrics(path, path[::-1])
        self.assertEqual(scores["Hausdorff"], 0)
        self.assertEqual(scores["Frechet"], 60)
        self.assertEqual(scores["FastDTW"], 120)

    def test_frechet_errors_are_not_perfect_scores(self):
        with patch("metrics.similaritymeasures.frechet_dist", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                evaluate_metrics([[0, 0]] * 3, [[3, 4]] * 3)

    def test_invalid_scanpaths(self):
        for invalid in ([], [[1, 2, 3]], [1, 2], [[np.nan, 0]], [[0, np.inf]], np.empty((0, 2))):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    evaluate_metrics(invalid, [[0, 0]] * 3)
                with self.assertRaises(ValueError):
                    evaluate_metrics([[0, 0]] * 3, invalid)

    def test_invalid_metric_parameters(self):
        for options in ({"height": 0}, {"width": np.inf}, {"Xbins": 0}, {"Ybins": 2.5}, {"Xbins": 27}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                scanpath_to_string([[0, 0]], **options)
        for options in ({"k": 0}, {"k": 2.5}, {"distance_mode": "median"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                time_delay_embedding([[0, 0]] * 3, [[0, 0]] * 3, **options)
        for options in ({"threshold": 0}, {"threshold": np.nan}, {"gap_penalty": np.inf}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                scanmatch_score([0], [0], **options)
        with self.assertRaises(ValueError):
            scanmatch_score([96], [0])


if __name__ == "__main__":
    unittest.main()
