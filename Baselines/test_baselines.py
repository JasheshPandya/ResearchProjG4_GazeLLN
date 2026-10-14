from pathlib import Path
import tempfile
import unittest

import numpy as np

from Baselines.maps import FrameControl, center_map, qp_offsets
from Baselines.evaluate_gaze import map_metrics, score_sequences, temporal_controls, load_targets, main
from Encoder.evaluate_compression import VariantResult, compute_summary


class BaselineTests(unittest.TestCase):
    def test_center_symmetry_and_flat_qp_policy(self):
        center = center_map(32, 48)
        self.assertAlmostEqual(float(center.sum()), 1, places=6)
        np.testing.assert_allclose(center, center[::-1, ::-1])
        offsets = qp_offsets(center, 68, 120)
        self.assertAlmostEqual(float(offsets.min()), -3, places=5)
        self.assertAlmostEqual(float(offsets.max()), 10, places=5)
        np.testing.assert_array_equal(qp_offsets(np.ones((32, 48)), 68, 120), 0)

    def test_motion_is_causal_and_resets_per_video(self):
        control = FrameControl("motion", (8, 12), spatial_sigma=0)
        blank = np.zeros((16, 24, 3), dtype=np.uint8)
        changed = blank.copy()
        changed[:4, :4] = 255
        uniform = np.full((8, 12), 1 / 96)
        np.testing.assert_allclose(control(blank), uniform)
        motion = control(changed)
        self.assertGreater(float(motion[:2, :2].sum()), 0.99)
        np.testing.assert_allclose(control(changed), uniform)
        np.testing.assert_allclose(FrameControl("motion", (8, 12), spatial_sigma=0)(changed), uniform)

    def test_metric_direction_and_undefined_correlation(self):
        target = center_map(8, 12)
        same = map_metrics(target, target)
        self.assertAlmostEqual(same["kl_gt_pred"], 0, places=10)
        self.assertAlmostEqual(same["similarity"], 1, places=6)
        self.assertAlmostEqual(same["cc"], 1)
        self.assertAlmostEqual(same["centroid_error_grid_px"], 0)
        flat = map_metrics(np.ones_like(target), target)
        self.assertIsNone(flat["cc"])
        self.assertGreater(flat["kl_gt_pred"], 0)

    def test_invalid_targets_are_excluded_for_all_methods(self):
        targets = np.stack([center_map(4, 6), np.zeros((4, 6)), center_map(4, 6)])
        predictions = np.repeat(targets[:1], 3, axis=0)
        rows, summary = score_sequences({"center": predictions}, targets, np.array([True, False, True]))
        self.assertEqual([r["frame"] for r in rows], [0, 2])
        self.assertEqual(summary[0]["frames_evaluated"], 2)
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            score_sequences({"wrong": predictions[:2]}, targets, np.array([True, False, True]))

    def test_lag_never_uses_current_or_future_prediction(self):
        predictions = np.eye(6).reshape(6, 2, 3)
        result = temporal_controls(predictions, seed=12)
        np.testing.assert_allclose(result["gaze_lag"][0], 1 / 6)
        np.testing.assert_array_equal(result["gaze_lag"][1:], predictions[:-1])
        np.testing.assert_array_equal(result["gaze_static"], np.repeat(predictions[:1], 6, axis=0))
        self.assertTrue((result["gaze_shifted"] != predictions).any(axis=(1, 2)).all())
        np.testing.assert_array_equal(result["gaze_shifted"], temporal_controls(predictions, 12)["gaze_shifted"])

    def test_empty_valid_heatmap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "targets.npz"
            np.savez(path, heatmaps=np.zeros((2, 4, 6)), valid=np.array([True, False]))
            with self.assertRaisesRegex(ValueError, "nonempty target"):
                load_targets(path)

    def test_diem_offset_excludes_unused_target_and_records_interval(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            targets = np.stack([np.zeros((4, 6)), center_map(4, 6), center_map(4, 6)])
            np.savez(root / "targets.npz", heatmaps=targets, valid=np.array([False, True, True]))
            main(["--targets", str(root / "targets.npz"), "--target-start-index", "1", "--frames", "2",
                  "--out-dir", str(root / "results")])
            result = json.loads((root / "results/results.json").read_text())
            self.assertEqual(result["protocol"]["target_start_index"], 1)
            self.assertEqual(result["protocol"]["total_frames"], 2)
            self.assertEqual(result["summary"][0]["frames_evaluated"], 2)
            self.assertIsNone(result["summary"][0]["cc"])

    def test_pairwise_bd_rate_compares_gaze_to_each_baseline(self):
        results = []
        for method, scale in (("uniform", 1), ("center", .9), ("motion", .85), ("gaze", .8)):
            for rate, quality in zip((100, 200, 400, 800), (30, 33, 36, 39)):
                results.append(VariantResult("clip", method, str(rate), "unused", bitrate_kbps=rate * scale,
                                             psnr_roi=quality))
        pairs = {(r["anchor_method"], r["test_method"]): r for r in compute_summary(results)}
        self.assertEqual(len(pairs), 5)
        self.assertAlmostEqual(pairs["uniform", "gaze"]["bdrate_psnr_roi_pct"], -20, places=8)
        self.assertAlmostEqual(pairs["center", "gaze"]["bdrate_psnr_roi_pct"], (0.8 / 0.9 - 1) * 100, places=8)
        self.assertIsNone(pairs["center", "gaze"]["n_uniform_points"])


if __name__ == "__main__":
    unittest.main()
