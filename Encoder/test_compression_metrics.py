"""Run with python -m unittest -v Encoder.test_compression_metrics."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from Encoder import evaluate_compression as ec
from Encoder import plot_bdrate as plot


class CompressionMetricTests(unittest.TestCase):
    def test_psnr_known_error_and_identity(self):
        error = ec.squared_error_map(np.zeros((4, 4), np.uint8), np.full((4, 4), 10, np.uint8))
        self.assertEqual(float(error.mean()), 100)
        self.assertAlmostEqual(ec.psnr_from_se_map(error), 28.1308036086791)
        self.assertTrue(np.isinf(ec.psnr_from_mse(0)))
        self.assertTrue(np.isnan(ec.psnr_from_se_map(error, np.zeros((4, 4), bool))))

    def test_ms_ssim_constant_luminance_change(self):
        a = np.full((192, 192), 100, np.uint8)
        b = np.full((192, 192), 110, np.uint8)
        self.assertEqual(ec.ms_ssim(a, a), 1)
        # Constant patches have contrast/structure=1 at every scale. Only the
        # coarsest-scale luminance term contributes, with exponent 0.1333.
        expected = ((2 * 100 * 110 + 6.5025) / (100 ** 2 + 110 ** 2 + 6.5025)) ** 0.1333
        self.assertAlmostEqual(ec.ms_ssim(a, b), expected, places=7)
        self.assertAlmostEqual(ec.ms_ssim(a, b), ec.ms_ssim(b, a), places=7)
        self.assertTrue(np.isnan(ec.ms_ssim(a[:64, :64], b[:64, :64])))

    def test_quality_tails_use_frames(self):
        self.assertEqual(ec.score_percentile([10, 20, 30, 40, 50]), 12)
        self.assertEqual(ec.score_percentile([10, np.nan, 30]), 11)
        self.assertIsNone(ec.score_percentile([]))
        self.assertEqual(ec.score_percentile([float("inf")] * 3), float("inf"))
        windows = ec.vmaf_windowed_averages(list(range(23)))
        self.assertEqual(windows, [(0, 19, 9.5), (20, 22, 21)])

    def test_roi_flat_ties_and_partial_edge_macroblocks(self):
        source = ec.RoiSource(2, 2, -3, 10, 1, Path("unused"))
        with patch.object(source, "saliency_grid", return_value=np.ones((2, 2))):
            mask = source.roi_mask(0, 20, 20, 25)
        self.assertEqual(mask.shape, (20, 20))
        self.assertEqual(mask.sum(), 16 * 16)
        self.assertTrue(mask[:16, :16].all())
        self.assertFalse(mask[16:, :].any())
        self.assertFalse(mask[:, 16:].any())

    def test_bd_rate_known_twenty_percent_saving(self):
        rates = np.array([100, 200, 400, 800])
        qualities = [30, 35, 40, 45]
        value, note = ec.bd_rate(rates, qualities, rates * 0.8, qualities)
        self.assertAlmostEqual(value, -20, places=8)
        self.assertEqual(note, "")
        self.assertEqual(plot.compute_bdrate(rates, qualities, rates * 0.8, qualities), (value, note))
        delta, _ = ec.bd_quality(rates, qualities, rates, np.array(qualities) + 2)
        self.assertAlmostEqual(delta, 2, places=8)

    def test_invalid_bd_curves_are_unavailable(self):
        for rates, qualities in (([0, 200], [30, 40]), ([100, 200], [30, np.nan]),
                                 ([100, 200], [30, 30]), ([100, 100], [30, 40]),
                                 ([100, 200], [40, 30])):
            with self.subTest(rates=rates, qualities=qualities):
                value, note = ec.bd_rate(rates, qualities, [100, 200], [30, 40])
                self.assertIsNone(value)
                self.assertTrue(note)
        self.assertIsNone(ec.bd_rate([100, 200], [10, 20], [100, 200], [30, 40])[0])

    def test_timing_does_not_invent_missing_gaze_cost(self):
        res = ec.VariantResult("clip", "gaze", "crf20", "unused")
        ec.apply_timings(res, {"encode_time_s": 2}, 100)
        self.assertEqual(res.encode_fps, 50)
        self.assertIsNone(res.pipeline_fps)
        ec.apply_timings(res, {"encode_time_s": 2, "qp_generation_time_s": 3}, 100)
        self.assertEqual(res.pipeline_time_s, 5)
        self.assertEqual(res.pipeline_fps, 20)
        self.assertEqual(res.pipeline_processing_ms_per_frame, 50)
        with self.assertRaises(ValueError):
            ec.apply_timings(res, {"encode_time_s": -1}, 100)

    def test_pixel_pooling_and_vmaf_tails_in_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp) / "dist.mp4"
            dist.touch()
            frames = [np.zeros((32, 32, 3), np.uint8)] * 2
            distorted = [np.full((32, 32, 3), value, np.uint8) for value in (10, 20)]
            def capture(sequence):
                cap = Mock()
                cap.isOpened.return_value = True
                cap.get.side_effect = lambda prop: {
                    cv2.CAP_PROP_FRAME_COUNT: 2, cv2.CAP_PROP_FPS: 25,
                    cv2.CAP_PROP_FRAME_WIDTH: 32, cv2.CAP_PROP_FRAME_HEIGHT: 32,
                }.get(prop, 0)
                cap.read.side_effect = [(True, frame) for frame in sequence]
                return cap
            roi = Mock(num_frames=2)
            mask = np.zeros((32, 32), bool)
            mask[:, :16] = True
            roi.roi_mask.return_value = mask
            info = dict(width=32, height=32, duration_s=.08, size_mb=1, bitrate_kbps=10,
                        bitrate_source="video_stream", fps=25, frame_count=2)
            with patch.object(ec, "get_media_info", return_value=info), \
                 patch.object(ec.cv2, "VideoCapture", side_effect=[capture(frames), capture(distorted)]), \
                 patch.object(ec, "compute_vmaf", return_value=(75, [50, 100])):
                result = ec.evaluate_variant("clip", Path("ref.mp4"),
                    {"path": str(dist), "method": "gaze", "encode_time_s": 1}, roi, 25, None, Path(tmp))
            self.assertEqual(result.mse_global, 250)
            self.assertEqual(result.mse_roi, 250)
            self.assertAlmostEqual(result.psnr_global, ec.psnr_from_mse(250))
            self.assertEqual(result.roi_fraction, .5)
            self.assertEqual(result.vmaf_p5, 52.5)
            self.assertEqual(result.vmaf_min, 50)
            self.assertEqual(result.vmaf_frames, 2)
            self.assertTrue(np.isnan(result.ms_ssim_global))

    def test_shared_roi_ignores_variant_encoding_qp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ref.mp4").touch()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"clips": [{"name": "clip", "reference": "ref.mp4",
                "roi_source": "canonical", "variants": [
                    {"method": "uniform", "path": "u.mp4"},
                    {"method": "gaze", "path": "g.mp4", "qp_dir": "other"}]}]}))
            canonical = Mock()
            def result(clip, ref, variant, roi, *args, **kwargs):
                return ec.VariantResult(clip, variant["method"], "test", variant["path"])
            with patch("sys.argv", ["evaluate", "--manifest", str(manifest), "--out_dir", str(root / "out")]), \
                 patch.object(ec, "load_roi_source", return_value=canonical) as loader, \
                 patch.object(ec, "vmaf_available", return_value=False), \
                 patch.object(ec, "find_tool", side_effect=lambda name, **kwargs: name), \
                 patch.object(ec, "evaluate_variant", side_effect=result) as evaluator:
                ec.main()
            loader.assert_called_once_with(root / "canonical")
            self.assertTrue(all(call.args[3] is canonical for call in evaluator.call_args_list))
            parsed = json.loads((root / "out/results.json").read_text())
            self.assertIsNone(parsed["detailed"][0]["mse_global"])

    def test_video_mismatches_are_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp) / "dist.mp4"
            dist.touch()
            info = dict(width=32, height=32, duration_s=1, size_mb=1, bitrate_kbps=10,
                        bitrate_source="video_stream", fps=25, frame_count=25)
            expected = {cv2.CAP_PROP_FRAME_COUNT: 25, cv2.CAP_PROP_FPS: 25,
                        cv2.CAP_PROP_FRAME_WIDTH: 32, cv2.CAP_PROP_FRAME_HEIGHT: 32}
            for prop, value in ((cv2.CAP_PROP_FRAME_COUNT, 24), (cv2.CAP_PROP_FPS, 30),
                                (cv2.CAP_PROP_FRAME_WIDTH, 64)):
                with self.subTest(prop=prop):
                    ref, encoded = Mock(), Mock()
                    ref.isOpened.return_value = encoded.isOpened.return_value = True
                    ref.get.side_effect = expected.get
                    encoded_values = {**expected, prop: value}
                    encoded.get.side_effect = encoded_values.get
                    with patch.object(ec, "get_media_info", return_value=info), \
                         patch.object(ec.cv2, "VideoCapture", side_effect=[ref, encoded]), \
                         patch.object(ec, "compute_vmaf") as vmaf:
                        result = ec.evaluate_variant("clip", Path("ref.mp4"),
                            {"path": str(dist), "method": "gaze"}, Mock(), 25, None, Path(tmp))
                    self.assertIn("Incompatible videos", result.notes)
                    self.assertTrue(np.isnan(result.psnr_global))
                    ref.read.assert_not_called()
                    vmaf.assert_not_called()
                    ref.release.assert_called_once()
                    encoded.release.assert_called_once()

    def test_vmaf_does_not_reuse_an_older_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "vmaf_logs"
            logs.mkdir()
            (logs / "vmaf_dist.json").write_text(json.dumps({
                "frames": [{"metrics": {"vmaf": 99}}]}))
            with patch.object(ec, "vmaf_available", return_value=True), \
                 patch.object(ec.subprocess, "run", return_value=Mock(returncode=0, stderr="")):
                score, frames = ec.compute_vmaf(Path("ref.mp4"), Path("dist.mp4"), None, root)
            self.assertIsNone(score)
            self.assertEqual(frames, [])

    def test_exports_include_roi_bd_rate(self):
        results = []
        for method, multiplier in (("uniform", 1), ("gaze", .8)):
            for rate, quality in zip((100, 200, 400, 800), (30, 35, 40, 45)):
                results.append(ec.VariantResult("clip", method, str(rate), "unused",
                    bitrate_kbps=rate * multiplier, psnr_roi=quality))
        summary = ec.compute_summary(results)
        self.assertAlmostEqual(summary[0]["bdrate_psnr_roi_pct"], -20, places=8)
        self.assertIsNone(summary[0]["bdrate_vmaf_pct"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ec.write_summary_csv(summary, root / "summary.csv")
            ec.write_detailed_csv(results, root / "detailed.csv")
            with (root / "summary.csv").open() as f:
                self.assertIn("bdrate_psnr_roi_pct", next(csv.reader(f)))
            with (root / "detailed.csv").open() as f:
                self.assertIn("ms_ssim_global", next(csv.reader(f)))
            with patch.object(ec, "vmaf_available", return_value=False):
                ec.write_markdown_report(results, summary, root / "report.md", Path("manifest.json"), 25)
            self.assertIn("psnr_roi", (root / "report.md").read_text())
        json.dumps(ec.json_safe({"missing": float("nan"), "perfect": float("inf")}), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
