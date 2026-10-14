"""Meaningful plotting/export regressions; run with python -m unittest."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from Encoder import evaluate_compression as ec
from Encoder import plot_bdrate as plot


def row(method, label, rate):
    return dict(method=method, label=label, bitrate_kbps=rate, frames_evaluated=4,
                psnr_roi=35, fps=10)


class CompressionPlotTests(unittest.TestCase):
    def test_pair_uses_bitrate_not_equal_crf(self):
        rows = [row("uniform", "crf18", 400), row("uniform", "crf24", 200),
                row("uniform", "crf30", 100), row("gaze", "crf18", 210),
                row("gaze", "crf24", 110), row("gaze", "crf30", 50)]
        u, g, pair = plot.select_bitrate_pair(rows)
        self.assertEqual((u["label"], g["label"]), ("crf24", "crf18"))
        self.assertAlmostEqual(pair["gaze_rate_delta_pct"], 5)
        self.assertTrue(pair["within_five_percent"])
        self.assertIn("210.0 kbps", plot.pair_caption(pair))
        with self.assertRaisesRegex(ValueError, "common measured bitrate"):
            plot.select_bitrate_pair([row("uniform", "u", 100), row("gaze", "g", 300)])

    def test_stale_vmaf_logs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vmaf_logs").mkdir()
            (root / "vmaf_logs/vmaf_clip.json").write_text(json.dumps({"frames": [
                {"frameNum": i, "metrics": {"vmaf": 60}} for i in range(4)]}))
            r = dict(path="clip.mp4", frames_evaluated=4, vmaf_frames=4, vmaf_global=80)
            with self.assertRaisesRegex(ValueError, "current aggregate"):
                plot.read_frame_series(r, root)
            r["vmaf_global"] = None
            with self.assertRaisesRegex(ValueError, "No current"):
                plot.read_frame_series(r, root)

    def test_legacy_windows_remain_averages_and_use_partial_window_weights(self):
        r = dict(vmaf_avg_f0_19="60", vmaf_avg_f20_39="80", vmaf_global=64)
        series = plot.read_windowed_vmaf(r, 25)
        np.testing.assert_array_equal(series["vmaf_window_mean"], [60, 80])
        np.testing.assert_array_equal(series["window_centers"], [9.5, 22])
        self.assertNotIn("vmaf_global", series)
        r["vmaf_global"] = 70
        with self.assertRaisesRegex(ValueError, "current aggregate"):
            plot.read_windowed_vmaf(r, 25)

    def test_frame_csv_preserves_missing_vmaf_and_rejects_misalignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = ec.VariantResult("clip", "gaze", "crf24", "g.mp4", frames_evaluated=4, fps=10)
            ec.write_frame_metrics(result, root, {"psnr_roi": [30, 31, 32, 33],
                                                "ssim_roi": [.9] * 4, "vmaf_global": []})
            info = dict(frame_metrics_path=result.frame_metrics_path, frames_evaluated=4)
            series = plot.read_frame_series(info, root)
            self.assertNotIn("vmaf_global", series)
            np.testing.assert_array_equal(series["psnr_roi"], [30, 31, 32, 33])
            p = root / result.frame_metrics_path
            text = p.read_text()
            p.write_text(text.replace("1,0.1,", "0,0.1,"))
            with self.assertRaisesRegex(ValueError, "misaligned"):
                plot.read_frame_series(info, root)

    def test_crop_uses_shared_mask_and_filename_is_portable(self):
        mask = np.zeros((64, 128), bool)
        mask[:, :32] = True
        for roi in (True, False):
            y, x, h, w = plot.choose_crop(mask, roi, 16, 16)
            self.assertEqual(bool(mask[y:y+h, x:x+w].all()), roi)
        self.assertNotEqual(plot.safe_name("Clip"), plot.safe_name("clip"))
        self.assertNotEqual(plot.safe_name("a/b"), plot.safe_name("a\\b"))
        self.assertNotIn("/", plot.safe_name("../../CON"))

    def test_suite_exports_actual_images_and_bd_values(self):
        with tempfile.TemporaryDirectory(prefix="gaze plots ") as tmp:
            root = Path(tmp)
            results = root / "evaluation"
            results.mkdir()
            qp = root / "qp"
            qp.mkdir()
            (qp / "metadata.json").write_text(json.dumps(dict(
                mb_width=8, mb_height=6, num_frames=4, qp_min=-3, qp_max=10)))
            for i in range(4):
                np.tile(np.linspace(-3, 10, 8, dtype=np.float32), (6, 1)).tofile(qp / f"qpoffset_{i:06d}.bin")
            for name, change in (("source", 0), ("uniform", 3), ("gaze", 2)):
                writer = cv2.VideoWriter(str(root / f"{name}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (128, 96))
                self.assertTrue(writer.isOpened(), "MJPEG fixture encoder unavailable")
                for frame in range(4):
                    image = np.tile(np.arange(128, dtype=np.uint8), (96, 1))
                    image = np.clip(image.astype(int) + change + frame, 0, 255).astype(np.uint8)
                    writer.write(np.repeat(image[:, :, None], 3, axis=2))
                writer.release()
            variants, measurements = [], []
            for method, multiplier in (("uniform", 1), ("gaze", .8)):
                for i, rate in enumerate((100, 200, 400, 800)):
                    label = f"point{i}"
                    variant = dict(method=method, label=label, path=f"{method}.avi")
                    variants.append(variant)
                    result = ec.VariantResult("clip", method, label, str(root / variant["path"]),
                        width=128, height=96, frames_evaluated=4, fps=10,
                        bitrate_kbps=rate * multiplier, psnr_roi=30 + 5*i,
                        psnr_periphery=30 + 5*i, vmaf_global=60 + 10*i,
                        vmaf_frames=4, ssim_roi=.8 + .03*i, ms_ssim_global=.9 + .02*i,
                        encode_time_s=.1, qp_generation_time_s=0 if method == "uniform" else .2)
                    ec.write_frame_metrics(result, results, {"psnr_roi": [30 + 5*i] * 4,
                        "ssim_roi": [.8 + .03*i] * 4, "vmaf_global": [59 + 10*i, 60 + 10*i, 60 + 10*i, 61 + 10*i]})
                    measurements.append(result)
            ec.write_detailed_csv(measurements, results / "results_detailed.csv")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(roi_percentile=25, clips=[dict(name="clip",
                reference="source.avi", roi_source="qp", variants=variants, scene_cut_frames=[2])])) )
            index = plot.generate_plots(results / "results_detailed.csv", manifest_path=manifest)
            self.assertEqual(index["skipped"], [])
            kinds = {artifact["kind"] for artifact in index["artifacts"]}
            self.assertTrue({"rd_panels", "runtime", "temporal_quality", "spatial_allocation",
                             "visual_comparison", "bdrate_summary"}.issubset(kinds))
            for artifact in index["artifacts"]:
                for filename in artifact["files"]:
                    path = results / "plots" / filename
                    self.assertGreater(path.stat().st_size, 1000)
                    if path.suffix == ".png":
                        self.assertIsNotNone(cv2.imread(str(path)))
            panels = next(a for a in index["artifacts"] if a["kind"] == "rd_panels")
            self.assertAlmostEqual(panels["metrics"][1]["bdrate_pct"], -20, places=7)
            temporal = next(a for a in index["artifacts"] if a["kind"] == "temporal_quality")
            self.assertEqual(temporal["metrics"], ["vmaf_global", "psnr_roi", "ssim_roi"])
            self.assertEqual(len(plot.plt.get_fignums()), 0)

    def test_evaluator_calls_plotting_after_exports_and_opt_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ref.mp4").touch()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"clips": [{"name": "clip", "reference": "ref.mp4",
                "roi_source": "qp", "variants": [{"method": "uniform", "path": "u.mp4"}]}]}))
            for disabled in (False, True):
                def generate(csv_path, **kwargs):
                    self.assertTrue(csv_path.exists())
                    self.assertTrue((csv_path.parent / "results.json").exists())
                    return {"skipped": []}
                argv = ["evaluate", "--manifest", str(manifest), "--out_dir", str(root / "results")]
                if disabled:
                    argv.append("--no-plots")
                with patch("sys.argv", argv), patch.object(ec, "find_tool", side_effect=lambda name, **kw: name), \
                     patch.object(ec, "vmaf_available", return_value=False), patch.object(ec, "load_roi_source"), \
                     patch.object(ec, "evaluate_variant", return_value=ec.VariantResult("clip", "uniform", "u", "u.mp4")), \
                     patch.object(plot, "generate_plots", side_effect=generate) as plotting:
                    ec.main()
                self.assertEqual(plotting.call_count, 0 if disabled else 1)


if __name__ == "__main__":
    unittest.main()
