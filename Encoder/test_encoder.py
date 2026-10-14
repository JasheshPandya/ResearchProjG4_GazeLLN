"""Native regression checks; build the encoder and install FFmpeg before running."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from Encoder.build import encoder_path
from Encoder.runtime import find_tool, subprocess_environment


class NativeEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = subprocess_environment()
        cls.encoder = encoder_path().resolve()
        cls.ffmpeg = find_tool("ffmpeg", env=cls.env, required=False)
        cls.ffprobe = find_tool("ffprobe", env=cls.env, required=False)
        if not cls.encoder.is_file() or not cls.ffmpeg or not cls.ffprobe:
            if os.environ.get("GAZELNN_REQUIRE_NATIVE_TESTS") == "1":
                raise RuntimeError("Native tools are required for this test run but were not found.")
            raise unittest.SkipTest("Build the native encoder and install FFmpeg/FFprobe to run encoder tests.")

    def probe(self, path):
        result = subprocess.run([self.ffprobe, "-v", "error", "-count_frames", "-show_streams",
                                 "-of", "json", str(path)], check=True, capture_output=True, text=True,
                                env=self.env)
        return json.loads(result.stdout)["streams"]

    def test_b_frame_tail_rate_duration_and_audio(self):
        for audio in (False, True):
            with self.subTest(audio=audio), tempfile.TemporaryDirectory(prefix="encoder test ") as tmp:
                root = Path(tmp)
                source, encoded = root / "source.mp4", root / "encoded.mp4"
                empty_qp = root / "empty qp"
                empty_qp.mkdir()
                command = [self.ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i",
                           "testsrc2=size=192x128:rate=30000/1001:duration=0.8008"]
                if audio:
                    command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.8008"]
                command += ["-c:v", "libx264", "-bf", "3", "-g", "24", "-pix_fmt", "yuv420p"]
                if audio:
                    command += ["-c:a", "aac"]
                subprocess.run(command + [str(source)], check=True, capture_output=True, env=self.env)
                run = subprocess.run([str(self.encoder), str(source), str(encoded), str(empty_qp),
                                      "--crf", "23", "--preset", "medium"],
                                     check=True, capture_output=True, text=True, env=self.env)
                before, after = self.probe(source), self.probe(encoded)
                ref = next(s for s in before if s["codec_type"] == "video")
                dist = next(s for s in after if s["codec_type"] == "video")
                self.assertEqual(ref["nb_read_frames"], "24")
                self.assertGreater(ref["has_b_frames"], 0)
                self.assertEqual(dist["nb_read_frames"], ref["nb_read_frames"])
                self.assertEqual(dist["avg_frame_rate"], "30000/1001")
                self.assertAlmostEqual(float(dist["duration"]), 24 * 1001 / 30000, places=5)
                self.assertIn("read: 24, written: 24", run.stdout)
                if audio:
                    ref_audio = next(s for s in before if s["codec_type"] == "audio")
                    dist_audio = next(s for s in after if s["codec_type"] == "audio")
                    self.assertEqual(dist_audio["codec_name"], ref_audio["codec_name"])
                    self.assertEqual(dist_audio["nb_read_frames"], ref_audio["nb_read_frames"])

    def test_unwritable_output_returns_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            subprocess.run([self.ffmpeg, "-v", "error", "-f", "lavfi", "-i",
                            "testsrc2=size=64x64:duration=0.2", "-c:v", "libx264", str(source)],
                           check=True, capture_output=True, env=self.env)
            run = subprocess.run([str(self.encoder), str(source), str(root / "absent" / "output.mp4"),
                                  str(root)], capture_output=True, env=self.env)
            self.assertNotEqual(run.returncode, 0)


if __name__ == "__main__":
    unittest.main()
