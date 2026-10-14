import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from Encoder import build, runtime, workflow
from Encoder.run_ablation_sweep import relative_path


class PortabilityTests(unittest.TestCase):
    def test_native_executable_name(self):
        for platform, name in (("nt", "encoder.exe"), ("posix", "encoder")):
            with patch.object(build, "os", SimpleNamespace(name=platform)):
                self.assertEqual(build.encoder_path(Path("native")), Path("native") / name)

    def test_windows_environment_only_adds_configured_msys_prefix(self):
        original = {"PATH": "original", "MSYS2_MINGW_PREFIX": "custom-msys/ucrt64"}
        windows_path = str(Path(original["MSYS2_MINGW_PREFIX"]) / "bin") + ";original"
        for platform, separator, expected in (("posix", ":", "original"),
                                               ("nt", ";", windows_path)):
            fake_os = SimpleNamespace(environ=original, name=platform, pathsep=separator)
            with patch.object(runtime, "os", fake_os):
                self.assertEqual(runtime.subprocess_environment()["PATH"], expected)
        self.assertEqual(original["PATH"], "original")

    def test_tool_discovery_in_unactivated_windows_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool = Path(tmp) / "Scripts" / "ffprobe.exe"
            tool.parent.mkdir()
            tool.touch()
            fake_os = SimpleNamespace(name="nt", access=lambda *_: True, X_OK=1)
            with patch.object(runtime, "os", fake_os), patch.object(runtime.sys, "prefix", tmp), \
                 patch.object(runtime.shutil, "which", return_value=None):
                self.assertEqual(runtime.find_tool("ffprobe", env={"PATH": ""}), str(tool.resolve()))
                with self.assertRaisesRegex(RuntimeError, "Cannot find"):
                    runtime.find_tool("ffprobe", explicit="wrong", env={"PATH": ""})

    def test_build_preserves_spaces_in_pkg_config_paths(self):
        with tempfile.TemporaryDirectory(prefix="build with spaces ") as tmp:
            result = Mock(returncode=0, stdout='-I"C:/SDK with spaces/include" -L"C:/SDK with spaces/lib" -lx264')
            with patch.object(build, "find_tool", side_effect=lambda name, **kwargs: name), \
                 patch.object(build.subprocess, "run", side_effect=[result, Mock()]) as run:
                output = build.build_encoder(Path(tmp) / "native encoder", compiler="clang", env={})
            command = run.call_args.args[0]
            self.assertIn("-IC:/SDK with spaces/include", command)
            self.assertIn(str(output), command)
            self.assertTrue(Path(command[command.index("-o") - 1]).is_absolute())
            self.assertNotIn("shell", run.call_args.kwargs)

    def test_missing_development_libraries_give_setup_hint(self):
        with patch.object(build, "find_tool", side_effect=lambda name, **kwargs: name), \
             patch.object(build.subprocess, "run", return_value=Mock(returncode=1, stderr="x264 missing")):
            with self.assertRaisesRegex(RuntimeError, "Encoder/README.md"):
                build.build_encoder(compiler="gcc", env={})

    def test_cached_build_requires_matching_source_os_and_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "encoder"
            binary.touch()
            self.assertFalse(build.is_current_encoder(binary))
            identity = build.build_identity()
            build.build_stamp(binary).write_text(json.dumps(identity))
            self.assertTrue(build.is_current_encoder(binary))
            with patch.object(build.platform, "system", return_value="another OS"):
                self.assertFalse(build.is_current_encoder(binary))
            for key in identity:
                build.build_stamp(binary).write_text(json.dumps({**identity, key: "wrong"}))
                self.assertFalse(build.is_current_encoder(binary))

    def test_qp_reuse_checks_every_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata.json").write_text(json.dumps({"num_frames": 3, "mb_width": 2, "mb_height": 2}))
            for i in range(3):
                (root / f"qpoffset_{i:06d}.bin").write_bytes(bytes(16))
            with patch.object(workflow, "video_info", return_value=(3, 24, 20, 30)):
                self.assertIsNone(workflow.prepare_qp_maps("video", root, reuse=True))
                (root / "qpoffset_000002.bin").write_bytes(bytes(12))
                with self.assertRaisesRegex(RuntimeError, "incomplete QP"):
                    workflow.validate_qp_maps("video", root)

    def test_failed_encode_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "old.mp4"
            output.write_bytes(b"previous video")
            with patch.object(workflow, "run_cmd", side_effect=RuntimeError("encoder failed")):
                with self.assertRaises(RuntimeError):
                    workflow.encode_video("encoder", "source", output, "maps", 23, "medium")
            self.assertEqual(output.read_bytes(), b"previous video")

    def test_manifest_path_round_trip_and_other_windows_drive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = relative_path(root / "source with spaces.mp4", root / "output")
            self.assertEqual(result, "../source with spaces.mp4")
            with patch("Encoder.run_ablation_sweep.os.path.relpath", side_effect=ValueError("different drives")):
                self.assertEqual(relative_path(root / "source.mp4", root), (root / "source.mp4").as_posix())


if __name__ == "__main__":
    unittest.main()
