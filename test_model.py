"""Regression checks for shared CoordConv architecture and notebook-free paths."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from gaze_model import GazeLLNArch, CoordConv, load_model_weights, load_gaze_model
from gaze_losses import KLDTWLoss
from project_paths import OSIE_CHECKPOINT, PROJECT_ROOT


class SharedModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = GazeLLNArch(image_h=32, image_w=48, S=8, hidden_size=16,
                                backbone_units=32, pretrained_backbone=False)

    def test_every_public_model_import_is_the_same_class(self):
        import OSIE_Modeldata
        from DIEM import DIEM_Modeldata, fine_tune
        import trainGazeLNN
        import testMITlowres
        import Evaluate_OSIE_model
        from Encoder import qp_map_generator
        for module in (OSIE_Modeldata, DIEM_Modeldata, fine_tune, trainGazeLNN,
                       testMITlowres, Evaluate_OSIE_model, qp_map_generator):
            self.assertIs(module.GazeLLNArch, GazeLLNArch)
        self.assertIs(OSIE_Modeldata.CoordConv, CoordConv)
        self.assertIs(DIEM_Modeldata.KLDTWLoss, KLDTWLoss)

    def test_both_coordconv_layers_run_and_receive_gradients(self):
        features = torch.randn(2, 960)
        prior = torch.rand(2, 4, 6)
        prior = prior / prior.sum(dim=(1, 2), keepdim=True)
        with patch.object(self.model.coordconv_in, "forward", wraps=self.model.coordconv_in.forward) as first, \
             patch.object(self.model.coordconv_out, "forward", wraps=self.model.coordconv_out.forward) as last:
            prediction, hx = self.model(features, prior, None, torch.tensor([0., 1.]))
        self.assertEqual(prediction.shape, (2, 1, 4, 6))
        self.assertEqual(hx.shape, (2, 16))
        torch.testing.assert_close(prediction.sum(dim=(1, 2, 3)), torch.ones(2))
        self.assertEqual(first.call_count, 1)
        self.assertEqual(last.call_count, 1)
        (-prediction[:, 0, 1, 2].log().mean()).backward()
        for layer in (self.model.coordconv_in, self.model.coordconv_out):
            self.assertGreater(float(layer.conv.weight.grad.abs().sum()), 0)

    def test_loading_retains_predictions_and_does_not_initialize_pretrained_backbone(self):
        self.model.eval()
        features = torch.randn(1, 960)
        prior = torch.ones(1, 1, 4, 6) / 24
        ts = torch.ones(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(self.model.state_dict(), path)
            import gaze_model
            with patch.object(gaze_model.models, "mobilenet_v3_large", wraps=gaze_model.models.mobilenet_v3_large) as backbone:
                restored = load_gaze_model(path, image_h=32, image_w=48, S=8,
                                           hidden_size=16, backbone_units=32).eval()
                self.assertIsNone(backbone.call_args.kwargs["weights"])
            with torch.no_grad():
                expected = self.model(features, prior, None, ts)
                actual = restored(features, prior, None, ts)
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_incompatible_single_coordconv_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pt"
            torch.save({"coordconv.conv.weight": torch.zeros(1, 3, 3, 3)}, path)
            with self.assertRaisesRegex(ValueError, "both input and output"):
                load_model_weights(self.model, path)

    def test_osie_training_handles_shared_output_and_updates_both_coordconv_layers(self):
        from OSIE_Modeldata import train_epoch, eval_epoch
        images = torch.rand(2, 3, 32, 48)
        targets = torch.rand(2, 3, 4, 6)
        targets = targets / targets.sum(dim=(2, 3), keepdim=True)
        batch = (images, targets, torch.tensor([[0., 1., 2.], [0., 1., 1.]]), torch.ones(2, 3, dtype=torch.bool))
        before = [layer.conv.weight.detach().clone() for layer in (self.model.coordconv_in, self.model.coordconv_out)]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss = train_epoch(self.model, [batch], optimizer, KLDTWLoss(), "cpu")
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        for layer, previous in zip((self.model.coordconv_in, self.model.coordconv_out), before):
            self.assertFalse(torch.equal(layer.conv.weight, previous))
        # MIT's manifest loader supplies an extra coordinate tensor; loss evaluation accepts it.
        test_loss = eval_epoch(self.model, [(*batch, torch.zeros(2, 3, 2))], KLDTWLoss(), "cpu")
        self.assertTrue(torch.isfinite(torch.tensor(test_loss)))

    def test_diem_window_uses_shared_model_and_carries_state(self):
        from DIEM.fine_tune import run_window, VideoStateStore
        batch = dict(frames=torch.rand(1, 2, 3, 32, 48), heatmaps=torch.ones(1, 2, 4, 6) / 24,
                     valid=torch.ones(1, 2, dtype=torch.bool), delta_t=torch.full((1, 2), 1 / 30),
                     is_video_start=torch.tensor([True]), video_name=["clip"])
        state = VideoStateStore()
        predictions, targets, valid = run_window(self.model.eval(), batch, state, "cpu")
        self.assertEqual(predictions.shape, targets.shape)
        self.assertEqual(valid.shape, (1, 2))
        self.assertEqual(state._store["clip"]["prev_hmap"].shape, (1, 4, 6))
        batch["is_video_start"] = torch.tensor([False])
        continued, _, _ = run_window(self.model, batch, state, "cpu")
        self.assertTrue(torch.isfinite(continued).all())

    def test_checkpoint_default_is_canonical_and_not_working_directory_dependent(self):
        self.assertTrue(OSIE_CHECKPOINT.is_absolute())
        self.assertEqual(OSIE_CHECKPOINT, PROJECT_ROOT / "checkpoints/OSIE_model.pt")
        import inspect
        from OSIE_Modeldata import train, evaluate_test
        self.assertEqual(inspect.signature(train).parameters["checkpoint"].default, OSIE_CHECKPOINT)
        self.assertEqual(inspect.signature(evaluate_test).parameters["checkpoint"].default, OSIE_CHECKPOINT)

    def test_converted_mit_runner_supports_original_mat_layout(self):
        import numpy as np
        import scipy.io as sio
        from PIL import Image
        from testMITlowres import build_dataloaders
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "eye").mkdir()
            Image.new("RGB", (48, 32)).save(root / "images/example.png")
            subject = dict(fix_x=np.array([5., 10., 15., 20.]), fix_y=np.array([5., 10., 15., 20.]),
                           fix_duration=np.array([1., 1., 1., 1.]))
            sio.savemat(root / "eye/fixations.mat", {"fixations": [{"img": "example.png", "subjects": [subject, subject]}]})
            loader = build_dataloaders(root, img_size=(32, 48), batch_size=2, num_workers=0)
            batch = next(iter(loader))
            self.assertEqual(len(batch), 4)
            self.assertEqual(batch[1].shape, (2, 8, 4, 6))
            from OSIE_Modeldata import eval_epoch
            loss = eval_epoch(self.model, loader, KLDTWLoss(), "cpu")
            self.assertTrue(torch.isfinite(torch.tensor(loss)))


if __name__ == "__main__":
    unittest.main()
