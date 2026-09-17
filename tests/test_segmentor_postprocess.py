"""Tests for the post-processing switches of ``S2C2Segmentation``.

The segmentor is instantiated without loading any model weights: only the
attributes read by ``_setup_inference_options`` / ``postprocess_result`` are set.
"""

import unittest

import torch

try:
    from models.segmentor import S2C2Segmentation
except Exception:  # pragma: no cover - mmseg not installed
    S2C2Segmentation = None


def _stub(x_options=None, logit_scale=40, prob_thd=0.2):
    obj = S2C2Segmentation.__new__(S2C2Segmentation)
    obj._apply_x_options(x_options or {})
    obj.logit_scale = logit_scale
    obj.prob_thd = prob_thd
    obj.query_idx = torch.tensor([0, 1, 1, 2])
    obj._setup_inference_options()
    return obj


@unittest.skipIf(S2C2Segmentation is None, "mmseg is not importable")
class TestSpatialRefineSwitches(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(1, 4, 24, 24) * 0.2
        self.image = torch.rand(1, 3, 24, 24)

    def _run(self, obj):
        obj._refine_image = self.image
        return obj.postprocess_result(self.logits.clone(), None)

    def test_dilations_switch_parsing(self):
        obj = _stub({'X_SPATIAL_REFINE': 3, 'X_SPATIAL_REFINE_DILATIONS': '[8, 16]'})
        self.assertEqual(obj._spatial_refine_dilations, (8, 16))
        self.assertIsNone(_stub({'X_SPATIAL_REFINE': 3})._spatial_refine_dilations)
        with self.assertRaises(ValueError):
            _stub({'X_SPATIAL_REFINE_DILATIONS': '0,4'})

    def test_temperature_switch_parsing(self):
        self.assertIsNone(_stub({'X_SPATIAL_REFINE': 3})._spatial_refine_temp)
        self.assertEqual(
            _stub({'X_SPATIAL_REFINE': 3, 'X_SPATIAL_REFINE_TEMP': 100})._spatial_refine_temp,
            100.0)
        with self.assertRaises(ValueError):
            _stub({'X_SPATIAL_REFINE_TEMP': -1})

    def test_temperature_equal_to_logit_scale_is_identity(self):
        base = self._run(_stub({'X_SPATIAL_REFINE': 3}))
        same = self._run(_stub({'X_SPATIAL_REFINE': 3, 'X_SPATIAL_REFINE_TEMP': 40}))
        self.assertTrue(torch.equal(base, same))

    def test_background_threshold_uses_calibrated_probabilities(self):
        # The background decision (label 0) must not move with the refinement temperature.
        base = self._run(_stub({'X_SPATIAL_REFINE': 3}))
        sharp = self._run(_stub({'X_SPATIAL_REFINE': 3, 'X_SPATIAL_REFINE_TEMP': 100}))
        self.assertTrue(torch.equal(base == 0, sharp == 0))

    def test_temperature_ignored_without_refinement(self):
        base = self._run(_stub({}))
        sharp = self._run(_stub({'X_SPATIAL_REFINE_TEMP': 100}))
        self.assertTrue(torch.equal(base, sharp))


if __name__ == '__main__':
    unittest.main()


@unittest.skipIf(S2C2Segmentation is None, "mmseg is not importable")
class TestSharpRefinementShortcut(unittest.TestCase):
    def test_no_background_prediction_equals_sharp_refined_argmax(self):
        from models import spatial_refine
        torch.manual_seed(2)
        logits = torch.randn(1, 4, 24, 24) * 0.2
        image = torch.rand(1, 3, 24, 24)
        obj = _stub({'X_SPATIAL_REFINE': 3, 'X_SPATIAL_REFINE_TEMP': 100}, prob_thd=0.0)
        obj._refine_image = image
        pred = obj.postprocess_result(logits.clone(), None)
        probs = obj._class_probabilities(logits[0], 100.0)
        expected = spatial_refine.refine(image[0], probs, 3).argmax(0, keepdim=True)
        self.assertTrue(torch.equal(pred, expected))
