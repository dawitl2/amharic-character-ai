import sys
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import CharacterTransform, PreprocessingSpec, preprocess_image  # noqa: E402


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.spec = PreprocessingSpec()
        self.image = Image.new("RGB", (160, 80), "white")
        ImageDraw.Draw(self.image).rectangle((65, 15, 94, 64), fill="black")

    def test_dataset_transform_has_expected_shape_and_range(self):
        tensor = CharacterTransform(self.spec)(self.image)
        self.assertEqual(tuple(tensor.shape), (1, 64, 64))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_external_preparation_centers_without_stretching(self):
        tensor = preprocess_image(self.image, self.spec, prepare_external=True)
        foreground = tensor[0] < 0.5
        rows, columns = torch.where(foreground)
        width = int(columns.max() - columns.min() + 1)
        height = int(rows.max() - rows.min() + 1)
        self.assertAlmostEqual(width / height, 30 / 50, delta=0.08)
        self.assertAlmostEqual(float(columns.float().mean()), 31.5, delta=1.0)
        self.assertAlmostEqual(float(rows.float().mean()), 31.5, delta=1.0)

    def test_zero_normalization_scale_is_rejected(self):
        invalid_spec = PreprocessingSpec(normalization_std=(0.0,))
        with self.assertRaisesRegex(ValueError, "cannot be zero"):
            preprocess_image(self.image, invalid_spec)


if __name__ == "__main__":
    unittest.main()
