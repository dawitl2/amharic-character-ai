import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from torchvision.datasets import ImageFolder


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preflight_training import verify_images, verify_model_output  # noqa: E402


class PreflightTrainingTests(unittest.TestCase):
    def test_verifies_decodable_nonblank_images_and_model_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for class_name, value in (("a", 20), ("b", 200)):
                class_dir = root / class_name
                class_dir.mkdir()
                image = Image.new("L", (16, 16), 255)
                image.putpixel((8, 8), value)
                image.save(class_dir / "sample.png")
            dataset = ImageFolder(root)
            self.assertEqual(
                verify_images(dataset, images_per_class=None, workers=2), 2
            )
            self.assertGreater(verify_model_output(2), 0)

    def test_rejects_blank_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class_dir = root / "a"
            class_dir.mkdir()
            Image.new("L", (16, 16), 255).save(class_dir / "blank.png")
            dataset = ImageFolder(root)
            with self.assertRaisesRegex(ValueError, "no visible intensity variation"):
                verify_images(dataset, images_per_class=None, workers=1)


if __name__ == "__main__":
    unittest.main()
