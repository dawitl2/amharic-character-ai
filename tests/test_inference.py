import sys
import unittest
from pathlib import Path

import torch
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoints import ModelBundle  # noqa: E402
from cnn_model import CharacterCNN  # noqa: E402
from inference import InferenceEngine  # noqa: E402
from preprocessing import PreprocessingSpec  # noqa: E402


class InferenceTests(unittest.TestCase):
    def test_batch_inference_matches_single_image_inference(self):
        torch.manual_seed(7)
        model = CharacterCNN(num_classes=3)
        model.eval()
        spec = PreprocessingSpec()
        bundle = ModelBundle(
            model=model,
            metadata={"class_to_idx": {"ሀ": 0, "ሁ": 1, "ሂ": 2}},
            preprocessing=spec,
            checkpoint_path=Path("test.pth"),
        )
        engine = InferenceEngine(bundle)
        first = Image.new("L", (80, 60), 255)
        second = Image.new("L", (40, 90), 180)

        direct = engine.predict_image(first, prepare_external=True)
        batch = engine.predict_images(
            [first, second], prepare_external=True
        )

        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].predicted_character, direct.predicted_character)
        self.assertEqual(batch[0].probabilities, direct.probabilities)
        self.assertAlmostEqual(batch[0].probability_sum, 1.0, places=5)
        self.assertAlmostEqual(batch[1].probability_sum, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
