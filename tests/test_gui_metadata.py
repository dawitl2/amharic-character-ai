import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui import format_top_predictions, model_information  # noqa: E402


class GuiMetadataTests(unittest.TestCase):
    def test_model_information_reads_saved_training_schema(self):
        info = model_information(
            {
                "cumulative_epochs_trained": 17,
                "dataset": {"train": 70, "validation": 15, "test": 15},
                "scheduler": {"name": "ReduceLROnPlateau"},
            }
        )
        self.assertEqual(info["cumulative_epochs"], 17)
        self.assertEqual(info["train_samples"], 70)
        self.assertEqual(info["validation_samples"], 15)
        self.assertEqual(info["test_samples"], 15)
        self.assertEqual(info["scheduler"], "ReduceLROnPlateau")

    def test_top_predictions_are_ranked_and_formatted(self):
        prediction = SimpleNamespace(
            top_predictions=(
                SimpleNamespace(character="ሀ", probability=0.75),
                SimpleNamespace(character="ሁ", probability=0.20),
            )
        )
        self.assertEqual(format_top_predictions(prediction), "1. ሀ — 75.0%   2. ሁ — 20.0%")


if __name__ == "__main__":
    unittest.main()
