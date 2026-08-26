import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoints import CheckpointCompatibilityError, load_cnn_bundle  # noqa: E402
from cnn_model import ARCHITECTURE_NAME, CharacterCNN  # noqa: E402
from preprocessing import PreprocessingSpec  # noqa: E402


def metadata(class_to_idx=None):
    mapping = class_to_idx or {"a": 0, "b": 1, "c": 2}
    return {
        "checkpoint_format_version": 1,
        "architecture": ARCHITECTURE_NAME,
        "class_to_idx": mapping,
        "num_classes": len(mapping),
        "preprocessing": PreprocessingSpec().to_metadata(),
    }


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "config.json"
        self.checkpoint_path = self.root / "model.pth"

    def write_artifacts(self, config, checkpoint_metadata, state_dict):
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        torch.save(
            {"model_state_dict": state_dict, "metadata": checkpoint_metadata},
            self.checkpoint_path,
        )

    def test_loads_matching_character_cnn(self):
        expected_metadata = metadata()
        model = CharacterCNN(num_classes=3)
        self.write_artifacts(expected_metadata, expected_metadata, model.state_dict())
        bundle = load_cnn_bundle(self.config_path, self.checkpoint_path)
        self.assertFalse(bundle.model.training)
        self.assertEqual(bundle.idx_to_class, {0: "a", 1: "b", 2: "c"})

    def test_rejects_legacy_linear_tensors(self):
        expected_metadata = metadata()
        self.write_artifacts(
            expected_metadata,
            expected_metadata,
            {"classifier.weight": torch.zeros(3, 4096), "classifier.bias": torch.zeros(3)},
        )
        with self.assertRaisesRegex(CheckpointCompatibilityError, "legacy linear"):
            load_cnn_bundle(self.config_path, self.checkpoint_path)

    def test_rejects_class_mapping_disagreement(self):
        config = metadata()
        checkpoint_metadata = metadata({"a": 0, "c": 1, "b": 2})
        model = CharacterCNN(num_classes=3)
        self.write_artifacts(config, checkpoint_metadata, model.state_dict())
        with self.assertRaisesRegex(CheckpointCompatibilityError, "class mappings differ"):
            load_cnn_bundle(self.config_path, self.checkpoint_path)


if __name__ == "__main__":
    unittest.main()
