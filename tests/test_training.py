import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoints import save_checkpoint_atomic  # noqa: E402
from cnn_model import ARCHITECTURE_NAME, CharacterCNN  # noqa: E402
from preprocessing import PreprocessingSpec  # noqa: E402
from training import (  # noqa: E402
    TrainingProgress,
    TrainingSettings,
    checkpoint_payload,
    resume_training_state,
    validate_split_coverage,
)
from train import parse_args  # noqa: E402


class TrainingTests(unittest.TestCase):
    def test_default_maximum_is_200_cumulative_epochs(self):
        self.assertEqual(TrainingSettings().max_epochs, 200)

    def test_fresh_flag_is_explicit_and_disabled_by_default(self):
        self.assertFalse(parse_args([]).fresh)
        self.assertTrue(parse_args(["--fresh"]).fresh)

    def test_data_loader_workers_can_be_selected_explicitly(self):
        self.assertEqual(parse_args([]).num_workers, 0)
        self.assertEqual(parse_args(["--num-workers", "2"]).num_workers, 2)

    def test_split_coverage_requires_every_class_in_every_partition(self):
        dataset = SimpleNamespace(
            classes=["a", "b"],
            samples=[("a1", 0), ("b1", 1), ("a2", 0), ("b2", 1)],
        )
        with self.assertRaisesRegex(ValueError, "validation split is missing"):
            validate_split_coverage(
                dataset,
                {"train": [0, 1], "validation": [2], "test": [2, 3]},
            )

    def test_split_coverage_accepts_complete_partitions(self):
        dataset = SimpleNamespace(
            classes=["a", "b"],
            samples=[("a1", 0), ("b1", 1)] * 3,
        )
        validate_split_coverage(
            dataset,
            {"train": [0, 1], "validation": [2, 3], "test": [4, 5]},
        )

    def test_resume_restores_epoch_optimizer_and_scheduler(self):
        settings = TrainingSettings()
        spec = PreprocessingSpec()
        mapping = {"a": 0, "b": 1, "c": 2}
        data = SimpleNamespace(
            dataset=SimpleNamespace(class_to_idx=mapping),
            manifest={"dataset_signature": "same-dataset"},
        )
        model = CharacterCNN(3)
        optimizer = torch.optim.SGD(model.parameters(), lr=settings.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
        optimizer.param_groups[0]["lr"] = 0.005
        progress = TrainingProgress(
            cumulative_epochs=17,
            best_validation_accuracy=92.5,
            best_validation_loss=0.2,
            epoch_of_best_checkpoint=14,
            epochs_without_improvement=3,
        )
        metadata = {
            "architecture": ARCHITECTURE_NAME,
            "class_to_idx": mapping,
            "num_classes": 3,
            "preprocessing": spec.to_metadata(),
            "optimizer": {"name": "SGD", "learning_rate": settings.learning_rate},
            "split": {"dataset_signature": "same-dataset"},
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "latest.pth"
            save_checkpoint_atomic(
                checkpoint_path,
                checkpoint_payload(model, optimizer, scheduler, metadata, progress),
            )
            restored_model = CharacterCNN(3)
            restored_optimizer = torch.optim.SGD(
                restored_model.parameters(), lr=settings.learning_rate
            )
            restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                restored_optimizer, mode="max"
            )
            restored = resume_training_state(
                checkpoint_path,
                restored_model,
                restored_optimizer,
                restored_scheduler,
                data,
                spec,
                settings,
                torch.device("cpu"),
            )
        self.assertEqual(restored.cumulative_epochs, 17)
        self.assertEqual(restored.epoch_of_best_checkpoint, 14)
        self.assertEqual(restored_optimizer.param_groups[0]["lr"], 0.005)
        for expected, actual in zip(model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
