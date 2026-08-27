import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_contract import validate_training_dataset  # noqa: E402


class DatasetContractTests(unittest.TestCase):
    def _manifest(self, root: Path, characters: list[str]) -> Path:
        path = root / "characters.json"
        path.write_text(json.dumps(characters, ensure_ascii=False), encoding="utf-8")
        return path

    def test_accepts_exact_balanced_character_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = type("Dataset", (), {"__len__": lambda self: 6})()
            dataset.classes = ["ሀ", "ሁ"]
            dataset.samples = [(f"image-{index}", index // 3) for index in range(6)]
            report = validate_training_dataset(
                dataset, character_manifest_path=self._manifest(root, dataset.classes)
            )
            self.assertEqual(report.class_count, 2)
            self.assertEqual(report.image_count, 6)
            self.assertEqual(report.minimum_images_per_class, 3)

    def test_rejects_dataset_class_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = type("Dataset", (), {"__len__": lambda self: 3})()
            dataset.classes = ["ሀ"]
            dataset.samples = [(f"image-{index}", 0) for index in range(3)]
            with self.assertRaisesRegex(ValueError, "do not exactly match"):
                validate_training_dataset(
                    dataset,
                    character_manifest_path=self._manifest(root, ["ሀ", "ሁ"]),
                )

    def test_rejects_duplicate_manifest_characters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = type("Dataset", (), {"__len__": lambda self: 3})()
            dataset.classes = ["ሀ"]
            dataset.samples = [(f"image-{index}", 0) for index in range(3)]
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_training_dataset(
                    dataset,
                    character_manifest_path=self._manifest(root, ["ሀ", "ሀ"]),
                )


if __name__ == "__main__":
    unittest.main()
