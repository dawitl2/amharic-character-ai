import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from torchvision.datasets import ImageFolder


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_splits import create_split_manifest, load_or_create_split_manifest  # noqa: E402


class DataSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        for class_name, base_color in (("a", 20), ("b", 220)):
            class_dir = self.root / class_name
            class_dir.mkdir()
            for index in range(12):
                Image.new("L", (16, 16), base_color + index).save(
                    class_dir / f"independent_{index:02d}.png"
                )
            duplicate = Image.new("L", (16, 16), base_color + 40)
            duplicate.save(class_dir / "duplicate_one.png")
            duplicate.save(class_dir / "duplicate_two.png")
            variant = Image.new("L", (16, 16), base_color + 60)
            variant.save(class_dir / "writer7_aug_1.png")
            variant.save(class_dir / "writer7_aug_2.png")
        self.dataset = ImageFolder(self.root)

    @staticmethod
    def split_for_path(manifest, path):
        return next(name for name, paths in manifest["splits"].items() if path in paths)

    def test_duplicate_and_variant_families_do_not_cross_splits(self):
        manifest = create_split_manifest(self.dataset, self.root)
        for class_name in ("a", "b"):
            duplicate_splits = {
                self.split_for_path(manifest, f"{class_name}/duplicate_{suffix}.png")
                for suffix in ("one", "two")
            }
            variant_splits = {
                self.split_for_path(manifest, f"{class_name}/writer7_aug_{number}.png")
                for number in (1, 2)
            }
            self.assertEqual(len(duplicate_splits), 1)
            self.assertEqual(len(variant_splits), 1)

    def test_manifest_is_deterministic_and_complete(self):
        first = create_split_manifest(self.dataset, self.root)
        second = create_split_manifest(self.dataset, self.root)
        self.assertEqual(first, second)
        all_paths = [path for paths in first["splits"].values() for path in paths]
        self.assertEqual(len(all_paths), len(self.dataset))
        self.assertEqual(len(set(all_paths)), len(self.dataset))

    def test_replaced_image_content_invalidates_existing_manifest(self):
        manifest_path = self.root / "split.json"
        first = load_or_create_split_manifest(self.dataset, self.root, manifest_path)
        changed_path = self.root / "a" / "independent_00.png"
        Image.new("L", (16, 16), 199).save(changed_path)
        refreshed_dataset = ImageFolder(self.root)
        second = load_or_create_split_manifest(
            refreshed_dataset, self.root, manifest_path
        )
        self.assertNotEqual(first["dataset_signature"], second["dataset_signature"])
        self.assertNotEqual(first["inventory_signature"], second["inventory_signature"])


if __name__ == "__main__":
    unittest.main()
