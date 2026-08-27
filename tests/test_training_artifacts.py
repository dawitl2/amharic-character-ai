import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training_artifacts import archive_active_training_artifacts  # noqa: E402


class TrainingArtifactTests(unittest.TestCase):
    def test_fresh_start_archives_existing_artifacts_without_deleting_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "latest.pth"
            existing.write_bytes(b"checkpoint")
            missing = root / "missing.json"

            archive = archive_active_training_artifacts(
                (existing, missing), archive_root=root / "archive"
            )

            self.assertIsNotNone(archive)
            self.assertFalse(existing.exists())
            self.assertEqual((archive / existing.name).read_bytes(), b"checkpoint")
            self.assertFalse((archive / missing.name).exists())

    def test_fresh_start_without_artifacts_does_not_create_an_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = archive_active_training_artifacts(
                (root / "missing.pth",), archive_root=root / "archive"
            )
            self.assertIsNone(archive)
            self.assertFalse((root / "archive").exists())


if __name__ == "__main__":
    unittest.main()
