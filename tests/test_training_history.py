import io
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training_history import build_training_figure, load_training_history  # noqa: E402


class TrainingHistoryTests(unittest.TestCase):
    def test_loads_real_metric_columns_and_builds_three_graphs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text(
                "epoch,train_loss,train_accuracy,validation_loss,validation_accuracy,learning_rate\n"
                "1,5.0,10.0,4.0,12.0,0.01\n"
                "2,3.0,30.0,2.0,35.0,0.005\n",
                encoding="utf-8",
            )
            history = load_training_history(path)
            self.assertEqual(len(history), 2)
            self.assertEqual(history[-1].validation_accuracy, 35.0)
            figure = build_training_figure(history, best_epoch=2)
            self.assertEqual(len(figure.axes), 3)
            output = io.BytesIO()
            figure.savefig(output, format="png")
            self.assertGreater(len(output.getvalue()), 1000)

    def test_missing_history_is_reported_without_inventing_values(self):
        history = load_training_history(Path("definitely-missing.csv"))
        self.assertEqual(history, ())
        figure = build_training_figure(history, best_epoch=None)
        self.assertEqual(len(figure.axes), 3)


if __name__ == "__main__":
    unittest.main()
