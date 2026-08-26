"""Smoke-test GUI startup and direct/GUI inference consistency."""

import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gui import AmharicAIApp  # noqa: E402


def wait_until(app, predicate, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.update()
        if predicate():
            return
        time.sleep(0.02)
    raise TimeoutError("GUI operation did not finish in time.")


def main() -> None:
    app = AmharicAIApp()
    app.withdraw()
    try:
        dataset_index = app.split_indices["test"][0]
        image_path = Path(app.dataset.samples[dataset_index][0])
        direct = app.engine.predict_path(image_path)
        app._select_image(image_path)
        wait_until(app, lambda: app.current_prediction is not None)
        through_gui = app.current_prediction
        assert through_gui is not None
        assert direct.predicted_character == through_gui.predicted_character
        assert direct.probabilities == through_gui.probabilities
        assert direct.logits == through_gui.logits
        assert app.correct_answer_label.cget("text") != "Correct answer: Unknown"
        print("GUI startup: PASS")
        print(f"Direct/GUI prediction: {direct.predicted_character}")
        print(f"Direct/GUI confidence: {direct.confidence:.8f}")
        print("Direct/GUI logits and probabilities: IDENTICAL")
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
