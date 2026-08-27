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
        paths = [p for p in app.split_paths.get("test", []) if p.exists()]
        if not paths:
            raise RuntimeError(
                "The active test split has no current images; train with --fresh first."
            )
        image_path = paths[0]
        direct = app.engine.predict_path(image_path)
        app._select_image(image_path)
        wait_until(app, lambda: app.current_prediction is not None)
        through_gui = app.current_prediction
        assert through_gui is not None
        assert direct.predicted_character == through_gui.predicted_character
        assert direct.probabilities == through_gui.probabilities
        assert direct.logits == through_gui.logits
        assert app.correct_answer_label.cget("text") != "Correct answer: Unknown"
        app.test_count_var.set("10")
        app.split_var.set("test")
        app._run_automatic_test()
        wait_until(app, lambda: "Accuracy:" in app.test_output.get("1.0", "end"), timeout=30.0)
        print("GUI startup: PASS")
        print(f"Direct/GUI prediction: {direct.predicted_character}")
        print(f"Direct/GUI confidence: {direct.confidence:.8f}")
        print("Direct/GUI logits and probabilities: IDENTICAL")
        print("GUI manifest-partition automatic test: PASS")
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
