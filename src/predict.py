"""Predict one image with the active CharacterCNN checkpoint."""

import argparse
import sys
from pathlib import Path

from inference import InferenceEngine


def predict_character(image_path: Path) -> None:
    engine = InferenceEngine.from_artifacts()
    print(engine.startup_summary())
    prediction = engine.predict_path(image_path, top_k=3)
    print(f"Input tensor: {prediction.tensor_shape}")
    print(f"Probability sum: {prediction.probability_sum:.6f}")
    print(f"Prediction: {prediction.predicted_character}")
    print(f"Confidence: {prediction.confidence:.1%}")
    print("Top predictions:")
    for rank, candidate in enumerate(prediction.top_predictions, start=1):
        print(f"  {rank}. {candidate.character} — {candidate.probability:.1%}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path, help="Image to classify")
    arguments = parser.parse_args()
    predict_character(arguments.image_path)
