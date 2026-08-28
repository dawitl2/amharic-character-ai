import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference import Prediction, RankedPrediction  # noqa: E402
from inference import InferenceEngine  # noqa: E402
from ocr_engine import OCREngine  # noqa: E402


FONT_PATH = Path(r"C:\Windows\Fonts\AbyssinicaSIL-Regular.ttf")


def render(text: str) -> Image.Image:
    font = ImageFont.truetype(str(FONT_PATH), 58)
    probe = Image.new("L", (1, 1), 255)
    bbox = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    image = Image.new("L", (bbox[2] - bbox[0] + 60, bbox[3] - bbox[1] + 60), 255)
    ImageDraw.Draw(image).text((30 - bbox[0], 30 - bbox[1]), text, font=font, fill=0)
    return image


def prediction(character: str, confidence: float) -> Prediction:
    ranked = RankedPrediction(character, confidence, 0)
    return Prediction(
        predicted_character=character,
        confidence=confidence,
        probabilities=(confidence,),
        logits=(1.0,),
        top_predictions=(ranked,),
        probability_sum=1.0,
        tensor_shape=(1, 1, 64, 64),
    )


class SequencedFakeEngine:
    def __init__(self, values):
        self.values = list(values)

    def predict_images(self, images, *, prepare_external, top_k):
        assert prepare_external is True
        return tuple(
            prediction(character, confidence)
            for character, confidence in self.values[: len(images)]
        )


class OCREngineTests(unittest.TestCase):
    def test_word_reconstruction_and_uncertain_marker_keep_raw_prediction(self):
        engine = OCREngine(
            SequencedFakeEngine((("ሰ", 0.99), ("ላ", 0.30), ("ም", 0.98))),
            confidence_threshold=0.50,
        )
        result = engine.recognize_word(render("ሰላም"))
        self.assertEqual(result.text, "ሰ?ም")
        self.assertEqual(result.raw_text, "ሰላም")
        self.assertEqual(result.uncertain_count, 1)
        self.assertEqual(len(result.characters), 3)
        self.assertEqual(len(result.characters[0].prediction.top_predictions), 1)

    def test_sentence_reconstruction_preserves_word_spaces(self):
        values = (("ሰ", 0.9), ("ላ", 0.9), ("ም", 0.9), ("ሀ", 0.9), ("ገ", 0.9), ("ር", 0.9))
        engine = OCREngine(SequencedFakeEngine(values), confidence_threshold=0.5)
        result = engine.recognize_sentence(render("ሰላም   ሀገር"))
        self.assertEqual(result.text, "ሰላም ሀገር")
        self.assertEqual(len(result.words), 2)
        self.assertEqual([len(word.characters) for word in result.words], [3, 3])

    def test_active_cnn_reconstructs_clean_printed_word(self):
        checkpoint = Path(__file__).resolve().parents[1] / "models" / "best_cnn_model.pth"
        if not FONT_PATH.is_file() or not checkpoint.is_file():
            self.skipTest("Active checkpoint or Ethiopic test font is unavailable.")
        result = OCREngine(
            InferenceEngine.from_artifacts(), confidence_threshold=0.50
        ).recognize_word(render("ሰላም"))
        self.assertEqual(len(result.characters), 3)
        self.assertEqual(result.raw_text, "ሰላም")
        self.assertEqual(result.text, "ሰላም")


if __name__ == "__main__":
    unittest.main()
