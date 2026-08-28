"""Word and sentence OCR built around the existing single-character CNN."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from inference import InferenceEngine, Prediction
from segmentation import (
    BoundingBox,
    SegmentationResult,
    crop_region,
    segmentation_overlay,
    segment_text,
)


@dataclass(frozen=True)
class OCRCharacter:
    box: BoundingBox
    prediction: Prediction
    displayed_character: str
    uncertain: bool
    crop: Image.Image


@dataclass(frozen=True)
class OCRWord:
    box: BoundingBox
    line_index: int
    characters: tuple[OCRCharacter, ...]

    @property
    def text(self) -> str:
        return "".join(character.displayed_character for character in self.characters)

    @property
    def raw_text(self) -> str:
        return "".join(
            character.prediction.predicted_character for character in self.characters
        )

    @property
    def mean_confidence(self) -> float:
        if not self.characters:
            return 0.0
        return sum(
            character.prediction.confidence for character in self.characters
        ) / len(self.characters)


@dataclass(frozen=True)
class OCRResult:
    mode: str
    words: tuple[OCRWord, ...]
    segmentation: SegmentationResult
    overlay: Image.Image
    confidence_threshold: float

    def _reconstruct(self, *, raw: bool) -> str:
        if not self.words:
            return ""
        lines: dict[int, list[str]] = {}
        for word in self.words:
            lines.setdefault(word.line_index, []).append(
                word.raw_text if raw else word.text
            )
        return "\n".join(" ".join(lines[index]) for index in sorted(lines))

    @property
    def text(self) -> str:
        return self._reconstruct(raw=False)

    @property
    def raw_text(self) -> str:
        return self._reconstruct(raw=True)

    @property
    def characters(self) -> tuple[OCRCharacter, ...]:
        return tuple(character for word in self.words for character in word.characters)

    @property
    def uncertain_count(self) -> int:
        return sum(character.uncertain for character in self.characters)


class OCREngine:
    """Coordinates segmentation and CNN recognition without mixing concerns."""

    def __init__(
        self,
        character_engine: InferenceEngine,
        *,
        confidence_threshold: float = 0.50,
        uncertain_marker: str = "?",
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        self.character_engine = character_engine
        self.confidence_threshold = confidence_threshold
        self.uncertain_marker = uncertain_marker

    def recognize(self, image: Image.Image, *, mode: str) -> OCRResult:
        segmentation = segment_text(image, mode=mode)
        regions = [
            character
            for word in segmentation.words
            for character in word.characters
        ]
        crops = [crop_region(image, region.box) for region in regions]
        predictions = self.character_engine.predict_images(
            crops, prepare_external=True, top_k=3
        )
        recognized_characters = []
        for region, crop, prediction in zip(regions, crops, predictions):
            uncertain = prediction.confidence < self.confidence_threshold
            recognized_characters.append(
                OCRCharacter(
                    box=region.box,
                    prediction=prediction,
                    displayed_character=(
                        self.uncertain_marker
                        if uncertain
                        else prediction.predicted_character
                    ),
                    uncertain=uncertain,
                    crop=crop,
                )
            )

        words = []
        cursor = 0
        for segmented_word in segmentation.words:
            count = len(segmented_word.characters)
            words.append(
                OCRWord(
                    box=segmented_word.box,
                    line_index=segmented_word.line_index,
                    characters=tuple(recognized_characters[cursor : cursor + count]),
                )
            )
            cursor += count
        return OCRResult(
            mode=mode,
            words=tuple(words),
            segmentation=segmentation,
            overlay=segmentation_overlay(image, segmentation),
            confidence_threshold=self.confidence_threshold,
        )

    def recognize_word(self, image: Image.Image) -> OCRResult:
        return self.recognize(image, mode="word")

    def recognize_sentence(self, image: Image.Image) -> OCRResult:
        return self.recognize(image, mode="sentence")
