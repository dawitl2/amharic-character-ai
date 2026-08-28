"""OpenCV segmentation for printed Ethiopic words and text lines.

OpenCV answers *where* character candidates are.  It never assigns character
labels; recognition remains the responsibility of the trained CNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def expanded(self, padding: int, image_size: tuple[int, int]) -> "BoundingBox":
        image_width, image_height = image_size
        left = max(0, self.x - padding)
        top = max(0, self.y - padding)
        right = min(image_width, self.right + padding)
        bottom = min(image_height, self.bottom + padding)
        return BoundingBox(left, top, right - left, bottom - top)

    def union(self, other: "BoundingBox") -> "BoundingBox":
        left = min(self.x, other.x)
        top = min(self.y, other.y)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)
        return BoundingBox(left, top, right - left, bottom - top)


@dataclass(frozen=True)
class CharacterRegion:
    box: BoundingBox


@dataclass(frozen=True)
class WordRegion:
    box: BoundingBox
    characters: tuple[CharacterRegion, ...]
    line_index: int


@dataclass(frozen=True)
class SegmentationResult:
    words: tuple[WordRegion, ...]
    binary_preview: Image.Image

    @property
    def characters(self) -> tuple[CharacterRegion, ...]:
        return tuple(character for word in self.words for character in word.characters)


@dataclass(frozen=True)
class SegmentationSettings:
    minimum_component_area_fraction: float = 0.00002
    minimum_component_height_fraction: float = 0.06
    component_overlap_ratio: float = 0.28
    wide_component_ratio: float = 1.85
    word_gap_width_ratio: float = 0.34
    word_gap_cluster_ratio: float = 2.4
    crop_padding_fraction: float = 0.05


def _to_grayscale(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.uint8)


def _foreground_mask(grayscale: np.ndarray) -> np.ndarray:
    """Return white foreground on black background with polarity detection."""
    height, width = grayscale.shape
    border = np.concatenate(
        (grayscale[0], grayscale[-1], grayscale[:, 0], grayscale[:, -1])
    )
    dark_background = float(np.median(border)) < 127.0
    blur = cv2.GaussianBlur(grayscale, (3, 3), 0)
    threshold_mode = cv2.THRESH_BINARY if dark_background else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(blur, 0, 255, threshold_mode | cv2.THRESH_OTSU)

    # Uneven illumination benefits from an adaptive mask.  Use it only when
    # the border itself varies substantially, so clean synthetic text remains
    # governed by the more stable global Otsu threshold.
    if float(np.std(border)) > 28.0 and min(height, width) >= 31:
        block_size = max(15, (min(height, width) // 8) | 1)
        adaptive_mode = (
            cv2.THRESH_BINARY if dark_background else cv2.THRESH_BINARY_INV
        )
        adaptive = cv2.adaptiveThreshold(
            grayscale,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            adaptive_mode,
            block_size,
            9,
        )
        mask = cv2.bitwise_and(mask, adaptive)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _component_boxes(
    mask: np.ndarray, settings: SegmentationSettings
) -> list[BoundingBox]:
    height, width = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    minimum_area = max(4, round(width * height * settings.minimum_component_area_fraction))
    minimum_height = max(2, round(height * settings.minimum_component_height_fraction))
    boxes = []
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        if area < minimum_area or component_height < minimum_height:
            continue
        boxes.append(
            BoundingBox(
                int(x), int(y), int(component_width), int(component_height)
            )
        )
    return boxes


def _horizontal_overlap(first: BoundingBox, second: BoundingBox) -> float:
    overlap = max(0, min(first.right, second.right) - max(first.x, second.x))
    return overlap / max(1, min(first.width, second.width))


def _vertical_gap(first: BoundingBox, second: BoundingBox) -> int:
    if first.bottom < second.y:
        return second.y - first.bottom
    if second.bottom < first.y:
        return first.y - second.bottom
    return 0


def _group_disconnected_pieces(
    boxes: list[BoundingBox], settings: SegmentationSettings
) -> list[BoundingBox]:
    """Join vertically separated pieces while keeping side-by-side glyphs apart."""
    if not boxes:
        return []
    reference_height = float(np.percentile([box.height for box in boxes], 70))
    ordered = sorted(boxes, key=lambda box: (box.x, box.y))
    groups: list[BoundingBox] = []
    for box in ordered:
        best_index = None
        best_score = -1.0
        for index, group in enumerate(groups):
            overlap = _horizontal_overlap(group, box)
            vertical_gap = _vertical_gap(group, box)
            if (
                overlap >= settings.component_overlap_ratio
                and vertical_gap <= reference_height * 0.42
            ):
                score = overlap - vertical_gap / max(reference_height, 1.0)
                if score > best_score:
                    best_index = index
                    best_score = score
        if best_index is None:
            groups.append(box)
        else:
            groups[best_index] = groups[best_index].union(box)
    return sorted(groups, key=lambda box: box.x)


def _split_wide_boxes(
    boxes: list[BoundingBox], mask: np.ndarray, settings: SegmentationSettings
) -> list[BoundingBox]:
    if not boxes:
        return []
    typical_width = float(median(box.width for box in boxes))
    typical_height = float(median(box.height for box in boxes))
    if len(boxes) == 1:
        typical_width = min(typical_width, typical_height * 0.82)

    result: list[BoundingBox] = []
    for box in boxes:
        target_width = max(4.0, typical_width)
        estimated_parts = max(1, round(box.width / target_width))
        if (
            estimated_parts < 2
            or box.width < settings.wide_component_ratio * target_width
        ):
            result.append(box)
            continue

        projection = np.count_nonzero(
            mask[box.y : box.bottom, box.x : box.right], axis=0
        )
        cuts = []
        last_cut = 0
        for part in range(1, estimated_parts):
            expected = round(part * box.width / estimated_parts)
            radius = max(2, round(target_width * 0.28))
            low = max(last_cut + 2, expected - radius)
            high = min(box.width - 2, expected + radius)
            if low >= high:
                continue
            cut = low + int(np.argmin(projection[low : high + 1]))
            cuts.append(cut)
            last_cut = cut
        edges = [0, *cuts, box.width]
        parts = [
            BoundingBox(box.x + left, box.y, right - left, box.height)
            for left, right in zip(edges, edges[1:])
            if right - left >= 3
        ]
        result.extend(parts if len(parts) >= 2 else [box])
    return sorted(result, key=lambda candidate: candidate.x)


def _vertical_overlap(first: BoundingBox, second: BoundingBox) -> float:
    overlap = max(0, min(first.bottom, second.bottom) - max(first.y, second.y))
    return overlap / max(1, min(first.height, second.height))


def _group_lines(boxes: list[BoundingBox]) -> list[list[BoundingBox]]:
    lines: list[tuple[BoundingBox, list[BoundingBox]]] = []
    for box in sorted(boxes, key=lambda candidate: (candidate.y, candidate.x)):
        best_index = None
        best_overlap = 0.0
        for index, (line_box, _) in enumerate(lines):
            overlap = _vertical_overlap(line_box, box)
            if overlap >= 0.32 and overlap > best_overlap:
                best_index = index
                best_overlap = overlap
        if best_index is None:
            lines.append((box, [box]))
        else:
            line_box, members = lines[best_index]
            members.append(box)
            lines[best_index] = (line_box.union(box), members)
    lines.sort(key=lambda item: item[0].y)
    return [sorted(members, key=lambda box: box.x) for _, members in lines]


def _word_gap_threshold(
    boxes: list[BoundingBox], settings: SegmentationSettings
) -> float:
    typical_width = float(median(box.width for box in boxes))
    gaps = [max(0, right.x - left.right) for left, right in zip(boxes, boxes[1:])]
    positive = sorted(gap for gap in gaps if gap > 0)
    if not positive:
        return typical_width * settings.word_gap_width_ratio
    lower_half = positive[: max(1, (len(positive) + 1) // 2)]
    typical_inner_gap = float(median(lower_half))
    return max(
        typical_width * settings.word_gap_width_ratio,
        typical_inner_gap * settings.word_gap_cluster_ratio + 1.0,
    )


def _boxes_to_words(
    lines: list[list[BoundingBox]], settings: SegmentationSettings
) -> tuple[WordRegion, ...]:
    words = []
    for line_index, boxes in enumerate(lines):
        if not boxes:
            continue
        gap_threshold = _word_gap_threshold(boxes, settings)
        current = [boxes[0]]
        grouped = []
        for previous, box in zip(boxes, boxes[1:]):
            if box.x - previous.right > gap_threshold:
                grouped.append(current)
                current = [box]
            else:
                current.append(box)
        grouped.append(current)
        for group in grouped:
            word_box = group[0]
            for box in group[1:]:
                word_box = word_box.union(box)
            words.append(
                WordRegion(
                    box=word_box,
                    characters=tuple(CharacterRegion(box) for box in group),
                    line_index=line_index,
                )
            )
    return tuple(words)


def segment_text(
    image: Image.Image,
    *,
    mode: str,
    settings: SegmentationSettings = SegmentationSettings(),
) -> SegmentationResult:
    """Segment a printed word or line; output is always in reading order."""
    if mode not in {"word", "sentence"}:
        raise ValueError("Segmentation mode must be 'word' or 'sentence'.")
    grayscale = _to_grayscale(image)
    mask = _foreground_mask(grayscale)
    components = _component_boxes(mask, settings)
    characters = _group_disconnected_pieces(components, settings)
    characters = _split_wide_boxes(characters, mask, settings)
    lines = _group_lines(characters)
    if mode == "word":
        ordered = [box for line in lines for box in line]
        lines = [ordered] if ordered else []
        words = _boxes_to_words(lines, settings)
        if len(words) > 1:
            # A word upload intentionally has no word-boundary semantics.
            all_boxes = [character.box for word in words for character in word.characters]
            words = _boxes_to_words([all_boxes], settings)
            if len(words) > 1:
                union = words[0].box
                all_characters = []
                for word in words:
                    union = union.union(word.box)
                    all_characters.extend(word.characters)
                words = (WordRegion(union, tuple(all_characters), 0),)
    else:
        words = _boxes_to_words(lines, settings)
    preview = Image.fromarray(mask, mode="L")
    return SegmentationResult(words=words, binary_preview=preview)


def crop_region(
    image: Image.Image,
    box: BoundingBox,
    *,
    padding_fraction: float = SegmentationSettings().crop_padding_fraction,
) -> Image.Image:
    padding = max(1, round(max(box.width, box.height) * padding_fraction))
    expanded = box.expanded(padding, image.size)
    return image.crop((expanded.x, expanded.y, expanded.right, expanded.bottom))


def segmentation_overlay(
    image: Image.Image, result: SegmentationResult
) -> Image.Image:
    """Blue word boxes and green character boxes make failures inspectable."""
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    line_width = max(2, round(max(image.size) / 500))
    for word_index, word in enumerate(result.words, start=1):
        draw.rectangle(
            (word.box.x, word.box.y, word.box.right, word.box.bottom),
            outline="#246BFD",
            width=line_width,
        )
        draw.text((word.box.x, max(0, word.box.y - 14)), f"W{word_index}", fill="#246BFD")
        for character_index, character in enumerate(word.characters, start=1):
            box = character.box
            draw.rectangle(
                (box.x, box.y, box.right, box.bottom),
                outline="#16A34A",
                width=line_width,
            )
            draw.text((box.x, box.y), str(character_index), fill="#16A34A")
    return overlay
