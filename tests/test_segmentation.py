import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation import crop_region, segmentation_overlay, segment_text  # noqa: E402


FONT_PATHS = (
    Path(r"C:\Windows\Fonts\AbyssinicaSIL-Regular.ttf"),
    Path(r"C:\Windows\Fonts\nyala.ttf"),
    Path(r"C:\Windows\Fonts\ebrima.ttf"),
)


def render_text(
    text: str,
    *,
    font_path: Path = FONT_PATHS[0],
    font_size: int = 54,
    padding: int = 28,
    shift: tuple[int, int] = (0, 0),
) -> Image.Image:
    font = ImageFont.truetype(str(font_path), font_size)
    probe = Image.new("L", (1, 1), 255)
    bbox = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0] + padding * 2 + abs(shift[0])
    height = bbox[3] - bbox[1] + padding * 2 + abs(shift[1])
    image = Image.new("L", (width, height), 255)
    position = (
        padding - bbox[0] + max(0, shift[0]),
        padding - bbox[1] + max(0, shift[1]),
    )
    ImageDraw.Draw(image).text(position, text, font=font, fill=0)
    return image


class SegmentationTests(unittest.TestCase):
    def test_clean_large_small_shifted_and_multiple_fonts_preserve_order(self):
        cases = (
            (FONT_PATHS[0], 54, 28, (0, 0)),
            (FONT_PATHS[0], 92, 45, (0, 0)),
            (FONT_PATHS[0], 28, 18, (0, 0)),
            (FONT_PATHS[1], 58, 60, (12, 7)),
            (FONT_PATHS[2], 58, 36, (0, 0)),
        )
        for font_path, size, padding, shift in cases:
            with self.subTest(font=font_path.name, size=size, shift=shift):
                result = segment_text(
                    render_text(
                        "ሰላም",
                        font_path=font_path,
                        font_size=size,
                        padding=padding,
                        shift=shift,
                    ),
                    mode="word",
                )
                self.assertEqual(len(result.words), 1)
                self.assertEqual(len(result.words[0].characters), 3)
                x_positions = [region.box.x for region in result.words[0].characters]
                self.assertEqual(x_positions, sorted(x_positions))

    def test_sentence_detects_word_gap_and_character_counts(self):
        image = render_text("ሰላም   ሀገር", font_size=62, padding=35)
        result = segment_text(image, mode="sentence")
        self.assertEqual(len(result.words), 2)
        self.assertEqual([len(word.characters) for word in result.words], [3, 3])
        self.assertLess(result.words[0].box.x, result.words[1].box.x)

    def test_debug_overlay_and_crops_match_detected_regions(self):
        image = render_text("ሰላም", font_size=58)
        result = segment_text(image, mode="word")
        overlay = segmentation_overlay(image, result)
        self.assertEqual(overlay.size, image.size)
        crops = [crop_region(image, region.box) for region in result.characters]
        self.assertEqual(len(crops), 3)
        self.assertTrue(all(crop.width > 0 and crop.height > 0 for crop in crops))


if __name__ == "__main__":
    unittest.main()
