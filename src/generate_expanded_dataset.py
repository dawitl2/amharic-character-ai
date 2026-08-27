"""Generate a balanced, reproducible synthetic Ethiopic dataset."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARACTERS_PATH = Path(__file__).with_name("characters.json")
DATA_DIR = PROJECT_ROOT / "data"
FONT_PATHS = (
    Path(r"C:\Windows\Fonts\AbyssinicaSIL-Regular.ttf"),
    Path(r"C:\Windows\Fonts\nyala.ttf"),
    Path(r"C:\Windows\Fonts\ebrima.ttf"),
    Path(r"C:\Windows\Fonts\ebrimabd.ttf"),
)
TARGET_IMAGES_PER_CLASS = 1200
GENERATION_SEED = 42


def load_characters(path: Path = CHARACTERS_PATH) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_fonts(characters: list[str], font_paths: tuple[Path, ...]) -> None:
    """Reject missing fonts, blank glyphs, and identical fallback/tofu renderings."""
    for font_path in font_paths:
        if not font_path.is_file():
            raise FileNotFoundError(f"Required Ethiopic font not found: {font_path}")
        font = ImageFont.truetype(str(font_path), 64)
        signatures: dict[str, str] = {}
        for character in characters:
            image = Image.new("L", (96, 96), 255)
            draw = ImageDraw.Draw(image)
            draw.text((8, 8), character, font=font, fill=0)
            if image.getextrema() == (255, 255):
                raise ValueError(f"{font_path.name} rendered U+{ord(character):04X} blank.")
            signature = hashlib.sha256(image.tobytes()).hexdigest()
            previous = signatures.get(signature)
            if previous is not None:
                raise ValueError(
                    f"{font_path.name} rendered {previous} and {character} identically; "
                    "this may be a missing-glyph fallback."
                )
            signatures[signature] = character


def add_scanning_artifacts(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    rng: random.Random,
) -> None:
    if rng.random() < 0.2:
        for _ in range(rng.randint(1, 3)):
            y = rng.randint(0, height - 1)
            draw.line([(0, y), (width, y)], fill=rng.randint(180, 230), width=1)
    if rng.random() < 0.2:
        for _ in range(rng.randint(1, 3)):
            x = rng.randint(0, width - 1)
            draw.line([(x, 0), (x, height)], fill=rng.randint(180, 230), width=1)


def generate_dataset() -> None:
    characters = load_characters()
    validate_fonts(characters, FONT_PATHS)
    rng = random.Random(GENERATION_SEED)
    np_rng = np.random.default_rng(GENERATION_SEED)
    stats = defaultdict(int)
    print(
        f"Ensuring at least {TARGET_IMAGES_PER_CLASS} images per class for "
        f"{len(characters)} classes (seed={GENERATION_SEED})..."
    )

    for character in characters:
        output_folder = DATA_DIR / character
        output_folder.mkdir(parents=True, exist_ok=True)
        existing = sorted(output_folder.glob("*.png"))
        current_count = len(existing)
        if current_count > TARGET_IMAGES_PER_CLASS:
            raise RuntimeError(
                f"{output_folder} has {current_count} images, above the target of "
                f"{TARGET_IMAGES_PER_CLASS}. No files were deleted."
            )

        needed = TARGET_IMAGES_PER_CLASS - current_count
        for index in range(current_count + 1, current_count + needed + 1):
            image_size = rng.choice((64, 128))
            font_path = rng.choice(FONT_PATHS)
            font_size = rng.randint(24, 56) if image_size == 64 else rng.randint(48, 100)
            shift_range = 8 if image_size == 64 else 14
            background = rng.randint(220, 255)
            image = Image.new("L", (image_size, image_size), color=background)
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(str(font_path), font_size)
            bbox = draw.textbbox((0, 0), character, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (image_size - text_width) // 2 - bbox[0] + rng.randint(-shift_range, shift_range)
            y = (image_size - text_height) // 2 - bbox[1] + rng.randint(-shift_range, shift_range)
            draw.text((x, y), character, font=font, fill=rng.randint(0, 80))
            image = image.rotate(rng.uniform(-15.0, 15.0), fillcolor=background)
            add_scanning_artifacts(ImageDraw.Draw(image), image_size, image_size, rng)
            if rng.random() < 0.5:
                image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.5, 1.5))
            if rng.random() < 0.5:
                image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.5, 1.5))
            if rng.random() < 0.3:
                image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))
            if rng.random() < 0.2:
                pixels = np.asarray(image, dtype=np.float32)
                noise = np_rng.normal(0, rng.uniform(5, 15), pixels.shape)
                image = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))

            font_tag = font_path.stem.lower().replace("-regular", "")
            image.save(output_folder / f"synthetic_exp_{index:04d}_{font_tag}.png")

        stats[character] = current_count + needed
        print(f"Verified {stats[character]} images for class {character}")

    print(f"Dataset complete: {len(characters)} classes, {sum(stats.values())} images")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    generate_dataset()
