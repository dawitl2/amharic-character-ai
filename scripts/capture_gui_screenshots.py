"""Capture honest documentation screenshots from the running desktop app.

This utility renders generated printed Ethiopic samples, runs them through the
real segmentation and active CNN pipeline, and captures the resulting pages.
It never substitutes mock predictions or invented evaluation values.
"""

from __future__ import annotations

import sys
import time
import ctypes
from ctypes import wintypes
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCREENSHOTS = ROOT / "docs" / "screenshots"
SAMPLES = ROOT / "docs" / "samples"
sys.path.insert(0, str(SRC))

from gui import AmharicAIApp  # noqa: E402
from ocr_engine import OCREngine  # noqa: E402


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = (
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    )


def _ethiopic_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("C:/Windows/Fonts/AbyssinicaSIL-Regular.ttf"),
        Path("C:/Windows/Fonts/nyala.ttf"),
        Path("C:/Windows/Fonts/ebrima.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    raise FileNotFoundError("No supported Ethiopic font is installed.")


def _render_sample(text: str, path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    font = _ethiopic_font(round(size[1] * 0.56))
    bounds = draw.textbbox((0, 0), text, font=font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    position = (
        max(20, (size[0] - text_width) // 2 - bounds[0]),
        max(12, (size[1] - text_height) // 2 - bounds[1]),
    )
    draw.text(position, text, font=font, fill="black")
    image.save(path)
    return image


def _pump(app: AmharicAIApp, seconds: float = 0.4) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.update()
        time.sleep(0.02)


def _capture(app: AmharicAIApp, filename: str) -> None:
    _pump(app)
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    window = app.winfo_id()
    parent = user32.GetParent(window)
    if parent:
        window = parent

    rectangle = wintypes.RECT()
    if not user32.GetWindowRect(window, ctypes.byref(rectangle)):
        raise OSError("Could not read the GUI window bounds.")
    width = rectangle.right - rectangle.left
    height = rectangle.bottom - rectangle.top
    source_dc = user32.GetWindowDC(window)
    memory_dc = gdi32.CreateCompatibleDC(source_dc)
    bitmap = gdi32.CreateCompatibleBitmap(source_dc, width, height)
    previous = gdi32.SelectObject(memory_dc, bitmap)
    try:
        # PW_RENDERFULLCONTENT requests the actual window surface even when the
        # desktop is covered or the app is not the foreground window.
        if not user32.PrintWindow(window, memory_dc, 2):
            raise OSError("Windows could not render the GUI window surface.")
        info = _BitmapInfoHeader()
        info.biSize = ctypes.sizeof(_BitmapInfoHeader)
        info.biWidth = width
        info.biHeight = -height
        info.biPlanes = 1
        info.biBitCount = 32
        info.biCompression = 0
        pixels = ctypes.create_string_buffer(width * height * 4)
        if not gdi32.GetDIBits(
            memory_dc,
            bitmap,
            0,
            height,
            pixels,
            ctypes.byref(info),
            0,
        ):
            raise OSError("Windows could not read the rendered GUI pixels.")
        Image.frombuffer(
            "RGB", (width, height), pixels, "raw", "BGRX", 0, 1
        ).save(SCREENSHOTS / filename)
    finally:
        gdi32.SelectObject(memory_dc, previous)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(window, source_dc)


def _wait_for(app: AmharicAIApp, predicate, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.update()
        if predicate():
            return
        time.sleep(0.03)
    raise TimeoutError("Timed out while preparing a documentation screenshot.")


def main() -> None:
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    word_image = _render_sample("ሰላም", SAMPLES / "printed-word-selam.png", (560, 170))
    sentence_image = _render_sample(
        "ሰላም   ሀገር", SAMPLES / "printed-sentence.png", (900, 180)
    )

    app = AmharicAIApp()
    app.lift()
    app.focus_force()
    _pump(app, 1.0)
    try:
        character_path = app.split_paths["validation"][0]
        app._select_image(character_path)
        _wait_for(app, lambda: app.current_prediction is not None)
        _capture(app, "character-prediction.png")

        ocr_engine = OCREngine(app.engine)
        word_result = ocr_engine.recognize_word(word_image)
        word_page = app.pages["Word OCR"]
        word_page.source_image = word_image
        word_page.source_path = SAMPLES / "printed-word-selam.png"
        word_page.apply_result(word_result)
        app._show_page("Word OCR")
        _capture(app, "word-ocr.png")
        word_result.overlay.save(SCREENSHOTS / "segmentation-overlay.png")

        sentence_result = ocr_engine.recognize_sentence(sentence_image)
        sentence_page = app.pages["Sentence OCR"]
        sentence_page.source_image = sentence_image
        sentence_page.source_path = SAMPLES / "printed-sentence.png"
        sentence_page.apply_result(sentence_result)
        app._show_page("Sentence OCR")
        _capture(app, "sentence-ocr.png")

        app._show_page("Evaluate")
        app.test_count_var.set("10")
        app._run_automatic_test()
        _wait_for(
            app,
            lambda: "Accuracy:" in app.test_output.get("1.0", "end"),
        )
        _capture(app, "evaluation.png")

        for page_name, filename in (
            ("Model Info", "model-information.png"),
            ("Pipeline", "ocr-pipeline.png"),
            ("Training Graphs", "training-graphs.png"),
        ):
            app._show_page(page_name)
            _capture(app, filename)
    finally:
        app.destroy()

    print(f"Captured {len(list(SCREENSHOTS.glob('*.png')))} real screenshots.")


if __name__ == "__main__":
    main()
