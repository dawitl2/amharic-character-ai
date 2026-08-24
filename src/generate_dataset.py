from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from pathlib import Path

characters = ["ሀ", "ለ", "መ"]

font_paths = [
    r"C:\Windows\Fonts\AbyssinicaSIL-Regular.ttf",
    r"C:\Windows\Fonts\nyala.ttf"
]

font_sizes = [36, 40, 44, 48, 52]

image_size = 64

for character in characters:

    output_folder = Path("data") / character
    output_folder.mkdir(parents=True, exist_ok=True)

    image_number = 1

    for font_path in font_paths:

        for font_size in font_sizes:

            image = Image.new(
                "L",
                (image_size, image_size),
                color=255
            )

            draw = ImageDraw.Draw(image)

            font = ImageFont.truetype(
                font_path,
                font_size
            )

            bbox = draw.textbbox(
                (0, 0),
                character,
                font=font
            )

            x = (
                image_size - (bbox[2] - bbox[0])
            ) // 2 - bbox[0]

            y = (
                image_size - (bbox[3] - bbox[1])
            ) // 2 - bbox[1]

            draw.text(
                (x, y),
                character,
                font=font,
                fill=0
            )

            filename = output_folder / f"synthetic_{image_number:03}.png"

            image.save(filename)

            image_number += 1