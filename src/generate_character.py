from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

image = Image.new("L", (64, 64), color=255)

draw = ImageDraw.Draw(image)

font = ImageFont.truetype(
    r"C:\Windows\Fonts\AbyssinicaSIL-Regular.ttf",
    48
)

character = "ለ"

bbox = draw.textbbox((0, 0), character, font=font)

x = (64 - (bbox[2] - bbox[0])) // 2 - bbox[0]
y = (64 - (bbox[3] - bbox[1])) // 2 - bbox[1]

draw.text((x, y), character, font=font, fill=0)

print(bbox)

image.save(f"data/{character}/sample_001.png")