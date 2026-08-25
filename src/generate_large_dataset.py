import random
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import numpy as np

characters = ["ሀ", "ለ", "መ"]

font_paths = [
    r"C:\Windows\Fonts\AbyssinicaSIL-Regular.ttf",
    r"C:\Windows\Fonts\nyala.ttf"
]

image_size = 64
images_per_class = 2000

print(f"Generating large dataset: {images_per_class} images per class...")

for character in characters:
    output_folder = Path("data") / character
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Check how many currently exist to avoid deleting, just overwrite or append
    # Actually, we will just generate synthetic_large_0001.png ...
    for i in range(1, images_per_class + 1):
        
        # 1. Random Font and Size
        font_path = random.choice(font_paths)
        font_size = random.randint(24, 56)
        
        # Create base image
        image = Image.new("L", (image_size, image_size), color=255)
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            # Fallback if font issues
            font = ImageFont.load_default()

        # Get text bounding box
        bbox = draw.textbbox((0, 0), character, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # Base center calculation
        x = (image_size - text_w) // 2 - bbox[0]
        y = (image_size - text_h) // 2 - bbox[1]
        
        # 2. Add random shifts (translations)
        x += random.randint(-8, 8)
        y += random.randint(-8, 8)
        
        draw.text((x, y), character, font=font, fill=0)
        
        # 3. Add random rotation
        angle = random.uniform(-15.0, 15.0)
        # Rotate with white background
        image = image.rotate(angle, fillcolor=255)
        
        # 4. Optional slight blur to simulate bad camera/scan
        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
            
        # 5. Optional noise
        if random.random() < 0.2:
            img_arr = np.array(image)
            noise = np.random.normal(0, 10, img_arr.shape)
            img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_arr)
            
        filename = output_folder / f"synthetic_large_{i:04}.png"
        image.save(filename)
        
    print(f"Finished generating {images_per_class} images for {character}")

print("Large dataset generation complete!")
