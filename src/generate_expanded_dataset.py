import random
import os
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Phase 24: Expanded Character Classes (10 classes)
characters = ["ሀ", "ለ", "ሐ", "መ", "ሠ", "ረ", "ሰ", "ሸ", "ቀ", "በ"]

# Phase 23: Improved Dataset Quality (More fonts)
font_paths = [
    r"C:\Windows\Fonts\AbyssinicaSIL-Regular.ttf",
    r"C:\Windows\Fonts\nyala.ttf",
    r"C:\Windows\Fonts\ebrima.ttf",
    r"C:\Windows\Fonts\ebrimabd.ttf"
]

images_per_class = 3000

print(f"Generating expanded dataset: {images_per_class} images per class for {len(characters)} classes...")

def add_scanning_artifacts(draw, width, height):
    # Random faint lines (scanning artifacts)
    if random.random() < 0.2:
        for _ in range(random.randint(1, 3)):
            y = random.randint(0, height - 1)
            # draw a faint horizontal line
            draw.line([(0, y), (width, y)], fill=random.randint(180, 230), width=1)
            
    if random.random() < 0.2:
        for _ in range(random.randint(1, 3)):
            x = random.randint(0, width - 1)
            # draw a faint vertical line
            draw.line([(x, 0), (x, height)], fill=random.randint(180, 230), width=1)

for character in characters:
    output_folder = Path("data") / character
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # We will generate synthetic_exp_0001.png etc.
    existing = list(output_folder.glob("*.png"))
    start_idx = len(existing) + 1
    
    for i in range(start_idx, start_idx + images_per_class):
        # Mix 64x64 and 128x128 randomly
        image_size = random.choice([64, 128])
        
        font_path = random.choice(font_paths)
        if image_size == 64:
            font_size = random.randint(24, 56)
            shift_range = 8
        else:
            font_size = random.randint(48, 100)
            shift_range = 14
            
        # Background variation (not always pure white)
        bg_color = random.randint(220, 255)
        image = Image.new("L", (image_size, image_size), color=bg_color)
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        # Text bounding box
        bbox = draw.textbbox((0, 0), character, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # Base center calculation
        x = (image_size - text_w) // 2 - bbox[0]
        y = (image_size - text_h) // 2 - bbox[1]
        
        # Translations (Horizontal and Vertical movement)
        x += random.randint(-shift_range, shift_range)
        y += random.randint(-shift_range, shift_range)
        
        # Draw text (with varying text color for contrast variation)
        text_color = random.randint(0, 80)
        draw.text((x, y), character, font=font, fill=text_color)
        
        # Rotations
        angle = random.uniform(-15.0, 15.0)
        image = image.rotate(angle, fillcolor=bg_color)
        
        # Artifacts
        draw_after_rot = ImageDraw.Draw(image)
        add_scanning_artifacts(draw_after_rot, image_size, image_size)
        
        # Contrast & Brightness variations
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.5, 1.5))
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.5, 1.5))
            
        # Blur
        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
            
        # Noise
        if random.random() < 0.2:
            img_arr = np.array(image, dtype=np.float32)
            noise = np.random.normal(0, random.uniform(5, 15), img_arr.shape)
            img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_arr)
            
        filename = output_folder / f"synthetic_exp_{i:04}.png"
        image.save(filename)
        
    print(f"Generated {images_per_class} images for {character}")

print("Expanded dataset generation complete!")
