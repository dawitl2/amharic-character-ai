import os
import random
import glob
from pathlib import Path

def get_random_images(num_images=5):
    """Selects a few random images from the dataset to display in the UI."""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
        
    all_images = []
    # Search for all pngs in all subdirectories of data/
    for ext in ('*.png', '*.jpg'):
        all_images.extend(data_dir.rglob(ext))
        
    if not all_images:
        return []
        
    # Pick randomly
    return random.sample(all_images, min(num_images, len(all_images)))

if __name__ == "__main__":
    images = get_random_images()
    print("Selected random images for UI:")
    for img in images:
        print(img)
