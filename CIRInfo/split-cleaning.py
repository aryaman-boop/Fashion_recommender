import os
import json
from pathlib import Path

# Set your paths
base_path = Path("/home/aminiant/CIR/fashion-iq")
cleaned_images_dir = base_path / "fashion-iq-cleaned-images"
split_dir = base_path / "image_splits" / "image_splits"

categories = ['dress', 'shirt', 'toptee']
splits = ['train', 'val', 'test']

# Get all valid image names (without extension)
valid_images = set(os.path.splitext(f)[0] for f in os.listdir(cleaned_images_dir) if f.endswith('.png'))

for category in categories:
    for split in splits:
        split_path = split_dir / f"split.{category}.{split}.json"
        if not split_path.exists():
            print(f"Split file not found: {split_path}")
            continue
        with open(split_path) as f:
            split_images = json.load(f)
        # Filter to only those that exist in valid_images
        filtered_split = [img for img in split_images if img in valid_images]
        print(f"{category} {split}: {len(split_images)} -> {len(filtered_split)} images")
        # Save the cleaned split (overwrite original)
        with open(split_path, "w") as f:
            json.dump(filtered_split, f)
