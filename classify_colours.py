import json
import numpy as np
from collections import Counter

# Load your file
with open('COMP-8120/data/json_files/cap.dress.train.json', 'r') as f:
    data = json.load(f)

# Define 5 base color centroids
color_centroids = {
    "red":   np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue":  np.array([0, 0, 255]),
    "black": np.array([0, 0, 0]),
    "white": np.array([255, 255, 255])
}

def classify_color(rgb):
    rgb = np.array(rgb)
    distances = {color: np.linalg.norm(rgb - centroid) for color, centroid in color_centroids.items()}
    return min(distances, key=distances.get)

# Classify all `target_colors` entries
classified_results = []

for entry in data:
    for color in entry.get("target_colors", []):
        if isinstance(color, list):  # Ensure it's a valid RGB list
            label = classify_color(color)
            classified_results.append(label)

# Optional: see frequency distribution
print("Color distribution:", Counter(classified_results))