import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ----------------------------
# Load the JSON dataset
# ----------------------------
with open('COMP-8120/data/json_files/cap.dress.train.json', 'r') as f:
    data = json.load(f)

# ----------------------------
# Define RGB centroids for 5 color classes
# ----------------------------
color_centroids = {
    "red":   np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue":  np.array([0, 0, 255]),
    "black": np.array([0, 0, 0]),
    "white": np.array([255, 255, 255])
}

# ----------------------------
# Function to classify RGB to nearest color category
# ----------------------------
def classify_color(rgb):
    rgb = np.array(rgb)
    distances = {color: np.linalg.norm(rgb - centroid) for color, centroid in color_centroids.items()}
    return min(distances, key=distances.get)

# ----------------------------
# Process the JSON and classify each dress
# ----------------------------
records = []

for entry in data:
    target_id = entry.get("target", "")
    for color in entry.get("target_colors", []):
        if isinstance(color, list) and len(color) == 3:
            label = classify_color(color)
            records.append({
                "dress_id": target_id,
                "rgb": color,
                "classified_color": label
            })

# ----------------------------
# Convert to DataFrame
# ----------------------------
df = pd.DataFrame(records)

# ----------------------------
# Plot color distribution
# ----------------------------
color_distribution = Counter(df['classified_color'])

plt.figure(figsize=(8, 5))
plt.bar(color_distribution.keys(), color_distribution.values(), color='skyblue')
plt.title("Dress Color Classification Distribution")
plt.xlabel("Color Category")
plt.ylabel("Number of Dresses")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ----------------------------
# Optional: Save results to CSV
# ----------------------------
df.to_csv('classified_dress_colors.csv', index=False)
print("✅ Classification complete. Results saved to 'classified_dress_colors.csv'")