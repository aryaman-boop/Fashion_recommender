import os
import json
import re
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------------
# Setup
# -------------------------------
INPUT_DIR = "data/json_files"
OUTPUT_FILE = os.path.join(INPUT_DIR, "instruction_dataset.jsonl")

ADJUSTMENT_KEYWORDS = [
    "long", "longer", "short", "shorter", "fitted", "casual",
    "sleeves", "sleeveless", "shorter sleeves", "longer sleeves",
    "v neck", "tank", "neck", "neckline", "collar", "strapless", "straps",
    "solid", "pattern", "graphic", "print", "striped", "stripes", "floral",
    "design", "logo", "revealing", "brighter", "darker", "lighter", "colorful",
    "buttons", "top", "shirt", "halter"
]

COLOR_RGB_LOOKUP = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "pink": (255, 192, 203),
    "purple": (128, 0, 128),
    "orange": (255, 165, 0),
    "brown": (165, 42, 42),
    "maroon": (128, 0, 0),
    "navy": (0, 0, 128),
    "beige": (245, 245, 220),
    "salmon": (250, 128, 114),
    "teal": (0, 128, 128),
    "gold": (255, 215, 0),
    "silver": (192, 192, 192),
    "olive": (128, 128, 0)
}

COLOR_NAMES = list(COLOR_RGB_LOOKUP.keys())

# -------------------------------
# Sentence Embedding for Fuzzy Color Matching
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
color_name_embeddings = model.encode(COLOR_NAMES)

def match_color_to_rgb(description):
    desc_emb = model.encode([description])
    scores = cosine_similarity(desc_emb, color_name_embeddings)[0]
    best_index = np.argmax(scores)
    best_color = COLOR_NAMES[best_index]
    return best_color, COLOR_RGB_LOOKUP[best_color]

# -------------------------------
# Helpers
# -------------------------------
def extract_color_name(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    matches = list(set(words) & set(COLOR_NAMES))
    if matches:
        return matches[0], COLOR_RGB_LOOKUP[matches[0]]
    if not words:
        return "unknown", (0, 0, 0)
    # fuzzy matching
    return match_color_to_rgb(text)

def extract_adjustments(text):
    found = [kw for kw in ADJUSTMENT_KEYWORDS if kw in text.lower()]
    return ", ".join(sorted(set(found))) if found else "none"

def normalize_rgb(rgb):
    return f"({rgb[0]}, {rgb[1]}, {rgb[2]})"

def load_json_file(path):
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in open(path) if line.strip()]
    else:
        return json.load(open(path))

# -------------------------------
# Main Processing Function
# -------------------------------
def process_file(file_path):
    try:
        data = load_json_file(file_path)
    except Exception as e:
        print(f"⚠️ Skipping {file_path}: {e}")
        return []

    examples = []
    for item in data:
        if not isinstance(item, dict):
            continue

        captions = item.get("captions", [])
        candidate_colors = item.get("candidate_colors", [])

        for i, caption in enumerate(captions):
            color_name, rgb = extract_color_name(caption)
            adjustments = extract_adjustments(caption)

            # Normalize RGB using candidate_colors fallback logic
            try:
                if isinstance(candidate_colors, list):
                    if all(isinstance(c, int) for c in candidate_colors) and len(candidate_colors) == 3:
                        # Format: [R, G, B]
                        rgb = tuple(candidate_colors)
                    elif all(isinstance(c, list) and len(c) == 3 for c in candidate_colors):
                        # Format: [[R, G, B], [R, G, B], ...]
                        rgb = tuple(candidate_colors[min(i, len(candidate_colors) - 1)])
            except Exception as e:
                print(f"⚠️ RGB parsing failed for caption: {caption} — {e}")

            examples.append({
                "instruction": "Extract dress features and RGB color from the description.",
                "input": caption.strip(),
                "output": f"Adjustments: {adjustments}\nColor: {color_name}\nRGB: {normalize_rgb(rgb)}"
            })

    return examples

# -------------------------------
# Execute All
# -------------------------------
all_examples = []
all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json") or f.endswith(".jsonl")]

for fname in all_files:
    path = os.path.join(INPUT_DIR, fname)
    print(f"📂 Processing: {fname}")
    examples = process_file(path)
    all_examples.extend(examples)

with open(OUTPUT_FILE, 'w') as out_file:
    for sample in all_examples:
        json.dump(sample, out_file)
        out_file.write('\n')

print(f"\n✅ Done! Saved {len(all_examples)} examples to {OUTPUT_FILE}")