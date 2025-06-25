import torch
import numpy as np
import hnswlib
import clip
from PIL import Image
from test2 import Combiner
import json
import matplotlib.pyplot as plt
import os
from data_utils import base_path
from pathlib import Path

results_dir = "retrieval_results"
os.makedirs(results_dir, exist_ok=True)

# --- Settings ---
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_name = "RN50x4"
clip_finetuned_path = base_path / "models/clip_finetuned_on_fiq_RN50x4_2025-06-06_18:42:41/saved_models/tuned_clip_best.pt"
combiner_path = base_path / "models/combiner_trained_on_fiq_RN50x4_2025-06-10_16:30:54/saved_models/combiner.pt"
image_embeddings_path = Path("vector_db_gnn_outputs/dress_image_embeddings.npy")
triplets_json_path =  Path("vector_db_gnn_outputs/dress_triplets.json")
image_folder = base_path / "fashion-iq-cleaned-images"

# --- Load fine-tuned CLIP model ---
clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
clip_model.eval()
# Load fine-tuned weights
clip_model.load_state_dict(torch.load(clip_finetuned_path, map_location=device)["CLIP"])

# --- Load Combiner ---
feature_dim = clip_model.visual.output_dim
combiner = Combiner(feature_dim, 512, 512).to(device)
combiner.eval()
combiner.load_state_dict(torch.load(combiner_path, map_location=device)["Combiner"])

# --- Load vector DB and image IDs ---
# Load the dict
embeddings = np.load(image_embeddings_path, allow_pickle=True)
if isinstance(embeddings, np.ndarray) and embeddings.dtype == object and embeddings.shape == ():  # 0-d array
    embeddings = embeddings.item()

# Now embeddings is a dict: {img_id: feature_vector}
image_ids = list(embeddings.keys())
embedding_list = [np.array(embeddings[img_id]) for img_id in image_ids]
embeddings_array = np.stack(embedding_list)
print("embeddings_array shape:", embeddings_array.shape)
with open(triplets_json_path, "r") as f:
    triplets = json.load(f)

# Collect all unique image IDs from 'reference' and 'target'
image_ids = set()
for triplet in triplets:
    image_ids.add(triplet["reference"])
    image_ids.add(triplet["target"])
image_ids = list(image_ids)

# --- Build HNSW index ---
dim = embeddings_array.shape[1]
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=embeddings_array.shape[0], ef_construction=200, M=16)
index.add_items(embeddings_array, np.arange(embeddings_array.shape[0]))
index.set_ef(50)

with open("vector_db_gnn_outputs/dress_color_metadata.json") as f:
    color_metadata = json.load(f)

missing = [img_id for img_id in image_ids if img_id not in color_metadata]
print(f"{len(missing)} out of {len(image_ids)} image IDs are missing from color metadata.")

def get_target_color_from_text(text):
    # Simple mapping, you can expand this
    color_map = {
        "blue": np.array([0, 0, 255]),
        "red": np.array([255, 0, 0]),
        "green": np.array([0, 255, 0]),
        "black": np.array([0, 0, 0]),
        "white": np.array([255, 255, 255]),
        "gray": np.array([128, 128, 128]),
        # Add more as needed
    }
    for color in color_map:
        if color in text.lower():
            return color_map[color]
    return None

# --- Interactive loop ---
while True:
    print("\n--- FashionIQ Interactive Retrieval ---")
    img_path = input("Enter path to reference image (or 'q' to quit): ")
    if img_path.lower() == 'q':
        break
    if not os.path.isfile(img_path):
        print("Image file not found. Try again.")
        continue
    mod_text = input("Enter modification text: ")
    if not mod_text.strip():
        print("Modification text cannot be empty. Try again.")
        continue

    # --- Preprocess and encode ---
    ref_img = Image.open(img_path).convert("RGB")
    ref_img_tensor = clip_preprocess(ref_img).unsqueeze(0).to(device)
    with torch.no_grad():
        ref_img_feat = clip_model.encode_image(ref_img_tensor).float()
        text_tokens = clip.tokenize([mod_text]).to(device)
        text_feat = clip_model.encode_text(text_tokens).float()

    # --- Fuse with Combiner ---
    with torch.no_grad():
        joint_feat = combiner.combine_features(ref_img_feat, text_feat)  # shape: [1, D]
        joint_feat = joint_feat.cpu().numpy().astype(np.float32)

    # --- Search in HNSW index ---
    labels, distances = index.knn_query(joint_feat, k=20)
    top_indices = labels[0]  # shape: (10,)

    # --- Show results ---
    print("Top 10 relevant images:")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}: {image_ids[idx]}")

    # Ask for color and rerank
    rgb_input = input("Enter target color as R,G,B (e.g., 0,0,255 for blue): ")
    try:
        target_rgb = np.array([int(x) for x in rgb_input.split(",")])
        assert target_rgb.shape == (3,)
    except Exception as e:
        print("Invalid RGB input. Please enter three comma-separated integers.")
        continue

    # Rerank
    color_distances = []
    valid_indices = []
    for idx in top_indices:
        img_id = image_ids[idx]
        if img_id in color_metadata:
            img_rgb = np.array(color_metadata[img_id])
            dist = np.linalg.norm(img_rgb - target_rgb)
            color_distances.append(dist)
            valid_indices.append(idx)
        else:
            print(f"Warning: {img_id} not found in color metadata, skipping.")

    if color_distances:
        reranked = np.argsort(color_distances)
        top_indices = [valid_indices[i] for i in reranked]

        print("Valid indices after color metadata filtering:", valid_indices)
        print("Color distances:", color_distances)
        print("Number of images to display:", len(valid_indices))

        print("Top 10 color-aware relevant images:")
        for i, idx in enumerate(top_indices):
            print(f"{i+1}: {image_ids[idx]}")
        plt.figure(figsize=(20, 10))
        for i, idx in enumerate(top_indices):
            img_id = image_ids[idx]
            for ext in [".jpg", ".jpeg", ".png"]:
                img_file = os.path.join(image_folder, f"{img_id}{ext}")
                if os.path.isfile(img_file):
                    img = Image.open(img_file)
                    plt.subplot(4, 5, i+1)
                    plt.imshow(img)
                    plt.title(f"Rank {i+1}")
                    plt.axis('off')
                    break
            save_path = os.path.join(results_dir, f"rank_{i+1}_{img_id}{ext}")
            img.save(save_path)
            print(f"Saved {save_path}")
        plt.tight_layout()
        grid_path = os.path.join(results_dir, "retrieved_grid.png")
        plt.savefig(grid_path)
        print(f"Saved grid of results to {grid_path}")
        plt.close()
    else:
        print("No retrieved images had color metadata. Skipping color rerank.")

