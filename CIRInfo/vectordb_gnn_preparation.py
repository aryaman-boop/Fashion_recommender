import torch
import clip
from pathlib import Path
import json
from data_utils import FashionIQDataset, targetpad_transform
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

# Set device and load fine-tuned CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)

from data_utils import base_path, targetpad_transform, FashionIQDataset
base_path = Path(base_path)
output_dir = Path("vector_db_gnn_outputs")
output_dir.mkdir(exist_ok=True)

# Load the fine-tuned weights
checkpoint = torch.load(base_path/"models"/"clip_finetuned_on_fiq_RN50x4_2025-06-06_18:42:41"/"saved_models"/"tuned_clip_best.pt", map_location=device)
clip_model.load_state_dict(checkpoint['CLIP'])
clip_model.eval()

categories = ['dress']  # Only dress for now
splits = ['train', 'val', 'test']
preprocess = targetpad_transform(target_ratio=1.25, dim=288)

# Helper to extract RGB
def extract_rgb(color_field):
    if isinstance(color_field, list):
        if len(color_field) == 0:
            return None
        if all(isinstance(x, int) for x in color_field) and len(color_field) == 3:
            return color_field
        if all(isinstance(x, list) and len(x) == 3 for x in color_field):
            return color_field[0]
    return None

# 1. Extract and save all image embeddings and color metadata
for category in categories:
    all_image_ids = set()
    color_metadata = {}
    triplet_list = []
    for split in splits:
        enriched_path = base_path / "captions" / "captions_enriched" / f"cap.{category}.{split}.json"
        with open(enriched_path) as f:
            triplets = json.load(f)
        if split in ['train', 'val']:
            for triplet in triplets:
                ref_id = triplet.get("candidate")
                target_id = triplet.get("target")
                captions = triplet.get("captions", [])
                ref_colors = extract_rgb(triplet.get("candidate_colors"))
                target_colors = extract_rgb(triplet.get("target_colors"))
                all_image_ids.add(ref_id)
                all_image_ids.add(target_id)
                if ref_id and ref_colors:
                    color_metadata[ref_id] = ref_colors
                if target_id and target_colors:
                    color_metadata[target_id] = target_colors
                for idx, mod_text in enumerate(captions):
                    triplet_list.append({
                        "reference": ref_id,
                        "modification": mod_text,
                        "mod_idx": idx,
                        "target": target_id,
                        "split": split
                    })
        elif split == 'test':
            for triplet in triplets:
                ref_id = triplet.get("candidate")
                captions = triplet.get("captions", [])
                ref_colors = extract_rgb(triplet.get("candidate_colors"))
                all_image_ids.add(ref_id)
                if ref_id and ref_colors:
                    color_metadata[ref_id] = ref_colors
                for idx, mod_text in enumerate(captions):
                    triplet_list.append({
                        "reference": ref_id,
                        "modification": mod_text,
                        "mod_idx": idx,
                        "target": None,
                        "split": split
                    })
    # Save color metadata
    with open(output_dir / f"{category}_color_metadata.json", "w") as f:
        json.dump(color_metadata, f)

    # 2. Extract and save image embeddings
    print(f"Extracting image embeddings for {len(all_image_ids)} images...")
    image_id_to_path = {}
    # Assume images are in base_path / "images" / f"{image_id}.jpg"
    for image_id in all_image_ids:
        for ext in [".jpg", ".png", ".jpeg"]:
            img_path = base_path / "fashion-iq-cleaned-images" / f"{image_id}{ext}"
            if img_path.exists():
                image_id_to_path[image_id] = img_path
                break
    image_embeddings = {}
    batch_size = 32
    image_ids = list(image_id_to_path.keys())
    for i in tqdm(range(0, len(image_ids), batch_size), desc="Image Embeddings"):
        batch_ids = image_ids[i:i+batch_size]
        batch_imgs = []
        for img_id in batch_ids:
            img = preprocess(Image.open(str(image_id_to_path[img_id]))).unsqueeze(0)
            batch_imgs.append(img)
        batch_imgs = torch.cat(batch_imgs, dim=0).to(device)
        with torch.no_grad():
            batch_embs = clip_model.encode_image(batch_imgs).cpu().numpy()
        for img_id, emb in zip(batch_ids, batch_embs):
            image_embeddings[img_id] = emb
    np.save(output_dir / f"{category}_image_embeddings.npy", image_embeddings)

    # 3. Extract and save text (modification) embeddings
    print(f"Extracting text embeddings for {len(triplet_list)} modifications...")
    text_embeddings = {}
    for triplet in tqdm(triplet_list, desc="Text Embeddings"):
        mod_text = triplet["modification"]
        ref_id = triplet["reference"]
        mod_idx = triplet["mod_idx"]
        key = f"{ref_id}_{mod_idx}"
        text = clip.tokenize([mod_text]).to(device)
        with torch.no_grad():
            text_emb = clip_model.encode_text(text).cpu().numpy()[0]
        text_embeddings[key] = text_emb
    np.save(output_dir / f"{category}_text_embeddings.npy", text_embeddings)

    # 4. Save triplet/pair information for graph construction
    with open(output_dir / f"{category}_triplets.json", "w") as f:
        json.dump(triplet_list, f, indent=2)

print("All embeddings and metadata saved for GNN pipeline.") 