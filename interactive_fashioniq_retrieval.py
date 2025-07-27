import os
import re
# Set OpenBLAS environment variable to prevent thread issues
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import torch
import numpy as np
import hnswlib
import clip
from PIL import Image
from attention_fusion_combiner import AttentionFusionCombiner
import json
import matplotlib.pyplot as plt
from data_utils import base_path
from pathlib import Path
from sklearn.neighbors import KDTree  # <-- Add this import
from sklearn.cluster import KMeans    # <-- Add this import

results_dir = "retrieval_results"
os.makedirs(results_dir, exist_ok=True)

# --- Settings ---
# Use GPU 1 to avoid memory conflicts with other processes
device = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu"
clip_model_name = "RN50x4"
clip_finetuned_path = base_path / "/home/banik8/composed image retrieval/Fashion_recommender/clip_finetuned_on_fiq_RN50x4/saved_models/tuned_clip_best.pt"
combiner_path = base_path / "/home/banik8/composed image retrieval/Fashion_recommender/checkpoint_epoch_19.pt"
image_folder = base_path

# Available categories
available_categories = ["dress", "toptee", "shirt"]

# --- Load fine-tuned CLIP model ---
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
clip_model.eval()
# Load fine-tuned weights
clip_model.load_state_dict(torch.load(clip_finetuned_path, map_location=device)["CLIP"])

# --- Load Combiner ---
# feature_dim = clip_model.visual.output_dim  # Remove this line, not used
combiner = AttentionFusionCombiner(640, 512, 512).to(device)
combiner.eval()

# Load checkpoint and remove 'module.' prefix if present
checkpoint = torch.load(combiner_path, map_location=device)
state_dict = checkpoint["model_state_dict"]

# Remove 'module.' prefix if it exists
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[new_key] = v

# Diagnostic: print keys in checkpoint and model
print("\nCheckpoint keys:", list(new_state_dict.keys()))
print("\nModel keys:", list(combiner.state_dict().keys()))

combiner.load_state_dict(new_state_dict)

# --- Debug: Print checkpoint parameter shapes ---
def print_checkpoint_shapes(path):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    print(f"\nParameter shapes in checkpoint {path}:")
    for k, v in state_dict.items():
        print(f"{k}: {getattr(v, 'shape', type(v))}")

print_checkpoint_shapes(combiner_path)

# --- Load all categories data ---
def load_category_data(category):
    """Load embeddings, triplets, and color metadata for a specific category"""
    image_embeddings_path = Path(f"vector_db_gnn_outputs/{category}_image_embeddings.npy")
    triplets_json_path = Path(f"vector_db_gnn_outputs/{category}_triplets.json")
    color_metadata_path = Path(f"vector_db_gnn_outputs/{category}_color_metadata.json")
    
    # Load embeddings
    embeddings = np.load(image_embeddings_path, allow_pickle=True)
    if isinstance(embeddings, np.ndarray) and embeddings.dtype == object and embeddings.shape == ():
        embeddings = embeddings.item()
    
    # Load triplets
    with open(triplets_json_path, "r") as f:
        triplets = json.load(f)
    
    # Load color metadata
    with open(color_metadata_path, "r") as f:
        color_metadata = json.load(f)
    
    return embeddings, triplets, color_metadata

# Load categories on-demand to reduce memory usage
all_embeddings = {}
all_triplets = {}
all_color_metadata = {}
all_image_ids = set()
category_mapping = {}  # Maps image_id to category

print("Loading category data...")
for category in available_categories:
    print(f"Loading {category} data...")
    try:
        embeddings, triplets, color_metadata = load_category_data(category)
        all_embeddings[category] = embeddings
        all_triplets[category] = triplets
        all_color_metadata[category] = color_metadata
        
        # Collect all image IDs from this category
        category_image_ids = set()
        for triplet in triplets:
            category_image_ids.add(triplet["reference"])
            category_image_ids.add(triplet["target"])
        all_image_ids.update(category_image_ids)
        
        # Create category mapping
        for img_id in embeddings.keys():
            category_mapping[img_id] = category
            
        print(f"Loaded {len(category_image_ids)} images for {category}")
    except Exception as e:
        print(f"Error loading {category}: {e}")
        continue

print(f"Total unique images across all categories: {len(all_image_ids)}")

# --- Build HNSW index for all categories ---
print("Building search index...")

# Create combined embeddings array
combined_embeddings = []
combined_image_ids = []

for category in available_categories:
    if category in all_embeddings:
        embeddings = all_embeddings[category]
        for img_id, embedding in embeddings.items():
            combined_embeddings.append(np.array(embedding))
            combined_image_ids.append(img_id)

combined_embeddings_array = np.stack(combined_embeddings)
print(f"Combined embeddings shape: {combined_embeddings_array.shape}")

dim = combined_embeddings_array.shape[1]
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=combined_embeddings_array.shape[0], ef_construction=200, M=16)
index.add_items(combined_embeddings_array, np.arange(combined_embeddings_array.shape[0]))
index.set_ef(50)

# --- Build KDTree for color pre-filtering (combined) ---
print("Building color clustering...")
color_img_ids = []
color_vectors = []

for img_id in combined_image_ids:
    # Find which category this image belongs to
    category = category_mapping[img_id]
    if img_id in all_color_metadata[category]:
        color_img_ids.append(img_id)
        color_vectors.append(all_color_metadata[category][img_id])

if color_vectors:
    color_vectors = np.array(color_vectors)
else:
    print("Warning: No color metadata found")
    color_vectors = np.array([])

# --- Cluster colors and build per-cluster KD-Trees ---
if len(color_vectors) > 0:
    num_clusters = min(8, len(color_vectors))  # Don't create more clusters than data points
    if num_clusters > 1:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(color_vectors)
        cluster_labels = kmeans.labels_  # Cluster assignment for each image
        cluster_centers = kmeans.cluster_centers_
    else:
        # If only one data point, create a single cluster
        cluster_labels = np.zeros(len(color_vectors), dtype=int)
        cluster_centers = color_vectors.reshape(1, -1)
    
    # Build a KD-Tree for each cluster
    cluster_img_ids = [[] for _ in range(num_clusters)]
    cluster_color_vectors = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(cluster_labels):
        cluster_img_ids[label].append(color_img_ids[idx])
        cluster_color_vectors[label].append(color_vectors[idx])
    cluster_kdtrees = []
    for i in range(num_clusters):
        if cluster_color_vectors[i]:
            cluster_kdtrees.append(KDTree(np.array(cluster_color_vectors[i])))
        else:
            cluster_kdtrees.append(None)
else:
    # No color data available
    cluster_centers = np.array([])
    cluster_kdtrees = []
    cluster_img_ids = []

missing = [img_id for img_id in combined_image_ids if img_id not in [cat for cat in all_color_metadata.values() for img_id in cat]]
print(f"{len(missing)} out of {len(combined_image_ids)} image IDs are missing from color metadata.")

def get_target_color_from_text(text):
    """Extract target color from text with support for color combinations and ranges"""
    # Extended color map with more colors and combinations
    color_map = {
        # Basic colors
        "blue": np.array([0, 0, 255]),
        "red": np.array([255, 0, 0]),
        "green": np.array([0, 255, 0]),
        "black": np.array([0, 0, 0]),
        "white": np.array([255, 255, 255]),
        "gray": np.array([128, 128, 128]),
        "yellow": np.array([255, 255, 0]),
        "orange": np.array([255, 165, 0]),
        "pink": np.array([255, 192, 203]),
        "purple": np.array([128, 0, 128]),
        "brown": np.array([139, 69, 19]),
        "beige": np.array([245, 245, 220]),
        "cyan": np.array([0, 255, 255]),
        "magenta": np.array([255, 0, 255]),
        "navy": np.array([0, 0, 128]),
        "maroon": np.array([128, 0, 0]),
        "olive": np.array([128, 128, 0]),
        "teal": np.array([0, 128, 128]),
        "gold": np.array([255, 215, 0]),
        "silver": np.array([192, 192, 192]),
        "violet": np.array([238, 130, 238]),
        
        # Color combinations and ranges
        "greenish-yellow": np.array([173, 255, 47]),  # GreenYellow
        "yellowish-green": np.array([173, 255, 47]),  # Same as greenish-yellow
        "blue-green": np.array([0, 128, 128]),  # Teal
        "green-blue": np.array([0, 128, 128]),  # Teal
        "red-orange": np.array([255, 69, 0]),  # RedOrange
        "orange-red": np.array([255, 69, 0]),  # RedOrange
        "blue-purple": np.array([75, 0, 130]),  # Indigo
        "purple-blue": np.array([75, 0, 130]),  # Indigo
        "pink-red": np.array([220, 20, 60]),  # Crimson
        "red-pink": np.array([220, 20, 60]),  # Crimson
        "yellow-orange": np.array([255, 140, 0]),  # DarkOrange
        "orange-yellow": np.array([255, 140, 0]),  # DarkOrange
        "light-blue": np.array([173, 216, 230]),  # LightBlue
        "dark-blue": np.array([0, 0, 139]),  # DarkBlue
        "light-green": np.array([144, 238, 144]),  # LightGreen
        "dark-green": np.array([0, 100, 0]),  # DarkGreen
        "light-red": np.array([255, 182, 193]),  # LightCoral
        "dark-red": np.array([139, 0, 0]),  # DarkRed
        "light-yellow": np.array([255, 255, 224]),  # LightYellow
        "dark-yellow": np.array([184, 134, 11]),  # DarkGoldenrod
        "light-pink": np.array([255, 228, 225]),  # MistyRose
        "dark-pink": np.array([199, 21, 133]),  # MediumVioletRed
        "light-purple": np.array([221, 160, 221]),  # Plum
        "dark-purple": np.array([47, 0, 150]),  # Indigo
        "light-orange": np.array([255, 218, 185]),  # PeachPuff
        "dark-orange": np.array([255, 69, 0]),  # RedOrange
        "light-gray": np.array([211, 211, 211]),  # LightGray
        "dark-gray": np.array([64, 64, 64]),  # DarkGray
        "cream": np.array([255, 253, 208]),  # Cream
        "ivory": np.array([255, 255, 240]),  # Ivory
        "tan": np.array([210, 180, 140]),  # Tan
        "khaki": np.array([240, 230, 140]),  # Khaki
        "coral": np.array([255, 127, 80]),  # Coral
        "salmon": np.array([250, 128, 114]),  # Salmon
        "turquoise": np.array([64, 224, 208]),  # Turquoise
        "lime": np.array([0, 255, 0]),  # Lime
        "mint": np.array([245, 255, 250]),  # MintCream
        "lavender": np.array([230, 230, 250]),  # Lavender
        "rose": np.array([255, 0, 127]),  # Rose
        "burgundy": np.array([128, 0, 32]),  # Burgundy
        "emerald": np.array([0, 128, 0]),  # Green
        "ruby": np.array([155, 17, 30]),  # Ruby
        "sapphire": np.array([15, 82, 186]),  # Sapphire
        "amber": np.array([255, 191, 0]),  # Amber
        "jade": np.array([0, 168, 107]),  # Jade
        "copper": np.array([184, 115, 51]),  # Copper
        "bronze": np.array([205, 127, 50]),  # Bronze
        "charcoal": np.array([54, 69, 79]),  # Charcoal
        "navy-blue": np.array([0, 0, 128]),  # Navy
        "forest-green": np.array([34, 139, 34]),  # ForestGreen
        "sky-blue": np.array([135, 206, 235]),  # SkyBlue
        "royal-blue": np.array([65, 105, 225]),  # RoyalBlue
        "hot-pink": np.array([255, 20, 147]),  # DeepPink
        "neon-green": np.array([57, 255, 20]),  # NeonGreen
        "neon-blue": np.array([31, 81, 255]),  # NeonBlue
        "neon-pink": np.array([255, 16, 240]),  # NeonPink
    }
    
    text_lower = text.lower()
    
    # First, try to find exact color matches
    for color in color_map:
        if color in text_lower:
            return color_map[color]
    
    # If no exact match, try to parse color combinations
    # Look for patterns like "color1-color2", "color1ish color2", etc.
    
    # Pattern for "color1-color2" or "color1 color2"
    color_patterns = [
        r'(\w+)-(\w+)',  # green-yellow
        r'(\w+)\s+(\w+)',  # green yellow
        r'(\w+)ish\s+(\w+)',  # greenish yellow
        r'(\w+)\s+(\w+)ish',  # green yellowish
    ]
    
    for pattern in color_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            color1, color2 = match
            # Check if both colors exist in our map
            if color1 in color_map and color2 in color_map:
                # Blend the colors (simple average)
                blended_color = (color_map[color1] + color_map[color2]) / 2
                return blended_color.astype(int)
    
    # If still no match, try to find any color-related words and use a broader search
    color_keywords = ["color", "colored", "colour", "coloured", "hue", "shade", "tint"]
    if any(keyword in text_lower for keyword in color_keywords):
        # Return a neutral color to indicate color search but no specific color
        return np.array([128, 128, 128])  # Gray as fallback
    
    return None

def suggest_category_from_query(query):
    """Suggest category based on query text"""
    query_lower = query.lower()
    if any(word in query_lower for word in ["dress", "gown", "frock", "outfit"]):
        return "dress"
    elif any(word in query_lower for word in ["shirt", "blouse", "top"]):
        return "shirt"
    elif any(word in query_lower for word in ["tee", "t-shirt", "tshirt", "tank"]):
        return "toptee"
    else:
        return None

def enhance_modification_text(text):
    """Enhance modification text to be more specific for better matching"""
    text_lower = text.lower()
    enhanced_text = text
    
    # Handle sleeve length descriptions
    if any(word in text_lower for word in ["long sleeve", "long-sleeve", "longsleeve", "long sleeved", "long-sleeved", "long sleeves"]):
        enhanced_text += " with long sleeves extending to wrists, full sleeve coverage"
    elif any(word in text_lower for word in ["short sleeve", "short-sleeve", "shortsleeve", "short sleeved", "short-sleeved", "short sleeves"]):
        enhanced_text += " with short sleeves above elbows, partial sleeve coverage"
    elif any(word in text_lower for word in ["sleeveless", "no sleeves", "tank top", "spaghetti straps"]):
        enhanced_text += " without sleeves, sleeveless design, no sleeve coverage"
    
    # Handle neckline descriptions
    if any(word in text_lower for word in ["v-neck", "v neck", "vneck"]):
        enhanced_text += " with v-shaped neckline"
    elif any(word in text_lower for word in ["crew neck", "crewneck", "round neck"]):
        enhanced_text += " with round crew neck"
    elif any(word in text_lower for word in ["scoop neck", "scoopneck"]):
        enhanced_text += " with wide scoop neckline"
    
    # Handle fit descriptions
    if any(word in text_lower for word in ["loose", "baggy", "oversized"]):
        enhanced_text += " loose fitting, relaxed fit"
    elif any(word in text_lower for word in ["tight", "fitted", "form-fitting"]):
        enhanced_text += " tight fitting, form-fitting"
    
    # Handle style descriptions
    if any(word in text_lower for word in ["casual", "everyday"]):
        enhanced_text += " casual style, everyday wear"
    elif any(word in text_lower for word in ["formal", "business", "professional"]):
        enhanced_text += " formal style, professional look"
    
    return enhanced_text

# --- Interactive loop ---
while True:
    print("\n--- FashionIQ Interactive Retrieval ---")
    print(f"Available categories: {', '.join(available_categories)}")
    
    img_input = input("Enter image ID (e.g., B002ZG8I60) or 'q' to quit: ")
    if img_input.lower() == 'q':
        break
    
    # Try to find the image file with the given ID
    img_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        potential_path = os.path.join(image_folder, f"{img_input}{ext}")
        if os.path.isfile(potential_path):
            img_path = potential_path
            break
    
    if img_path is None:
        print(f"Image file not found for ID: {img_input}. Try again.")
        continue
    
    mod_text = input("Enter modification text: ")
    if not mod_text.strip():
        print("Modification text cannot be empty. Try again.")
        continue

    # Enhance the modification text for better attribute matching
    enhanced_mod_text = enhance_modification_text(mod_text)
    if enhanced_mod_text != mod_text:
        print(f"Enhanced query: '{enhanced_mod_text}'")
        print("Detected attributes:")
        text_lower = mod_text.lower()
        if any(word in text_lower for word in ["long sleeve", "long-sleeve", "longsleeve", "long sleeved", "long-sleeved", "long sleeves"]):
            print("  ✓ Long sleeves")
        if any(word in text_lower for word in ["short sleeve", "short-sleeve", "shortsleeve", "short sleeved", "short-sleeved", "short sleeves"]):
            print("  ✓ Short sleeves")
        if any(word in text_lower for word in ["sleeveless", "no sleeves", "tank top", "spaghetti straps"]):
            print("  ✓ Sleeveless")
        if any(word in text_lower for word in ["v-neck", "v neck", "vneck"]):
            print("  ✓ V-neck")
        if any(word in text_lower for word in ["crew neck", "crewneck", "round neck"]):
            print("  ✓ Crew neck")
        if any(word in text_lower for word in ["loose", "baggy", "oversized"]):
            print("  ✓ Loose fit")
        if any(word in text_lower for word in ["tight", "fitted", "form-fitting"]):
            print("  ✓ Fitted")
        
        confirm = input("Use enhanced query? (y/n, default: y): ").strip().lower()
        if confirm == 'n':
            enhanced_mod_text = mod_text
            print("Using original query.")

    # Suggest category based on query
    suggested_category = suggest_category_from_query(mod_text)
    if suggested_category:
        print(f"Query suggests category: {suggested_category}")
    
    # Ask user for category preference
    print("Search options:")
    print("1. Search in all categories")
    if suggested_category:
        print(f"2. Search only in {suggested_category}")
    print("3. Choose specific category")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        search_categories = available_categories
        print("Searching in all categories...")
    elif choice == "2" and suggested_category:
        search_categories = [suggested_category]
        print(f"Searching only in {suggested_category}...")
    elif choice == "3":
        print("Available categories:", ", ".join(available_categories))
        selected = input("Enter category name: ").strip().lower()
        if selected in available_categories:
            search_categories = [selected]
            print(f"Searching in {selected}...")
        else:
            print("Invalid category. Searching in all categories.")
            search_categories = available_categories
    else:
        print("Invalid choice. Searching in all categories.")
        search_categories = available_categories

    # --- Preprocess and encode ---
    ref_img = Image.open(img_path).convert("RGB")
    ref_img_tensor = clip_preprocess(ref_img).unsqueeze(0).to(device)
    with torch.no_grad():
        ref_img_feat = clip_model.encode_image(ref_img_tensor).float()
        text_tokens = clip.tokenize([enhanced_mod_text]).to(device)
        text_feat = clip_model.encode_text(text_tokens).float()

    # --- Fuse with Combiner ---
    with torch.no_grad():
        joint_feat = combiner.combine_features(ref_img_feat, text_feat)  # shape: [1, D]
        joint_feat = joint_feat.cpu().numpy().astype(np.float32)

    # --- Filter embeddings based on selected categories ---
    if len(search_categories) < len(available_categories):
        # Filter to only selected categories
        filtered_indices = []
        filtered_embeddings = []
        filtered_img_ids = []
        
        for i, img_id in enumerate(combined_image_ids):
            if category_mapping[img_id] in search_categories:
                filtered_indices.append(i)
                filtered_embeddings.append(combined_embeddings_array[i])
                filtered_img_ids.append(img_id)
        
        filtered_embeddings_array = np.stack(filtered_embeddings)
        print(f"Filtered to {len(filtered_img_ids)} images from {search_categories}")
        
        # Build temporary index for filtered data
        temp_index = hnswlib.Index(space='cosine', dim=filtered_embeddings_array.shape[1])
        temp_index.init_index(max_elements=filtered_embeddings_array.shape[0], ef_construction=100, M=16)
        temp_index.add_items(filtered_embeddings_array, np.arange(filtered_embeddings_array.shape[0]))
        temp_index.set_ef(50)
        search_index = temp_index
        search_img_ids = filtered_img_ids
    else:
        # Use all categories
        search_index = index
        search_img_ids = combined_image_ids

    # --- Automatic color pre-filtering if color is mentioned ---
    target_rgb = get_target_color_from_text(mod_text)
    if target_rgb is not None and len(cluster_centers) > 0:
        print(f"Detected color in prompt: {target_rgb}. Applying color pre-filtering.")
        
        # Enhanced color filtering for combinations and ranges
        text_lower = mod_text.lower()
        
        # Determine color radius based on the type of color request
        if any(word in text_lower for word in ["ish", "ish-", "-ish", "combination", "mix", "blend", "range"]):
            # For color combinations/ranges, use a larger radius
            color_radius = 100
            print(f"Using adaptive radius: {color_radius} (color combination/range detected)")
        elif any(word in text_lower for word in ["light", "pale", "soft"]):
            # For light colors, use medium radius
            color_radius = 80
            print(f"Using adaptive radius: {color_radius} (light color detected)")
        elif any(word in text_lower for word in ["dark", "deep", "rich"]):
            # For dark colors, use medium radius
            color_radius = 80
            print(f"Using adaptive radius: {color_radius} (dark color detected)")
        else:
            # Default radius for exact colors
            color_radius = 60
            print(f"Using adaptive radius: {color_radius} (exact color)")
        
        # --- Find nearest cluster center to target color ---
        dists_to_centers = np.linalg.norm(cluster_centers - target_rgb, axis=1)
        nearest_cluster = np.argmin(dists_to_centers)
        print(f"Using cluster {nearest_cluster} for color pre-filtering.")
        
        # --- Use KD-Tree for that cluster ---
        if nearest_cluster < len(cluster_kdtrees) and cluster_kdtrees[nearest_cluster] is not None:
            indices = cluster_kdtrees[nearest_cluster].query_radius([target_rgb], r=color_radius)[0]
            filtered_img_ids = [cluster_img_ids[nearest_cluster][i] for i in indices]
            
            if filtered_img_ids:
                # Filter to only include images from selected categories
                filtered_img_ids = [img_id for img_id in filtered_img_ids if img_id in search_img_ids]
                if filtered_img_ids:
                    # Use a more efficient approach - search in the main index but filter results
                    labels, distances = search_index.knn_query(joint_feat, k=min(100, len(search_img_ids)))
                    top_indices = labels[0]
                    top_img_ids = [search_img_ids[idx] for idx in top_indices]
                    
                    # Filter results to only include color-matching images
                    color_filtered_results = []
                    for img_id in top_img_ids:
                        if img_id in filtered_img_ids:
                            color_filtered_results.append(img_id)
                            if len(color_filtered_results) >= 10:
                                break
                    
                    if color_filtered_results:
                        top_img_ids = color_filtered_results
                        print(f"Found {len(color_filtered_results)} images matching color criteria")
                    else:
                        print("No color matches found in top results. Using regular search.")
                        # Fall back to regular search
                        labels, distances = search_index.knn_query(joint_feat, k=10)
                        top_indices = labels[0]
                        top_img_ids = [search_img_ids[idx] for idx in top_indices]
                else:
                    print("No images found within color radius in selected categories. Using regular search.")
                    labels, distances = search_index.knn_query(joint_feat, k=10)
                    top_indices = labels[0]
                    top_img_ids = [search_img_ids[idx] for idx in top_indices]
            else:
                print("No images found within color radius in this cluster. Using regular search.")
                labels, distances = search_index.knn_query(joint_feat, k=10)
                top_indices = labels[0]
                top_img_ids = [search_img_ids[idx] for idx in top_indices]
        else:
            print("No images in this cluster. Using regular search.")
            labels, distances = search_index.knn_query(joint_feat, k=10)
            top_indices = labels[0]
            top_img_ids = [search_img_ids[idx] for idx in top_indices]
    else:
        # --- Search in HNSW index ---
        labels, distances = search_index.knn_query(joint_feat, k=10)
        top_indices = labels[0]  # shape: (10,)
        top_img_ids = [search_img_ids[idx] for idx in top_indices]

    # --- Show results ---
    print("Top 10 relevant images:")
    for i, img_id in enumerate(top_img_ids):
        category = category_mapping[img_id]
        print(f"{i+1}: {img_id} ({category})")

    # Ask for color and rerank
    print("\nColor reranking options:")
    print("1. Skip color reranking")
    print("2. Enter RGB values (e.g., 0,0,255 for blue)")
    print("3. Enter color name (e.g., blue, greenish-yellow, light-blue)")
    
    color_choice = input("Enter your choice (1-3): ").strip()
    
    if color_choice == "1":
        print("Skipping color reranking.")
        target_rgb = None
    elif color_choice == "2":
        rgb_input = input("Enter target color as R,G,B (e.g., 0,0,255 for blue): ")
        try:
            target_rgb = np.array([int(x) for x in rgb_input.split(",")])
            assert target_rgb.shape == (3,)
        except Exception as e:
            print("Invalid RGB input. Please enter three comma-separated integers.")
            continue
    elif color_choice == "3":
        color_name = input("Enter color name: ").strip()
        target_rgb = get_target_color_from_text(color_name)
        if target_rgb is None:
            print("Color not recognized. Skipping color reranking.")
            target_rgb = None
        else:
            print(f"Detected color: {target_rgb}")
    else:
        print("Invalid choice. Skipping color reranking.")
        target_rgb = None

    # Rerank
    if target_rgb is not None:
        color_distances = []
        valid_indices = []
        for idx in top_indices:
            img_id = search_img_ids[idx]
            category = category_mapping[img_id]
            if img_id in all_color_metadata[category]:
                img_rgb = np.array(all_color_metadata[category][img_id])
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
                img_id = search_img_ids[idx]
                category = category_mapping[img_id]
                print(f"{i+1}: {img_id} ({category})")
        else:
            print("No retrieved images had color metadata. Skipping color rerank.")
            print("Top 10 relevant images (no color reranking):")
            for i, idx in enumerate(top_indices):
                img_id = search_img_ids[idx]
                category = category_mapping[img_id]
                print(f"{i+1}: {img_id} ({category})")
    else:
        print("Top 10 relevant images (no color reranking):")
        for i, idx in enumerate(top_indices):
            img_id = search_img_ids[idx]
            category = category_mapping[img_id]
            print(f"{i+1}: {img_id} ({category})")
    
    # Display and save results - limit to top 10
    display_indices = top_indices[:10]  # Ensure we only display max 10 images
    num_images = len(display_indices)
    
    print(f"Displaying {num_images} images in 2x5 grid...")
    
    if num_images == 0:
        print("No images to display.")
        continue
        
    plt.figure(figsize=(20, 8))  # Adjusted height for 10 images
    for i, idx in enumerate(display_indices):
        img_id = search_img_ids[idx]
        category = category_mapping[img_id]
        for ext in [".jpg", ".jpeg", ".png"]:
            img_file = os.path.join(image_folder, f"{img_id}{ext}")
            if os.path.isfile(img_file):
                img = Image.open(img_file)
                plt.subplot(2, 5, i+1)  # Changed to 2x5 grid for 10 images
                plt.imshow(img)
                plt.title(f"Rank {i+1} ({category})")
                plt.axis('off')
                break
        save_path = os.path.join(results_dir, f"rank_{i+1}_{img_id}_{category}{ext}")
        img.save(save_path)
        print(f"Saved {save_path}")
    plt.tight_layout()
    grid_path = os.path.join(results_dir, "retrieved_grid.png")
    plt.savefig(grid_path)
    print(f"Saved grid of results to {grid_path}")
    plt.close()

