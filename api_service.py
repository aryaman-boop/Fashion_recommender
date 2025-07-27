import os
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
from data_utils import base_path
from pathlib import Path
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["http://localhost:8001", "http://127.0.0.1:8001"], supports_credentials=True)  # Enable CORS for frontend integration

# --- Settings ---
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_name = "RN50x4"
clip_finetuned_path = base_path / "/home/banik8/composed image retrieval/Fashion_recommender/clip_finetuned_on_fiq_RN50x4/saved_models/tuned_clip_best.pt"
combiner_path = base_path / "/home/banik8/composed image retrieval/Fashion_recommender/checkpoint_epoch_19.pt"
image_folder = base_path

# Available categories
available_categories = ["dress", "toptee", "shirt"]

# Global variables for loaded models and data
clip_model = None
clip_preprocess = None
combiner = None
all_embeddings = {}
all_triplets = {}
all_color_metadata = {}
all_image_ids = set()
category_mapping = {}
combined_embeddings_array = None
combined_image_ids = []
index = None
category_indices = {}  # Pre-built indices for each category
cluster_centers = None
cluster_kdtrees = []
cluster_img_ids = []

def load_models():
    """Load CLIP and Combiner models"""
    global clip_model, clip_preprocess, combiner
    
    logger.info("Loading CLIP model...")
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    clip_model.eval()
    clip_model.load_state_dict(torch.load(clip_finetuned_path, map_location=device)["CLIP"])
    
    logger.info("Loading Combiner model...")
    combiner = AttentionFusionCombiner(640, 512, 512).to(device)
    combiner.eval()
    
    checkpoint = torch.load(combiner_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    
    # Remove 'module.' prefix if it exists
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    
    combiner.load_state_dict(new_state_dict)
    logger.info("Models loaded successfully")

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

def initialize_data():
    """Initialize all data structures"""
    global all_embeddings, all_triplets, all_color_metadata, all_image_ids, category_mapping
    global combined_embeddings_array, combined_image_ids, index, cluster_centers, cluster_kdtrees, cluster_img_ids
    
    logger.info("Loading category data...")
    for category in available_categories:
        logger.info(f"Loading {category} data...")
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
                
            logger.info(f"Loaded {len(category_image_ids)} images for {category}")
        except Exception as e:
            logger.error(f"Error loading {category}: {e}")
            continue

    logger.info(f"Total unique images across all categories: {len(all_image_ids)}")

    # Build combined embeddings array
    logger.info("Building search index...")
    combined_embeddings = []
    combined_image_ids = []

    for category in available_categories:
        if category in all_embeddings:
            embeddings = all_embeddings[category]
            for img_id, embedding in embeddings.items():
                combined_embeddings.append(np.array(embedding))
                combined_image_ids.append(img_id)

    combined_embeddings_array = np.stack(combined_embeddings)
    logger.info(f"Combined embeddings shape: {combined_embeddings_array.shape}")

    # Build HNSW index for all categories
    dim = combined_embeddings_array.shape[1]
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=combined_embeddings_array.shape[0], ef_construction=200, M=16)
    index.add_items(combined_embeddings_array, np.arange(combined_embeddings_array.shape[0]))
    index.set_ef(50)

    # Pre-build indices for each category for faster filtering
    logger.info("Building category-specific indices...")
    for category in available_categories:
        category_embeddings = []
        category_img_ids = []
        
        for i, img_id in enumerate(combined_image_ids):
            if category_mapping[img_id] == category:
                category_embeddings.append(combined_embeddings_array[i])
                category_img_ids.append(img_id)
        
        if category_embeddings:
            category_embeddings_array = np.stack(category_embeddings)
            category_index = hnswlib.Index(space='cosine', dim=dim)
            category_index.init_index(max_elements=category_embeddings_array.shape[0], ef_construction=100, M=16)
            category_index.add_items(category_embeddings_array, np.arange(category_embeddings_array.shape[0]))
            category_index.set_ef(50)
            category_indices[category] = {
                'index': category_index,
                'embeddings': category_embeddings_array,
                'img_ids': category_img_ids
            }
            logger.info(f"Built index for {category}: {len(category_img_ids)} images")

    # Build KDTree for color pre-filtering
    logger.info("Building color clustering...")
    color_img_ids = []
    color_vectors = []

    for img_id in combined_image_ids:
        category = category_mapping[img_id]
        if img_id in all_color_metadata[category]:
            color_img_ids.append(img_id)
            color_vectors.append(all_color_metadata[category][img_id])

    if color_vectors:
        color_vectors = np.array(color_vectors)
        
        # Cluster colors and build per-cluster KD-Trees
        num_clusters = min(8, len(color_vectors))
        if num_clusters > 1:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(color_vectors)
            cluster_labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_
        else:
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
        cluster_centers = np.array([])
        cluster_kdtrees = []
        cluster_img_ids = []

    logger.info("Data initialization completed")

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
    import re
    
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

def perform_retrieval(image_data, modification_text, search_categories=None, target_color=None):
    """Perform the actual image retrieval"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess and encode
        ref_img_tensor = clip_preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            ref_img_feat = clip_model.encode_image(ref_img_tensor).float()
            text_tokens = clip.tokenize([modification_text]).to(device)
            text_feat = clip_model.encode_text(text_tokens).float()

        # Fuse with Combiner
        with torch.no_grad():
            joint_feat = combiner.combine_features(ref_img_feat, text_feat)
            joint_feat = joint_feat.cpu().numpy().astype(np.float32)

        # Determine search categories
        if search_categories is None:
            # Only suggest category if no explicit categories are provided
            # This allows the frontend to control category selection
            search_categories = available_categories

        # Use pre-built category indices for faster filtering
        if len(search_categories) < len(available_categories):
            if len(search_categories) == 1:
                # Single category - use pre-built index
                category = search_categories[0]
                if category in category_indices:
                    search_index = category_indices[category]['index']
                    search_img_ids = category_indices[category]['img_ids']
                else:
                    search_index = index
                    search_img_ids = combined_image_ids
            else:
                # Multiple categories - combine pre-built indices
                combined_embeddings = []
                combined_img_ids = []
                
                for category in search_categories:
                    if category in category_indices:
                        combined_embeddings.append(category_indices[category]['embeddings'])
                        combined_img_ids.extend(category_indices[category]['img_ids'])
                
                if combined_embeddings:
                    combined_embeddings_array = np.vstack(combined_embeddings)
                    # Build temporary index for combined categories
                    temp_index = hnswlib.Index(space='cosine', dim=combined_embeddings_array.shape[1])
                    temp_index.init_index(max_elements=combined_embeddings_array.shape[0], ef_construction=100, M=16)
                    temp_index.add_items(combined_embeddings_array, np.arange(combined_embeddings_array.shape[0]))
                    temp_index.set_ef(50)
                    search_index = temp_index
                    search_img_ids = combined_img_ids
                else:
                    search_index = index
                    search_img_ids = combined_image_ids
        else:
            search_index = index
            search_img_ids = combined_image_ids

        # Color pre-filtering if color is mentioned or provided
        if target_color is None:
            target_color = get_target_color_from_text(modification_text)
            
        if target_color is not None and len(cluster_centers) > 0:
            # Enhanced color filtering for combinations and ranges
            text_lower = modification_text.lower()
            
            # Determine color radius based on the type of color request
            if any(word in text_lower for word in ["ish", "ish-", "-ish", "combination", "mix", "blend", "range"]):
                # For color combinations/ranges, use a larger radius
                color_radius = 100
            elif any(word in text_lower for word in ["light", "pale", "soft"]):
                # For light colors, use medium radius
                color_radius = 80
            elif any(word in text_lower for word in ["dark", "deep", "rich"]):
                # For dark colors, use medium radius
                color_radius = 80
            else:
                # Default radius for exact colors
                color_radius = 60
            
            dists_to_centers = np.linalg.norm(cluster_centers - target_color, axis=1)
            nearest_cluster = np.argmin(dists_to_centers)
            
            if nearest_cluster < len(cluster_kdtrees) and cluster_kdtrees[nearest_cluster] is not None:
                indices = cluster_kdtrees[nearest_cluster].query_radius([target_color], r=color_radius)[0]
                filtered_img_ids = [cluster_img_ids[nearest_cluster][i] for i in indices]
                
                if filtered_img_ids:
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
                        # If no color matches found in top results, fall back to regular search
                    else:
                        labels, distances = search_index.knn_query(joint_feat, k=10)
                        top_indices = labels[0]
                        top_img_ids = [search_img_ids[idx] for idx in top_indices]
                else:
                    # Fall back to regular search
                    labels, distances = search_index.knn_query(joint_feat, k=10)
                    top_indices = labels[0]
                    top_img_ids = [search_img_ids[idx] for idx in top_indices]
            else:
                # Fall back to regular search
                labels, distances = search_index.knn_query(joint_feat, k=10)
                top_indices = labels[0]
                top_img_ids = [search_img_ids[idx] for idx in top_indices]
        else:
            # Regular search without color filtering
            labels, distances = search_index.knn_query(joint_feat, k=10)
            top_indices = labels[0]
            top_img_ids = [search_img_ids[idx] for idx in top_indices]

        # Prepare results
        results = []
        result_categories = set()
        for i, img_id in enumerate(top_img_ids):
            category = category_mapping[img_id]
            result_categories.add(category)
            
            # Try to find the image file
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                potential_path = os.path.join(image_folder, f"{img_id}{ext}")
                if os.path.isfile(potential_path):
                    img_path = potential_path
                    break
            
            # Get color metadata if available
            color_info = None
            if img_id in all_color_metadata[category]:
                color_info = all_color_metadata[category][img_id]
            
            result = {
                "rank": i + 1,
                "image_id": img_id,
                "category": category,
                "image_path": img_path,
                "color_info": color_info,
                "similarity_score": float(1 - distances[0][i]) if i < len(distances[0]) else 0.0
            }
            results.append(result)

        return results

    except Exception as e:
        logger.error(f"Error in retrieval: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "API is running"})

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint for debugging"""
    return jsonify({"status": "success", "message": "API connection working", "timestamp": "2025-07-22"})

@app.route('/retrieve', methods=['POST'])
def retrieve_images():
    """Main endpoint for image retrieval"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        image_data = data.get('image')  # Base64 encoded image
        modification_text = data.get('modification_text', '')
        search_categories = data.get('search_categories')  # Optional: list of categories
        target_color = data.get('target_color')  # Optional: [R, G, B] array
        
        if not image_data:
            return jsonify({"error": "Image data is required"}), 400
        
        if not modification_text:
            return jsonify({"error": "Modification text is required"}), 400
        
        # Convert target_color to numpy array if provided
        if target_color:
            target_color = np.array(target_color)
        
        # Perform retrieval
        results = perform_retrieval(image_data, modification_text, search_categories, target_color)
        
        return jsonify({
            "success": True,
            "results": results,
            "total_results": len(results),
            "search_categories": search_categories if search_categories else "auto-detected"
        })
        
    except Exception as e:
        logger.error(f"Error in retrieve endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get available categories"""
    return jsonify({
        "categories": available_categories,
        "total_categories": len(available_categories)
    })

@app.route('/images/<image_id>', methods=['GET'])
def serve_image(image_id):
    """Serve image files by ID"""
    try:
        # Try different file extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            image_path = os.path.join(image_folder, f"{image_id}{ext}")
            if os.path.isfile(image_path):
                from flask import send_file
                return send_file(image_path, mimetype='image/jpeg')
        
        # If no image found, return a placeholder
        logger.warning(f"Image not found for ID: {image_id}")
        return jsonify({"error": "Image not found"}), 404
        
    except Exception as e:
        logger.error(f"Error serving image {image_id}: {e}")
        return jsonify({"error": "Error serving image"}), 500

if __name__ == '__main__':
    # Initialize models and data
    logger.info("Initializing API service...")
    load_models()
    initialize_data()
    logger.info("API service initialized successfully")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5002, debug=False) 