import requests
import base64
import json
from PIL import Image
import io

# API Configuration
API_BASE_URL = "http://localhost:5002"

def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"

def retrieve_images(image_path, modification_text, search_categories=None, target_color=None):
    """
    Call the image retrieval API
    
    Args:
        image_path (str): Path to the reference image file
        modification_text (str): Text describing the desired modification
        search_categories (list, optional): List of categories to search in (e.g., ["dress", "shirt"])
        target_color (list, optional): Target color as [R, G, B] array (e.g., [0, 0, 255] for blue)
    
    Returns:
        dict: API response with retrieval results
    """
    
    # Encode image to base64
    image_data = encode_image_to_base64(image_path)
    
    # Prepare request payload
    payload = {
        "image": image_data,
        "modification_text": modification_text
    }
    
    if search_categories:
        payload["search_categories"] = search_categories
    
    if target_color:
        payload["target_color"] = target_color
    
    # Make API request
    try:
        response = requests.post(
            f"{API_BASE_URL}/retrieve",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def get_available_categories():
    """Get list of available categories"""
    try:
        response = requests.get(f"{API_BASE_URL}/categories")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting categories: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False

# Example usage
if __name__ == "__main__":
    # Check if API is running
    if not check_api_health():
        print("API is not running. Please start the API service first.")
        exit(1)
    
    print("API is running!")
    
    # Get available categories
    categories = get_available_categories()
    if categories:
        print(f"Available categories: {categories['categories']}")
    
    # Example 1: Basic retrieval
    print("\n--- Example 1: Basic Retrieval ---")
    image_path = "B002ZG8I60.jpg"  # Replace with actual image path
    modification_text = "make it blue"
    
    result = retrieve_images(image_path, modification_text)
    if result and result.get("success"):
        print(f"Found {result['total_results']} results")
        for item in result['results'][:5]:  # Show first 5 results
            print(f"Rank {item['rank']}: {item['image_id']} ({item['category']})")
    
    # Example 2: Category-specific search
    print("\n--- Example 2: Category-Specific Search ---")
    result = retrieve_images(
        image_path, 
        modification_text, 
        search_categories=["dress", "shirt"]
    )
    if result and result.get("success"):
        print(f"Found {result['total_results']} results in dress and shirt categories")
    
    # Example 3: With target color
    print("\n--- Example 3: With Target Color ---")
    result = retrieve_images(
        image_path, 
        modification_text, 
        target_color=[0, 0, 255]  # Blue
    )
    if result and result.get("success"):
        print(f"Found {result['total_results']} results with blue color preference")

# Frontend Integration Example (JavaScript)
"""
// JavaScript example for frontend integration

async function retrieveImages(imageFile, modificationText, searchCategories = null, targetColor = null) {
    // Convert image file to base64
    const base64Image = await fileToBase64(imageFile);
    
    // Prepare request payload
    const payload = {
        image: base64Image,
        modification_text: modificationText
    };
    
    if (searchCategories) {
        payload.search_categories = searchCategories;
    }
    
    if (targetColor) {
        payload.target_color = targetColor;
    }
    
    try {
        const response = await fetch('http://localhost:5000/retrieve', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });
        
        if (response.ok) {
            const result = await response.json();
            return result;
        } else {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
    });
}

// Usage example:
// const imageFile = document.getElementById('imageInput').files[0];
// const modificationText = document.getElementById('textInput').value;
// const results = await retrieveImages(imageFile, modificationText, ['dress', 'shirt']);
""" 