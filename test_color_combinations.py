#!/usr/bin/env python3
"""
Test script to demonstrate enhanced color combination functionality
"""

import requests
import base64
from PIL import Image
import io
import json

# API endpoint
API_BASE_URL = "http://localhost:5001"

def encode_image_to_base64(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_color_combinations():
    """Test various color combination queries"""
    
    # Test queries with different color combinations
    test_queries = [
        "make it greenish-yellow",
        "change to blue-green",
        "I want it light-blue",
        "make it dark-red",
        "change to orange-yellow",
        "I want a pink-red color",
        "make it turquoise",
        "change to lavender",
        "I want it coral colored",
        "make it emerald green",
        "change to navy-blue",
        "I want it sky-blue",
        "make it hot-pink",
        "change to forest-green",
        "I want it burgundy",
        "make it cream colored",
        "change to khaki",
        "I want it salmon",
        "make it mint green",
        "change to rose colored"
    ]
    
    print("🎨 Testing Enhanced Color Combination Functionality")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing: '{query}'")
        
        # Create a simple test request (you can replace with actual image)
        test_data = {
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=",  # Minimal valid JPEG
            "modification_text": query,
            "search_categories": ["dress", "shirt", "toptee"]
        }
        
        try:
            response = requests.post(f"{API_BASE_URL}/retrieve", json=test_data, timeout=30)
            
            if response.status_code == 200:
                response_data = response.json()
                results = response_data.get('results', [])
                print(f"   ✅ Success! Found {len(results)} results")
                
                # Show first few results
                for j, result in enumerate(results[:3], 1):
                    print(f"      {j}. {result['image_id']} ({result['category']}) - Score: {result['similarity_score']:.3f}")
                    if result.get('color_info'):
                        print(f"         Color: {result['color_info']}")
            else:
                print(f"   ❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Color Combination Testing Complete!")
    print("\n💡 Supported Color Patterns:")
    print("   • Exact colors: 'blue', 'red', 'green'")
    print("   • Color combinations: 'green-yellow', 'blue-green'")
    print("   • Color modifiers: 'greenish-yellow', 'light-blue', 'dark-red'")
    print("   • Named colors: 'turquoise', 'lavender', 'coral', 'emerald'")
    print("   • Fashion colors: 'navy-blue', 'hot-pink', 'forest-green'")

if __name__ == "__main__":
    test_color_combinations() 