#!/usr/bin/env python3
"""
Debug script to test shirt category filtering specifically
"""

import requests
import json
import base64
from PIL import Image
import io

# API configuration
API_BASE_URL = "http://localhost:5002"

def create_test_image():
    """Create a simple test image"""
    # Create a simple 100x100 red image
    img = Image.new('RGB', (100, 100), color='red')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{img_str}"

def test_shirt_filtering():
    """Test shirt category filtering specifically"""
    
    # Create test image
    test_image = create_test_image()
    modification_text = "blue shirt"
    
    print("🔍 Debugging Shirt Category Filtering...")
    print("=" * 60)
    
    # Test 1: All categories (should include shirt)
    print("\n1️⃣ Testing ALL categories:")
    payload = {
        "image": test_image,
        "modification_text": modification_text
    }
    
    response = requests.post(f"{API_BASE_URL}/retrieve", json=payload)
    if response.status_code == 200:
        result = response.json()
        categories = set(item['category'] for item in result['results'])
        print(f"   ✅ Found {len(result['results'])} results")
        print(f"   📊 Categories: {list(categories)}")
        print(f"   📋 All results categories:")
        for i, item in enumerate(result['results']):
            print(f"      {i+1}. {item['image_id']} - {item['category']}")
    else:
        print(f"   ❌ Error: {response.status_code}")
    
    # Test 2: Only shirt category
    print("\n2️⃣ Testing SHIRT category only:")
    payload = {
        "image": test_image,
        "modification_text": modification_text,
        "search_categories": ["shirt"]
    }
    
    response = requests.post(f"{API_BASE_URL}/retrieve", json=payload)
    if response.status_code == 200:
        result = response.json()
        categories = set(item['category'] for item in result['results'])
        print(f"   ✅ Found {len(result['results'])} results")
        print(f"   📊 Categories: {list(categories)}")
        print(f"   📋 Shirt-only results:")
        for i, item in enumerate(result['results']):
            print(f"      {i+1}. {item['image_id']} - {item['category']}")
        
        if len(categories) == 1 and 'shirt' in categories:
            print("   🎯 SUCCESS: Only shirt category returned!")
        else:
            print("   ⚠️  WARNING: Unexpected categories found!")
            print(f"   Expected: ['shirt'], Got: {list(categories)}")
    else:
        print(f"   ❌ Error: {response.status_code}")
    
    # Test 3: Check if shirt category exists in the system
    print("\n3️⃣ Checking available categories:")
    response = requests.get(f"{API_BASE_URL}/categories")
    if response.status_code == 200:
        result = response.json()
        print(f"   📊 Available categories: {result['categories']}")
        print(f"   📊 Total categories: {result['total_categories']}")
    else:
        print(f"   ❌ Error: {response.status_code}")
    
    # Test 4: Test with different query
    print("\n4️⃣ Testing with different query (no category hint):")
    payload = {
        "image": test_image,
        "modification_text": "make it blue",
        "search_categories": ["shirt"]
    }
    
    response = requests.post(f"{API_BASE_URL}/retrieve", json=payload)
    if response.status_code == 200:
        result = response.json()
        categories = set(item['category'] for item in result['results'])
        print(f"   ✅ Found {len(result['results'])} results")
        print(f"   📊 Categories: {list(categories)}")
        print(f"   📋 Results with neutral query:")
        for i, item in enumerate(result['results']):
            print(f"      {i+1}. {item['image_id']} - {item['category']}")
    else:
        print(f"   ❌ Error: {response.status_code}")
    
    print("\n" + "=" * 60)
    print("🏁 Shirt filtering debug completed!")

if __name__ == "__main__":
    test_shirt_filtering() 