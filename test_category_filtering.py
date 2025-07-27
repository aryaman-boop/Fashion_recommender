#!/usr/bin/env python3
"""
Test script to verify category filtering in the API
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

def test_category_filtering():
    """Test category filtering with different categories"""
    
    # Create test image
    test_image = create_test_image()
    modification_text = "make it blue"
    
    print("🧪 Testing Category Filtering...")
    print("=" * 50)
    
    # Test 1: All categories
    print("\n1️⃣ Testing ALL categories:")
    payload = {
        "image": test_image,
        "modification_text": modification_text
        # No search_categories = all categories
    }
    
    response = requests.post(f"{API_BASE_URL}/retrieve", json=payload)
    if response.status_code == 200:
        result = response.json()
        categories = set(item['category'] for item in result['results'])
        print(f"   ✅ Found {len(result['results'])} results")
        print(f"   📊 Categories: {list(categories)}")
    else:
        print(f"   ❌ Error: {response.status_code}")
    
    # Test 2: Only dress category
    print("\n2️⃣ Testing DRESS category only:")
    payload = {
        "image": test_image,
        "modification_text": modification_text,
        "search_categories": ["dress"]
    }
    
    response = requests.post(f"{API_BASE_URL}/retrieve", json=payload)
    if response.status_code == 200:
        result = response.json()
        categories = set(item['category'] for item in result['results'])
        print(f"   ✅ Found {len(result['results'])} results")
        print(f"   📊 Categories: {list(categories)}")
        if len(categories) == 1 and 'dress' in categories:
            print("   🎯 SUCCESS: Only dress category returned!")
        else:
            print("   ⚠️  WARNING: Unexpected categories found")
    else:
        print(f"   ❌ Error: {response.status_code}")
    
    # Test 3: Only shirt category
    print("\n3️⃣ Testing SHIRT category only:")
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
        if len(categories) == 1 and 'shirt' in categories:
            print("   🎯 SUCCESS: Only shirt category returned!")
        else:
            print("   ⚠️  WARNING: Unexpected categories found")
    else:
        print(f"   ❌ Error: {response.status_code}")
    
    # Test 4: Only toptee category
    print("\n4️⃣ Testing TOP/TEE category only:")
    payload = {
        "image": test_image,
        "modification_text": modification_text,
        "search_categories": ["toptee"]
    }
    
    response = requests.post(f"{API_BASE_URL}/retrieve", json=payload)
    if response.status_code == 200:
        result = response.json()
        categories = set(item['category'] for item in result['results'])
        print(f"   ✅ Found {len(result['results'])} results")
        print(f"   📊 Categories: {list(categories)}")
        if len(categories) == 1 and 'toptee' in categories:
            print("   🎯 SUCCESS: Only toptee category returned!")
        else:
            print("   ⚠️  WARNING: Unexpected categories found")
    else:
        print(f"   ❌ Error: {response.status_code}")
    
    # Test 5: Multiple categories (dress + shirt)
    print("\n5️⃣ Testing MULTIPLE categories (dress + shirt):")
    payload = {
        "image": test_image,
        "modification_text": modification_text,
        "search_categories": ["dress", "shirt"]
    }
    
    response = requests.post(f"{API_BASE_URL}/retrieve", json=payload)
    if response.status_code == 200:
        result = response.json()
        categories = set(item['category'] for item in result['results'])
        print(f"   ✅ Found {len(result['results'])} results")
        print(f"   📊 Categories: {list(categories)}")
        if len(categories) == 2 and 'dress' in categories and 'shirt' in categories:
            print("   🎯 SUCCESS: Only dress and shirt categories returned!")
        else:
            print("   ⚠️  WARNING: Unexpected categories found")
    else:
        print(f"   ❌ Error: {response.status_code}")
    
    print("\n" + "=" * 50)
    print("🏁 Category filtering test completed!")

if __name__ == "__main__":
    test_category_filtering() 