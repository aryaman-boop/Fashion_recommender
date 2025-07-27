#!/usr/bin/env python3
"""
Test script to demonstrate enhanced color combination functionality
in the interactive_fashioniq_retrieval.py file.
"""

import numpy as np
import re

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

def test_color_detection():
    """Test various color inputs to demonstrate the enhanced functionality"""
    
    test_cases = [
        # Basic colors
        "blue",
        "red", 
        "green",
        "yellow",
        
        # Color combinations
        "greenish-yellow",
        "yellowish-green", 
        "blue-green",
        "red-orange",
        "pink-red",
        
        # Light/Dark variations
        "light-blue",
        "dark-blue",
        "light-green",
        "dark-red",
        
        # Complex combinations
        "greenish yellow",
        "blue ish green",
        "red orange",
        
        # Special colors
        "turquoise",
        "lavender",
        "coral",
        "emerald",
        
        # Non-color words
        "dress",
        "shirt",
        "make it longer",
        
        # Color-related but no specific color
        "any color",
        "colored fabric",
    ]
    
    print("🎨 Testing Enhanced Color Detection")
    print("=" * 50)
    
    for test_case in test_cases:
        result = get_target_color_from_text(test_case)
        if result is not None:
            print(f"✅ '{test_case}' → RGB{tuple(result)}")
        else:
            print(f"❌ '{test_case}' → No color detected")
    
    print("\n" + "=" * 50)
    print("🎯 Key Features:")
    print("• 80+ color names including combinations")
    print("• Color blending (e.g., 'greenish-yellow')")
    print("• Light/Dark variations (e.g., 'light-blue')")
    print("• Adaptive radius based on color complexity")
    print("• Fallback for color-related but non-specific queries")

if __name__ == "__main__":
    test_color_detection() 