#!/usr/bin/env python3
"""
Test script to demonstrate enhanced text processing for clothing attributes
"""

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

def test_enhanced_text():
    """Test the enhanced text processing with various queries"""
    
    test_queries = [
        "make it long sleeved",
        "change to short sleeve",
        "long-sleeved blue shirt",
        "sleeveless dress",
        "v-neck top",
        "crew neck t-shirt",
        "loose fitting shirt",
        "tight fitted dress",
        "casual everyday wear",
        "formal business shirt",
        "blue long sleeve shirt with v-neck",
        "red short sleeved casual top"
    ]
    
    print("🧪 Testing Enhanced Text Processing...")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        enhanced = enhance_modification_text(query)
        print(f"\n{i:2d}. Original: '{query}'")
        print(f"    Enhanced: '{enhanced}'")
        
        if enhanced != query:
            print("    ✓ Enhanced with additional details")
        else:
            print("    - No enhancement needed")
    
    print("\n" + "=" * 60)
    print("🏁 Enhanced text processing test completed!")

if __name__ == "__main__":
    test_enhanced_text() 