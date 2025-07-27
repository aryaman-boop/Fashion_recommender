# Negative Sampling in Composed Image Retrieval (CIR) System

## 🎯 Overview
Negative sampling is a crucial technique in triplet loss training that helps the model learn to distinguish between similar and dissimilar items.

## 📊 Diagrammatic Representation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEGATIVE SAMPLING PROCESS                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ANCHOR IMAGE  │    │  POSITIVE IMAGE │    │ NEGATIVE IMAGE  │
│   (Reference)   │    │   (Target)      │    │   (Random)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLIP Encoder  │    │   CLIP Encoder  │    │   CLIP Encoder  │
│   (Image)       │    │   (Image)       │    │   (Image)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Anchor Feature │    │ Positive Feature│    │ Negative Feature│
│   Vector (640)  │    │   Vector (640)  │    │   Vector (640)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   COMBINER NETWORK      │
                    │  (AttentionFusion)      │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   JOINT FEATURES        │
                    │   (Fused Vectors)       │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   TRIPLET LOSS          │
                    │   Calculation           │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   LOSS = max(0, margin + │
                    │   d(anchor, positive) -  │
                    │   d(anchor, negative))   │
                    └─────────────────────────┘
```

## 🔄 Negative Sampling Strategies

### 1. **Simple Random Negative Sampling** (Current Implementation)
```
┌─────────────────────────────────────────────────────────────────┐
│                    DATASET (37,825 images)                      │
├─────────────────────────────────────────────────────────────────┤
│  Dress: 11,642  │  Shirt: 13,918  │  Top/Tee: 12,265           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RANDOM SELECTION                             │
│                                                                 │
│  For each triplet:                                              │
│  1. Pick any random image from entire dataset                  │
│  2. Use as negative example                                    │
│  3. No consideration of similarity or difficulty               │
└─────────────────────────────────────────────────────────────────┘
```

### 2. **Hard Negative Sampling** (Alternative Approach)
```
┌─────────────────────────────────────────────────────────────────┐
│                    SEMI-HARD NEGATIVES                          │
│                                                                 │
│  Criteria:                                                      │
│  • d(anchor, negative) > d(anchor, positive)                   │
│  • d(anchor, negative) < d(anchor, positive) + margin          │
│                                                                 │
│  This ensures negative is:                                      │
│  • Harder than positive (further away)                         │
│  • Not too hard (within margin)                                │
└─────────────────────────────────────────────────────────────────┘
```

### 3. **Category-Aware Negative Sampling** (Enhanced Approach)
```
┌─────────────────────────────────────────────────────────────────┐
│                    CATEGORY-BASED SAMPLING                      │
│                                                                 │
│  Strategy:                                                      │
│  • 70%: Same category, different item                          │
│  • 20%: Different category                                     │
│  • 10%: Random from entire dataset                             │
│                                                                 │
│  Benefits:                                                      │
│  • Teaches fine-grained discrimination                         │
│  • Maintains category awareness                                 │
│  • Provides diverse negative examples                          │
└─────────────────────────────────────────────────────────────────┘
```

## 📈 Training Flow with Negative Sampling

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING EPOCH                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   BATCH 1   │    │   BATCH 2   │    │   BATCH N   │
│             │    │             │    │             │
│ Anchor: A1  │    │ Anchor: A2  │    │ Anchor: AN  │
│ Pos: P1     │    │ Pos: P2     │    │ Pos: PN     │
│ Neg: N1     │    │ Neg: N2     │    │ Neg: NN     │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  TRIPLET 1  │    │  TRIPLET 2  │    │  TRIPLET N  │
│   Loss: L1  │    │   Loss: L2  │    │   Loss: LN  │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
                ┌─────────────────────┐
                │   BATCH LOSS        │
                │   L = (L1+L2+...+LN)/N │
                └─────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   BACKPROPAGATION   │
                │   Update Weights    │
                └─────────────────────┘
```

## 🎯 Current Implementation Details

### Code Location: `traintest2.py`
```python
# Current Simple Random Negative Sampling
neg_idx = np.random.randint(0, len(relative_train_dataset))
neg_ref_img, neg_tgt_img, _ = relative_train_dataset[neg_idx]
batch_hard_negatives.append(neg_tgt_img.unsqueeze(0))
```

### Process Flow:
1. **For each positive pair** (anchor + positive):
   - Randomly select any image from dataset
   - Use as negative example
   - No filtering or difficulty consideration

2. **Triplet Formation**:
   - Anchor: Reference image
   - Positive: Target image (from same triplet)
   - Negative: Random image (from any triplet)

3. **Loss Calculation**:
   ```
   loss = max(0, margin + d(anchor, positive) - d(anchor, negative))
   ```

## 🔧 Potential Improvements

### 1. **Hard Negative Mining**
```python
# Find negatives that are harder than positives
for anchor, positive in positive_pairs:
    # Get top-k similar images to anchor
    similarities = compute_similarity(anchor, all_images)
    hard_negatives = []
    
    for candidate in top_k_similar:
        if d(anchor, candidate) > d(anchor, positive):
            if d(anchor, candidate) < d(anchor, positive) + margin:
                hard_negatives.append(candidate)
```

### 2. **Category-Balanced Sampling**
```python
# Ensure diverse negative examples
def sample_negative(anchor_category):
    if random.random() < 0.7:
        # Same category, different item
        return sample_from_category(anchor_category)
    elif random.random() < 0.9:
        # Different category
        return sample_from_different_category(anchor_category)
    else:
        # Random from all
        return sample_random()
```

### 3. **Online Hard Negative Mining**
```python
# During training, dynamically select hardest negatives
def mine_hard_negatives(anchor, positive, candidates, k=10):
    distances = []
    for candidate in candidates:
        dist = compute_distance(anchor, candidate)
        distances.append((dist, candidate))
    
    # Sort by distance (hardest first)
    distances.sort(reverse=True)
    return [candidate for _, candidate in distances[:k]]
```

## 📊 Performance Impact

### Current Simple Random:
- ✅ **Fast**: O(1) sampling time
- ✅ **Simple**: Easy to implement
- ❌ **Inefficient**: May select easy negatives
- ❌ **Slow convergence**: Less informative training

### Enhanced Approaches:
- ✅ **Better learning**: Harder negatives
- ✅ **Faster convergence**: More informative
- ❌ **Computational cost**: More expensive sampling
- ❌ **Complexity**: Harder to implement

## 🎯 Summary

The current system uses **Simple Random Negative Sampling**, which:
1. Randomly selects any image as negative
2. Is computationally efficient
3. Provides basic training signal
4. Could be improved with harder negative mining

**Recommendation**: For better performance, consider implementing **Hard Negative Mining** or **Category-Aware Sampling** in future iterations. 