#!/usr/bin/env python3
"""
Create a beautiful PNG diagram of negative sampling in CIR system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set style for beautiful diagrams
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

def create_negative_sampling_diagram():
    """Create a beautiful diagram of negative sampling process"""
    
    # Create figure with high DPI for crisp output
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors
    colors = {
        'blue': '#1f77b4',
        'orange': '#ff7f0e',
        'green': '#2ca02c',
        'red': '#d62728',
        'purple': '#9467bd',
        'brown': '#8c564b',
        'pink': '#e377c2',
        'gray': '#7f7f7f',
        'olive': '#bcbd22',
        'cyan': '#17becf'
    }
    
    # Title
    ax.text(8, 11.5, 'Negative Sampling in Composed Image Retrieval (CIR)', 
            fontsize=20, fontweight='bold', ha='center', color='#2c3e50')
    
    # Dataset Section
    dataset_box = FancyBboxPatch((0.5, 9.5), 15, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['blue'], alpha=0.3, 
                                edgecolor=colors['blue'], linewidth=2)
    ax.add_patch(dataset_box)
    ax.text(8, 10.25, 'DATASET (37,825 images)', fontsize=14, fontweight='bold', 
            ha='center', color='#2c3e50')
    
    # Category boxes
    dress_box = FancyBboxPatch((1, 9.7), 4, 0.8, 
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['pink'], alpha=0.7, 
                              edgecolor=colors['pink'], linewidth=1.5)
    ax.add_patch(dress_box)
    ax.text(3, 10.1, 'DRESS\n11,642 images', fontsize=10, fontweight='bold', 
            ha='center', va='center', color='white')
    
    shirt_box = FancyBboxPatch((6, 9.7), 4, 0.8, 
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['green'], alpha=0.7, 
                              edgecolor=colors['green'], linewidth=1.5)
    ax.add_patch(shirt_box)
    ax.text(8, 10.1, 'SHIRT\n13,918 images', fontsize=10, fontweight='bold', 
            ha='center', va='center', color='white')
    
    toptee_box = FancyBboxPatch((11, 9.7), 4, 0.8, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['orange'], alpha=0.7, 
                               edgecolor=colors['orange'], linewidth=1.5)
    ax.add_patch(toptee_box)
    ax.text(13, 10.1, 'TOP/TEE\n12,265 images', fontsize=10, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Triplet Formation Section
    ax.text(8, 8.8, 'TRIPLET FORMATION', fontsize=16, fontweight='bold', 
            ha='center', color='#2c3e50')
    
    # Anchor, Positive, Negative boxes
    anchor_box = FancyBboxPatch((1, 7), 3.5, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['purple'], alpha=0.8, 
                               edgecolor=colors['purple'], linewidth=2)
    ax.add_patch(anchor_box)
    ax.text(2.75, 7.6, 'ANCHOR\n(Reference)', fontsize=11, fontweight='bold', 
            ha='center', va='center', color='white')
    
    positive_box = FancyBboxPatch((6.25, 7), 3.5, 1.2, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['green'], alpha=0.8, 
                                 edgecolor=colors['green'], linewidth=2)
    ax.add_patch(positive_box)
    ax.text(8, 7.6, 'POSITIVE\n(Target)', fontsize=11, fontweight='bold', 
            ha='center', va='center', color='white')
    
    negative_box = FancyBboxPatch((11.5, 7), 3.5, 1.2, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['red'], alpha=0.8, 
                                 edgecolor=colors['red'], linewidth=2)
    ax.add_patch(negative_box)
    ax.text(13.25, 7.6, 'NEGATIVE\n(Random)', fontsize=11, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # CLIP Encoders
    anchor_clip = FancyBboxPatch((1, 5.5), 3.5, 0.8, 
                                boxstyle="round,pad=0.05", 
                                facecolor=colors['cyan'], alpha=0.7, 
                                edgecolor=colors['cyan'], linewidth=1.5)
    ax.add_patch(anchor_clip)
    ax.text(2.75, 5.9, 'CLIP Encoder', fontsize=9, fontweight='bold', 
            ha='center', va='center', color='white')
    
    positive_clip = FancyBboxPatch((6.25, 5.5), 3.5, 0.8, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor=colors['cyan'], alpha=0.7, 
                                  edgecolor=colors['cyan'], linewidth=1.5)
    ax.add_patch(positive_clip)
    ax.text(8, 5.9, 'CLIP Encoder', fontsize=9, fontweight='bold', 
            ha='center', va='center', color='white')
    
    negative_clip = FancyBboxPatch((11.5, 5.5), 3.5, 0.8, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor=colors['cyan'], alpha=0.7, 
                                  edgecolor=colors['cyan'], linewidth=1.5)
    ax.add_patch(negative_clip)
    ax.text(13.25, 5.9, 'CLIP Encoder', fontsize=9, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Feature Vectors
    anchor_feat = FancyBboxPatch((1, 4.5), 3.5, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor=colors['gray'], alpha=0.6, 
                                edgecolor=colors['gray'], linewidth=1)
    ax.add_patch(anchor_feat)
    ax.text(2.75, 4.8, '640-dim Feature', fontsize=8, fontweight='bold', 
            ha='center', va='center', color='white')
    
    positive_feat = FancyBboxPatch((6.25, 4.5), 3.5, 0.6, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor=colors['gray'], alpha=0.6, 
                                  edgecolor=colors['gray'], linewidth=1)
    ax.add_patch(positive_feat)
    ax.text(8, 4.8, '640-dim Feature', fontsize=8, fontweight='bold', 
            ha='center', va='center', color='white')
    
    negative_feat = FancyBboxPatch((11.5, 4.5), 3.5, 0.6, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor=colors['gray'], alpha=0.6, 
                                  edgecolor=colors['gray'], linewidth=1)
    ax.add_patch(negative_feat)
    ax.text(13.25, 4.8, '640-dim Feature', fontsize=8, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Combiner Network
    combiner_box = FancyBboxPatch((5, 3), 6, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['brown'], alpha=0.8, 
                                 edgecolor=colors['brown'], linewidth=2)
    ax.add_patch(combiner_box)
    ax.text(8, 3.5, 'COMBINER NETWORK\n(AttentionFusion)', fontsize=12, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Joint Features
    joint_box = FancyBboxPatch((5, 1.8), 6, 0.6, 
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['olive'], alpha=0.7, 
                              edgecolor=colors['olive'], linewidth=1.5)
    ax.add_patch(joint_box)
    ax.text(8, 2.1, 'Joint Feature Vector (640-dim)', fontsize=10, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Triplet Loss
    loss_box = FancyBboxPatch((4, 0.5), 8, 0.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['red'], alpha=0.8, 
                             edgecolor=colors['red'], linewidth=2)
    ax.add_patch(loss_box)
    ax.text(8, 0.9, 'TRIPLET LOSS = max(0, margin + d(anchor, positive) - d(anchor, negative))', 
            fontsize=11, fontweight='bold', ha='center', va='center', color='white')
    
    # Arrows
    # Dataset to Triplet
    arrow1 = ConnectionPatch((8, 9.5), (8, 8.2), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=20, fc=colors['blue'], ec=colors['blue'], linewidth=2)
    ax.add_patch(arrow1)
    
    # Triplet to CLIP
    arrow2 = ConnectionPatch((2.75, 7), (2.75, 6.3), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=15, fc=colors['purple'], ec=colors['purple'], linewidth=1.5)
    ax.add_patch(arrow2)
    
    arrow3 = ConnectionPatch((8, 7), (8, 6.3), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=15, fc=colors['green'], ec=colors['green'], linewidth=1.5)
    ax.add_patch(arrow3)
    
    arrow4 = ConnectionPatch((13.25, 7), (13.25, 6.3), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=15, fc=colors['red'], ec=colors['red'], linewidth=1.5)
    ax.add_patch(arrow4)
    
    # CLIP to Features
    arrow5 = ConnectionPatch((2.75, 5.5), (2.75, 5.1), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=15, fc=colors['cyan'], ec=colors['cyan'], linewidth=1.5)
    ax.add_patch(arrow5)
    
    arrow6 = ConnectionPatch((8, 5.5), (8, 5.1), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=15, fc=colors['cyan'], ec=colors['cyan'], linewidth=1.5)
    ax.add_patch(arrow6)
    
    arrow7 = ConnectionPatch((13.25, 5.5), (13.25, 5.1), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=15, fc=colors['cyan'], ec=colors['cyan'], linewidth=1.5)
    ax.add_patch(arrow7)
    
    # Features to Combiner
    arrow8 = ConnectionPatch((4.5, 4.8), (5, 3.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=15, fc=colors['purple'], ec=colors['purple'], linewidth=1.5)
    ax.add_patch(arrow8)
    
    arrow9 = ConnectionPatch((11.5, 4.8), (11, 3.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=15, fc=colors['red'], ec=colors['red'], linewidth=1.5)
    ax.add_patch(arrow9)
    
    # Combiner to Joint Features
    arrow10 = ConnectionPatch((8, 3), (8, 2.4), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=15, fc=colors['brown'], ec=colors['brown'], linewidth=1.5)
    ax.add_patch(arrow10)
    
    # Joint Features to Loss
    arrow11 = ConnectionPatch((8, 1.8), (8, 1.3), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=15, fc=colors['olive'], ec=colors['olive'], linewidth=1.5)
    ax.add_patch(arrow11)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['blue'], alpha=0.3, label='Dataset'),
        patches.Patch(color=colors['purple'], alpha=0.8, label='Anchor'),
        patches.Patch(color=colors['green'], alpha=0.8, label='Positive'),
        patches.Patch(color=colors['red'], alpha=0.8, label='Negative'),
        patches.Patch(color=colors['cyan'], alpha=0.7, label='CLIP Encoder'),
        patches.Patch(color=colors['brown'], alpha=0.8, label='Combiner Network'),
        patches.Patch(color=colors['olive'], alpha=0.7, label='Joint Features'),
        patches.Patch(color=colors['red'], alpha=0.8, label='Triplet Loss')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98),
             fontsize=9, framealpha=0.9)
    
    # Add current implementation note
    note_box = FancyBboxPatch((0.5, 0.1), 15, 0.3, 
                             boxstyle="round,pad=0.05", 
                             facecolor='lightyellow', alpha=0.8, 
                             edgecolor='orange', linewidth=1)
    ax.add_patch(note_box)
    ax.text(8, 0.25, 'Current Implementation: Simple Random Negative Sampling', 
            fontsize=10, fontweight='bold', ha='center', va='center', color='#2c3e50')
    
    plt.tight_layout()
    return fig

def create_sampling_strategies_diagram():
    """Create a diagram showing different negative sampling strategies"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8), dpi=300)
    
    colors = {
        'blue': '#1f77b4',
        'orange': '#ff7f0e',
        'green': '#2ca02c',
        'red': '#d62728',
        'purple': '#9467bd',
        'brown': '#8c564b',
        'pink': '#e377c2',
        'gray': '#7f7f7f',
        'olive': '#bcbd22',
        'cyan': '#17becf'
    }
    
    # Strategy 1: Simple Random
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    ax1.text(5, 9.5, 'Simple Random Sampling', fontsize=14, fontweight='bold', ha='center')
    
    # Dataset representation
    dataset_circles = []
    for i in range(20):
        x = np.random.uniform(1, 9)
        y = np.random.uniform(6, 8.5)
        color = np.random.choice([colors['blue'], colors['green'], colors['orange']])
        circle = plt.Circle((x, y), 0.2, color=color, alpha=0.6)
        ax1.add_patch(circle)
        dataset_circles.append((x, y))
    
    # Random selection arrow
    selected_idx = np.random.randint(0, len(dataset_circles))
    selected_x, selected_y = dataset_circles[selected_idx]
    ax1.plot([5, selected_x], [5, selected_y], 'r->', linewidth=3, markersize=10)
    ax1.add_patch(plt.Circle((selected_x, selected_y), 0.3, color='red', fill=False, linewidth=3))
    
    ax1.text(5, 4.5, 'Random Selection\n(No filtering)', fontsize=10, ha='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    
    # Strategy 2: Hard Negative
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    ax2.text(5, 9.5, 'Hard Negative Mining', fontsize=14, fontweight='bold', ha='center')
    
    # Anchor and positive
    ax2.add_patch(plt.Circle((3, 6), 0.3, color=colors['purple'], alpha=0.8))
    ax2.text(3, 6, 'A', fontsize=12, fontweight='bold', ha='center', va='center', color='white')
    
    ax2.add_patch(plt.Circle((4, 6), 0.3, color=colors['green'], alpha=0.8))
    ax2.text(4, 6, 'P', fontsize=12, fontweight='bold', ha='center', va='center', color='white')
    
    # Hard negatives (closer to anchor than positive)
    hard_negatives = [(6, 6.5), (7, 5.5), (5.5, 7), (6.5, 6)]
    for x, y in hard_negatives:
        ax2.add_patch(plt.Circle((x, y), 0.2, color=colors['red'], alpha=0.6))
        ax2.text(x, y, 'N', fontsize=10, fontweight='bold', ha='center', va='center', color='white')
    
    # Margin circle
    margin_circle = plt.Circle((3, 6), 2, color='red', fill=False, linestyle='--', linewidth=2)
    ax2.add_patch(margin_circle)
    
    ax2.text(5, 4.5, 'Hard Negatives\n(Within margin)', fontsize=10, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Strategy 3: Category-Aware
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    ax3.text(5, 9.5, 'Category-Aware Sampling', fontsize=14, fontweight='bold', ha='center')
    
    # Categories
    ax3.add_patch(plt.Rectangle((1, 7), 2, 1.5, color=colors['pink'], alpha=0.6))
    ax3.text(2, 7.75, 'Dress', fontsize=10, fontweight='bold', ha='center', va='center')
    
    ax3.add_patch(plt.Rectangle((4, 7), 2, 1.5, color=colors['green'], alpha=0.6))
    ax3.text(5, 7.75, 'Shirt', fontsize=10, fontweight='bold', ha='center', va='center')
    
    ax3.add_patch(plt.Rectangle((7, 7), 2, 1.5, color=colors['orange'], alpha=0.6))
    ax3.text(8, 7.75, 'Top/Tee', fontsize=10, fontweight='bold', ha='center', va='center')
    
    # Sampling percentages
    ax3.text(2, 5.5, '70%\nSame Category', fontsize=9, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['pink'], alpha=0.7))
    
    ax3.text(5, 5.5, '20%\nDifferent\nCategory', fontsize=9, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['green'], alpha=0.7))
    
    ax3.text(8, 5.5, '10%\nRandom', fontsize=9, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['orange'], alpha=0.7))
    
    ax3.text(5, 3.5, 'Balanced Strategy', fontsize=10, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Creating negative sampling diagrams...")
    
    # Create main diagram
    fig1 = create_negative_sampling_diagram()
    fig1.savefig('negative_sampling_process.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ Saved: negative_sampling_process.png")
    
    # Create strategies diagram
    fig2 = create_sampling_strategies_diagram()
    fig2.savefig('negative_sampling_strategies.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ Saved: negative_sampling_strategies.png")
    
    plt.show()
    print("🎉 Diagrams created successfully!") 