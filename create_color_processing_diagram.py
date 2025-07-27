#!/usr/bin/env python3
"""
Create a beautiful PNG diagram of color pre-processing and searching in CIR system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# Set style for beautiful diagrams
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

def create_color_processing_diagram():
    """Create a beautiful diagram of color processing and searching pipeline"""
    
    # Create figure with high DPI for crisp output
    fig, ax = plt.subplots(1, 1, figsize=(20, 14), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
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
        'cyan': '#17becf',
        'yellow': '#f7dc6f',
        'lime': '#7dce94',
        'coral': '#f1948a',
        'lavender': '#bb8fce'
    }
    
    # Title
    ax.text(10, 13.5, 'Color Pre-processing and Searching Pipeline in CIR System', 
            fontsize=22, fontweight='bold', ha='center', color='#2c3e50')
    
    # Phase 1: Image Collection
    ax.text(10, 12.8, 'PHASE 1: IMAGE COLLECTION & COLOR EXTRACTION', 
            fontsize=16, fontweight='bold', ha='center', color='#2c3e50')
    
    # Dataset representation
    dataset_box = FancyBboxPatch((0.5, 11.5), 19, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['blue'], alpha=0.3, 
                                edgecolor=colors['blue'], linewidth=2)
    ax.add_patch(dataset_box)
    ax.text(10, 12, 'FASHION DATASET (37,825 images)', fontsize=14, fontweight='bold', 
            ha='center', color='#2c3e50')
    
    # Sample images with color patches
    sample_positions = [(2, 11.7), (5, 11.7), (8, 11.7), (11, 11.7), (14, 11.7), (17, 11.7)]
    sample_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    
    for i, (pos, color_name) in enumerate(zip(sample_positions, sample_colors)):
        # Image frame
        img_frame = FancyBboxPatch((pos[0]-0.3, pos[1]-0.2), 0.6, 0.4, 
                                  boxstyle="round,pad=0.02", 
                                  facecolor='white', alpha=0.9, 
                                  edgecolor='black', linewidth=1)
        ax.add_patch(img_frame)
        
        # Color patch inside image
        color_patch = FancyBboxPatch((pos[0]-0.25, pos[1]-0.15), 0.5, 0.3, 
                                    boxstyle="round,pad=0.01", 
                                    facecolor=colors[color_name], alpha=0.8, 
                                    edgecolor='none')
        ax.add_patch(color_patch)
        
        # Image number
        ax.text(pos[0], pos[1]-0.3, f'Img {i+1}', fontsize=8, ha='center', fontweight='bold')
    
    # Phase 2: Color Extraction
    ax.text(10, 10.8, 'PHASE 2: COLOR EXTRACTION & CLUSTERING', 
            fontsize=16, fontweight='bold', ha='center', color='#2c3e50')
    
    # K-Means Clustering
    clustering_box = FancyBboxPatch((1, 9.5), 6, 1, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['purple'], alpha=0.8, 
                                   edgecolor=colors['purple'], linewidth=2)
    ax.add_patch(clustering_box)
    ax.text(4, 10, 'K-MEANS CLUSTERING\n(Extract dominant colors)', fontsize=11, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Color clusters visualization
    cluster_positions = [(9, 10), (11, 10), (13, 10), (15, 10), (17, 10)]
    cluster_colors = ['red', 'blue', 'green', 'yellow', 'purple']
    
    for pos, color_name in zip(cluster_positions, cluster_colors):
        # Cluster circle
        cluster_circle = Circle(pos, 0.4, color=colors[color_name], alpha=0.7, linewidth=2)
        ax.add_patch(cluster_circle)
        ax.text(pos[0], pos[1], color_name.upper(), fontsize=9, fontweight='bold', 
                ha='center', va='center', color='white')
    
    # Phase 3: Color Metadata Storage
    ax.text(10, 8.8, 'PHASE 3: COLOR METADATA STORAGE', 
            fontsize=16, fontweight='bold', ha='center', color='#2c3e50')
    
    # Color metadata structure
    metadata_box = FancyBboxPatch((1, 7.5), 18, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['green'], alpha=0.6, 
                                 edgecolor=colors['green'], linewidth=2)
    ax.add_patch(metadata_box)
    
    # Metadata fields
    fields = [
        ('Image ID', 3, 8),
        ('RGB Values', 6, 8),
        ('HSV Values', 9, 8),
        ('Dominant Color', 12, 8),
        ('Color Cluster', 15, 8)
    ]
    
    for field_name, x, y in fields:
        field_box = FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor='white', alpha=0.9, 
                                  edgecolor='black', linewidth=1)
        ax.add_patch(field_box)
        ax.text(x, y, field_name, fontsize=9, fontweight='bold', 
                ha='center', va='center', color='#2c3e50')
    
    # Phase 4: KD-Tree Construction
    ax.text(10, 6.8, 'PHASE 4: SPATIAL INDEXING (KD-TREE)', 
            fontsize=16, fontweight='bold', ha='center', color='#2c3e50')
    
    # KD-Tree visualization
    kdtree_box = FancyBboxPatch((1, 5.5), 6, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['orange'], alpha=0.8, 
                               edgecolor=colors['orange'], linewidth=2)
    ax.add_patch(kdtree_box)
    ax.text(4, 6, 'KD-TREE CONSTRUCTION\n(Fast color search)', fontsize=11, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Tree structure visualization
    tree_positions = [
        (10, 6),      # Root
        (8, 5.2), (12, 5.2),  # Level 1
        (7, 4.4), (9, 4.4), (11, 4.4), (13, 4.4)  # Level 2
    ]
    
    for i, pos in enumerate(tree_positions):
        node_circle = Circle(pos, 0.2, color=colors['cyan'], alpha=0.7, linewidth=2)
        ax.add_patch(node_circle)
        ax.text(pos[0], pos[1], f'N{i+1}', fontsize=8, fontweight='bold', 
                ha='center', va='center', color='white')
    
    # Tree connections
    connections = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    for start_idx, end_idx in connections:
        start_pos = tree_positions[start_idx]
        end_pos = tree_positions[end_idx]
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                'k-', linewidth=1.5, alpha=0.7)
    
    # Phase 5: Query Processing
    ax.text(10, 3.8, 'PHASE 5: COLOR QUERY PROCESSING', 
            fontsize=16, fontweight='bold', ha='center', color='#2c3e50')
    
    # Text query input
    query_box = FancyBboxPatch((1, 2.5), 4, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['pink'], alpha=0.8, 
                              edgecolor=colors['pink'], linewidth=2)
    ax.add_patch(query_box)
    ax.text(3, 2.9, 'TEXT QUERY\n"greenish-yellow"', fontsize=10, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Color parsing
    parsing_box = FancyBboxPatch((6, 2.5), 4, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['lime'], alpha=0.8, 
                                edgecolor=colors['lime'], linewidth=2)
    ax.add_patch(parsing_box)
    ax.text(8, 2.9, 'COLOR PARSING\n(Extended color map)', fontsize=10, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Target color
    target_box = FancyBboxPatch((11, 2.5), 3, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['yellow'], alpha=0.8, 
                               edgecolor=colors['yellow'], linewidth=2)
    ax.add_patch(target_box)
    ax.text(12.5, 2.9, 'TARGET COLOR\nRGB: (154, 205, 50)', fontsize=9, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Adaptive radius
    radius_box = FancyBboxPatch((15, 2.5), 4, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['coral'], alpha=0.8, 
                               edgecolor=colors['coral'], linewidth=2)
    ax.add_patch(radius_box)
    ax.text(17, 2.9, 'ADAPTIVE RADIUS\n(80 for "ish" colors)', fontsize=10, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Phase 6: Color Filtering & Retrieval
    ax.text(10, 1.8, 'PHASE 6: COLOR FILTERING & IMAGE RETRIEVAL', 
            fontsize=16, fontweight='bold', ha='center', color='#2c3e50')
    
    # KD-Tree search
    search_box = FancyBboxPatch((1, 0.5), 5, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['lavender'], alpha=0.8, 
                               edgecolor=colors['lavender'], linewidth=2)
    ax.add_patch(search_box)
    ax.text(3.5, 0.9, 'KD-TREE SEARCH\n(Within radius)', fontsize=10, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Color filtered results
    results_box = FancyBboxPatch((7, 0.5), 6, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['olive'], alpha=0.8, 
                                edgecolor=colors['olive'], linewidth=2)
    ax.add_patch(results_box)
    ax.text(10, 0.9, 'COLOR-FILTERED RESULTS\n(Pre-filtered images)', fontsize=10, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Final retrieval
    final_box = FancyBboxPatch((14, 0.5), 5, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['red'], alpha=0.8, 
                              edgecolor=colors['red'], linewidth=2)
    ax.add_patch(final_box)
    ax.text(16.5, 0.9, 'FINAL RETRIEVAL\n(Top 10 results)', fontsize=10, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Arrows connecting phases
    arrows = [
        ((10, 11.5), (10, 10.8)),  # Dataset to Color Extraction
        ((10, 9.5), (10, 8.8)),    # Extraction to Storage
        ((10, 7.5), (10, 6.8)),    # Storage to KD-Tree
        ((10, 5.5), (10, 3.8)),    # KD-Tree to Query
        ((10, 2.9), (10, 1.8)),    # Query to Filtering
        ((10, 0.9), (10, 0.3))     # Filtering to Final
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc='#2c3e50', ec='#2c3e50', linewidth=3)
        ax.add_patch(arrow)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['blue'], alpha=0.3, label='Dataset'),
        patches.Patch(color=colors['purple'], alpha=0.8, label='Color Clustering'),
        patches.Patch(color=colors['green'], alpha=0.6, label='Metadata Storage'),
        patches.Patch(color=colors['orange'], alpha=0.8, label='KD-Tree Index'),
        patches.Patch(color=colors['pink'], alpha=0.8, label='Text Query'),
        patches.Patch(color=colors['lime'], alpha=0.8, label='Color Parsing'),
        patches.Patch(color=colors['yellow'], alpha=0.8, label='Target Color'),
        patches.Patch(color=colors['coral'], alpha=0.8, label='Adaptive Radius'),
        patches.Patch(color=colors['lavender'], alpha=0.8, label='Spatial Search'),
        patches.Patch(color=colors['olive'], alpha=0.8, label='Color Filtering'),
        patches.Patch(color=colors['red'], alpha=0.8, label='Final Results')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98),
             fontsize=8, framealpha=0.9, ncol=2)
    
    plt.tight_layout()
    return fig

def create_color_map_diagram():
    """Create a diagram showing the extended color map and parsing"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), dpi=300)
    
    colors = {
        'blue': '#1f77b4', 'red': '#d62728', 'green': '#2ca02c', 'yellow': '#f7dc6f',
        'purple': '#9467bd', 'orange': '#ff7f0e', 'pink': '#e377c2', 'brown': '#8c564b',
        'gray': '#7f7f7f', 'cyan': '#17becf', 'lime': '#7dce94', 'coral': '#f1948a'
    }
    
    # Left: Basic Colors
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    ax1.text(5, 9.5, 'Basic Color Mapping', fontsize=16, fontweight='bold', ha='center')
    
    basic_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
    positions = [(2, 8), (4, 8), (6, 8), (8, 8), (2, 6), (4, 6), (6, 6), (8, 6)]
    
    for color_name, pos in zip(basic_colors, positions):
        # Color patch
        color_patch = FancyBboxPatch((pos[0]-0.4, pos[1]-0.3), 0.8, 0.6, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor=colors[color_name], alpha=0.8, 
                                    edgecolor='black', linewidth=1)
        ax1.add_patch(color_patch)
        
        # Color name
        ax1.text(pos[0], pos[1]-0.5, color_name, fontsize=10, fontweight='bold', 
                ha='center', va='center')
    
    # Right: Color Combinations
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    ax2.text(5, 9.5, 'Color Combinations & Blending', fontsize=16, fontweight='bold', ha='center')
    
    combinations = [
        ('greenish-yellow', 'lime', (2, 8)),
        ('bluish-green', 'cyan', (4, 8)),
        ('reddish-orange', 'coral', (6, 8)),
        ('purplish-blue', 'purple', (8, 8)),
        ('light blue', 'cyan', (2, 6)),
        ('dark red', 'red', (4, 6)),
        ('pale green', 'lime', (6, 6)),
        ('deep purple', 'purple', (8, 6))
    ]
    
    for combo_name, color_name, pos in combinations:
        # Color patch
        color_patch = FancyBboxPatch((pos[0]-0.4, pos[1]-0.3), 0.8, 0.6, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor=colors[color_name], alpha=0.8, 
                                    edgecolor='black', linewidth=1)
        ax2.add_patch(color_patch)
        
        # Combination name
        ax2.text(pos[0], pos[1]-0.5, combo_name, fontsize=8, fontweight='bold', 
                ha='center', va='center')
    
    # Add regex patterns
    ax1.text(5, 4, 'Regex Patterns:\n• "greenish-yellow" → blend(green, yellow)\n• "light blue" → lighten(blue)\n• "dark red" → darken(red)', 
             fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax2.text(5, 4, 'Blending Logic:\n• "ish" = 50% blend\n• "light" = +30% brightness\n• "dark" = -30% brightness\n• "pale" = +20% brightness', 
             fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_adaptive_radius_diagram():
    """Create a diagram showing adaptive radius logic"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    colors = {
        'blue': '#1f77b4', 'red': '#d62728', 'green': '#2ca02c', 'yellow': '#f7dc6f',
        'purple': '#9467bd', 'orange': '#ff7f0e', 'pink': '#e377c2', 'brown': '#8c564b'
    }
    
    ax.text(7, 9.5, 'Adaptive Radius Logic', fontsize=18, fontweight='bold', ha='center')
    
    # Query examples with different radii
    examples = [
        ('"blue"', 60, 'blue', (2, 7)),
        ('"light blue"', 80, 'cyan', (5, 7)),
        ('"greenish-yellow"', 100, 'lime', (8, 7)),
        ('"dark red"', 80, 'red', (11, 7))
    ]
    
    for query, radius, color_name, pos in examples:
        # Query box
        query_box = FancyBboxPatch((pos[0]-1, pos[1]-0.5), 2, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors[color_name], alpha=0.7, 
                                  edgecolor=colors[color_name], linewidth=2)
        ax.add_patch(query_box)
        ax.text(pos[0], pos[1], query, fontsize=11, fontweight='bold', 
                ha='center', va='center', color='white')
        
        # Radius circle
        radius_circle = Circle((pos[0], pos[1]-2), radius/50, color=colors[color_name], 
                              alpha=0.3, linewidth=2, fill=False)
        ax.add_patch(radius_circle)
        
        # Radius label
        ax.text(pos[0], pos[1]-2.5, f'Radius: {radius}', fontsize=10, fontweight='bold', 
                ha='center', va='center')
        
        # Arrow
        ax.plot([pos[0], pos[0]], [pos[1]-0.5, pos[1]-1.5], 'k->', linewidth=2, markersize=8)
    
    # Logic explanation
    logic_text = """
    ADAPTIVE RADIUS RULES:
    
    • Exact colors (e.g., "blue", "red") → Radius = 60
    • Light/Dark modifiers → Radius = 80  
    • Color combinations (e.g., "ish") → Radius = 100
    • Complex descriptions → Radius = 100
    
    This ensures:
    - Precise matching for simple colors
    - Flexible matching for complex descriptions
    - Better user experience with natural language
    """
    
    ax.text(7, 3, logic_text, fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Creating color processing diagrams...")
    
    # Create main color processing diagram
    fig1 = create_color_processing_diagram()
    fig1.savefig('color_processing_pipeline.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ Saved: color_processing_pipeline.png")
    
    # Create color map diagram
    fig2 = create_color_map_diagram()
    fig2.savefig('color_mapping_system.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ Saved: color_mapping_system.png")
    
    # Create adaptive radius diagram
    fig3 = create_adaptive_radius_diagram()
    fig3.savefig('adaptive_radius_logic.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ Saved: adaptive_radius_logic.png")
    
    plt.show()
    print("🎉 Color processing diagrams created successfully!") 