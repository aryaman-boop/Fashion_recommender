#!/usr/bin/env python3
"""
Create a comprehensive detailed diagram of color processing with explanations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle, Rectangle
import numpy as np

# Set style for beautiful diagrams
plt.style.use('default')
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'DejaVu Sans'

def create_comprehensive_color_diagram():
    """Create a comprehensive diagram with detailed explanations"""
    
    # Create figure with high DPI for crisp output
    fig, ax = plt.subplots(1, 1, figsize=(24, 18), dpi=300)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Colors
    colors = {
        'blue': '#1f77b4', 'orange': '#ff7f0e', 'green': '#2ca02c', 'red': '#d62728',
        'purple': '#9467bd', 'brown': '#8c564b', 'pink': '#e377c2', 'gray': '#7f7f7f',
        'olive': '#bcbd22', 'cyan': '#17becf', 'yellow': '#f7dc6f', 'lime': '#7dce94',
        'coral': '#f1948a', 'lavender': '#bb8fce', 'lightblue': '#87ceeb'
    }
    
    # Title
    ax.text(12, 17.5, 'COMPREHENSIVE COLOR PROCESSING & SEARCHING PIPELINE', 
            fontsize=24, fontweight='bold', ha='center', color='#2c3e50')
    ax.text(12, 17, 'Composed Image Retrieval (CIR) System - Detailed Analysis', 
            fontsize=16, ha='center', color='#34495e')
    
    # ============================================================================
    # PHASE 1: DATA PREPARATION & COLOR EXTRACTION
    # ============================================================================
    
    # Phase 1 Header
    phase1_box = FancyBboxPatch((0.5, 15.5), 23, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['blue'], alpha=0.8, 
                               edgecolor=colors['blue'], linewidth=3)
    ax.add_patch(phase1_box)
    ax.text(12, 16, 'PHASE 1: DATA PREPARATION & COLOR EXTRACTION', 
            fontsize=18, fontweight='bold', ha='center', color='white')
    
    # Dataset Info
    dataset_info = FancyBboxPatch((0.5, 14), 7, 1.2, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['lightblue'], alpha=0.7, 
                                 edgecolor=colors['blue'], linewidth=2)
    ax.add_patch(dataset_info)
    ax.text(4, 14.6, 'DATASET STATISTICS', fontsize=12, fontweight='bold', ha='center', color='#2c3e50')
    ax.text(4, 14.2, '• Total Images: 37,825', fontsize=10, ha='center', color='#2c3e50')
    ax.text(4, 13.8, '• Dress: 11,642 images', fontsize=10, ha='center', color='#2c3e50')
    ax.text(4, 13.4, '• Shirt: 13,918 images', fontsize=10, ha='center', color='#2c3e50')
    ax.text(4, 13.0, '• Top/Tee: 12,265 images', fontsize=10, ha='center', color='#2c3e50')
    
    # Color Extraction Process
    extraction_box = FancyBboxPatch((8, 14), 7, 1.2, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['purple'], alpha=0.7, 
                                   edgecolor=colors['purple'], linewidth=2)
    ax.add_patch(extraction_box)
    ax.text(11.5, 14.6, 'COLOR EXTRACTION PROCESS', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(11.5, 14.2, '• K-Means Clustering (k=5)', fontsize=10, ha='center', color='white')
    ax.text(11.5, 13.8, '• Extract dominant colors', fontsize=10, ha='center', color='white')
    ax.text(11.5, 13.4, '• RGB & HSV conversion', fontsize=10, ha='center', color='white')
    ax.text(11.5, 13.0, '• Color histogram analysis', fontsize=10, ha='center', color='white')
    
    # Sample Images with Colors
    sample_box = FancyBboxPatch((16, 14), 7, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['green'], alpha=0.7, 
                               edgecolor=colors['green'], linewidth=2)
    ax.add_patch(sample_box)
    ax.text(19.5, 14.6, 'SAMPLE COLOR EXTRACTION', fontsize=12, fontweight='bold', ha='center', color='white')
    
    # Sample color patches
    sample_colors = ['red', 'blue', 'green', 'yellow', 'purple']
    for i, color in enumerate(sample_colors):
        x = 16.5 + i * 1.2
        color_patch = FancyBboxPatch((x, 13.5), 0.8, 0.4, 
                                    boxstyle="round,pad=0.02", 
                                    facecolor=colors[color], alpha=0.8, 
                                    edgecolor='black', linewidth=1)
        ax.add_patch(color_patch)
        ax.text(x + 0.4, 13.3, color, fontsize=8, ha='center', fontweight='bold', color='white')
    
    # ============================================================================
    # PHASE 2: COLOR METADATA STORAGE & INDEXING
    # ============================================================================
    
    # Phase 2 Header
    phase2_box = FancyBboxPatch((0.5, 12.5), 23, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['orange'], alpha=0.8, 
                               edgecolor=colors['orange'], linewidth=3)
    ax.add_patch(phase2_box)
    ax.text(12, 12.9, 'PHASE 2: COLOR METADATA STORAGE & SPATIAL INDEXING', 
            fontsize=16, fontweight='bold', ha='center', color='white')
    
    # Metadata Structure
    metadata_box = FancyBboxPatch((0.5, 10.5), 11, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['cyan'], alpha=0.7, 
                                 edgecolor=colors['cyan'], linewidth=2)
    ax.add_patch(metadata_box)
    ax.text(6, 11.7, 'COLOR METADATA STRUCTURE', fontsize=14, fontweight='bold', ha='center', color='white')
    
    # Metadata fields
    fields = [
        ('Image ID', 2, 11.2),
        ('RGB Values', 4, 11.2),
        ('HSV Values', 6, 11.2),
        ('Dominant Color', 8, 11.2),
        ('Color Cluster', 10, 11.2)
    ]
    
    for field_name, x, y in fields:
        field_box = FancyBboxPatch((x-0.6, y-0.2), 1.2, 0.4, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor='white', alpha=0.9, 
                                  edgecolor='black', linewidth=1)
        ax.add_patch(field_box)
        ax.text(x, y, field_name, fontsize=9, fontweight='bold', ha='center', va='center', color='#2c3e50')
    
    # Example data
    ax.text(6, 10.8, 'Example: {"img_001": {"rgb": [255,0,0], "hsv": [0,100,100], "dominant": "red", "cluster": 1}}', 
            fontsize=9, ha='center', va='center', color='#2c3e50',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.7))
    
    # KD-Tree Indexing
    kdtree_box = FancyBboxPatch((12, 10.5), 11, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['pink'], alpha=0.7, 
                               edgecolor=colors['pink'], linewidth=2)
    ax.add_patch(kdtree_box)
    ax.text(17.5, 11.7, 'SPATIAL INDEXING (KD-TREE)', fontsize=14, fontweight='bold', ha='center', color='white')
    
    # Tree structure
    tree_positions = [
        (17.5, 11.2),      # Root
        (15.5, 10.8), (19.5, 10.8),  # Level 1
        (14.5, 10.4), (16.5, 10.4), (18.5, 10.4), (20.5, 10.4)  # Level 2
    ]
    
    for i, pos in enumerate(tree_positions):
        node_circle = Circle(pos, 0.15, color=colors['purple'], alpha=0.8, linewidth=2)
        ax.add_patch(node_circle)
        ax.text(pos[0], pos[1], f'N{i+1}', fontsize=8, fontweight='bold', ha='center', va='center', color='white')
    
    # Tree connections
    connections = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    for start_idx, end_idx in connections:
        start_pos = tree_positions[start_idx]
        end_pos = tree_positions[end_idx]
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'k-', linewidth=1.5, alpha=0.7)
    
    ax.text(17.5, 10.1, 'O(log n) search complexity for color-based filtering', 
            fontsize=9, ha='center', va='center', color='#2c3e50',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.7))
    
    # ============================================================================
    # PHASE 3: QUERY PROCESSING & COLOR PARSING
    # ============================================================================
    
    # Phase 3 Header
    phase3_box = FancyBboxPatch((0.5, 9.5), 23, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['green'], alpha=0.8, 
                               edgecolor=colors['green'], linewidth=3)
    ax.add_patch(phase3_box)
    ax.text(12, 9.9, 'PHASE 3: NATURAL LANGUAGE QUERY PROCESSING & COLOR PARSING', 
            fontsize=16, fontweight='bold', ha='center', color='white')
    
    # Text Query Input
    query_box = FancyBboxPatch((0.5, 7.5), 5, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['yellow'], alpha=0.8, 
                              edgecolor=colors['yellow'], linewidth=2)
    ax.add_patch(query_box)
    ax.text(3, 8.5, 'TEXT QUERY INPUT', fontsize=12, fontweight='bold', ha='center', color='#2c3e50')
    ax.text(3, 8.1, 'User Query:', fontsize=10, ha='center', color='#2c3e50')
    ax.text(3, 7.7, '"greenish-yellow dress"', fontsize=11, ha='center', color='#2c3e50', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
    
    # Color Parsing Engine
    parsing_box = FancyBboxPatch((6, 7.5), 5, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['lime'], alpha=0.8, 
                                edgecolor=colors['lime'], linewidth=2)
    ax.add_patch(parsing_box)
    ax.text(8.5, 8.5, 'COLOR PARSING ENGINE', fontsize=12, fontweight='bold', ha='center', color='#2c3e50')
    ax.text(8.5, 8.1, 'Extended Color Map:', fontsize=10, ha='center', color='#2c3e50')
    ax.text(8.5, 7.7, '80+ colors + combinations', fontsize=10, ha='center', color='#2c3e50')
    
    # Color Map Examples
    color_examples = [
        ('greenish-yellow', 'lime', (11.5, 8.3)),
        ('bluish-green', 'cyan', (11.5, 7.9)),
        ('reddish-orange', 'coral', (11.5, 7.5))
    ]
    
    for combo_name, color_name, pos in color_examples:
        color_patch = FancyBboxPatch((pos[0]-0.8, pos[1]-0.15), 1.6, 0.3, 
                                    boxstyle="round,pad=0.02", 
                                    facecolor=colors[color_name], alpha=0.8, 
                                    edgecolor='black', linewidth=1)
        ax.add_patch(color_patch)
        ax.text(pos[0], pos[1], combo_name, fontsize=8, ha='center', va='center', color='white', fontweight='bold')
    
    # Target Color Generation
    target_box = FancyBboxPatch((14, 7.5), 5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['coral'], alpha=0.8, 
                               edgecolor=colors['coral'], linewidth=2)
    ax.add_patch(target_box)
    ax.text(16.5, 8.5, 'TARGET COLOR GENERATION', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(16.5, 8.1, 'RGB: (154, 205, 50)', fontsize=10, ha='center', color='white')
    ax.text(16.5, 7.7, 'HSV: (60, 76, 80)', fontsize=10, ha='center', color='white')
    
    # Adaptive Radius Logic
    radius_box = FancyBboxPatch((20, 7.5), 3.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['lavender'], alpha=0.8, 
                               edgecolor=colors['lavender'], linewidth=2)
    ax.add_patch(radius_box)
    ax.text(21.75, 8.5, 'ADAPTIVE RADIUS', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(21.75, 8.1, '"ish" = 100', fontsize=10, ha='center', color='white')
    ax.text(21.75, 7.7, '"light/dark" = 80', fontsize=10, ha='center', color='white')
    
    # ============================================================================
    # PHASE 4: COLOR FILTERING & SEARCH
    # ============================================================================
    
    # Phase 4 Header
    phase4_box = FancyBboxPatch((0.5, 6.5), 23, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['red'], alpha=0.8, 
                               edgecolor=colors['red'], linewidth=3)
    ax.add_patch(phase4_box)
    ax.text(12, 6.9, 'PHASE 4: COLOR-BASED FILTERING & SPATIAL SEARCH', 
            fontsize=16, fontweight='bold', ha='center', color='white')
    
    # KD-Tree Search Process
    search_box = FancyBboxPatch((0.5, 4.5), 7, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['olive'], alpha=0.8, 
                               edgecolor=colors['olive'], linewidth=2)
    ax.add_patch(search_box)
    ax.text(4, 5.7, 'KD-TREE SPATIAL SEARCH', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(4, 5.3, '1. Query target color', fontsize=10, ha='center', color='white')
    ax.text(4, 4.9, '2. Find within radius', fontsize=10, ha='center', color='white')
    ax.text(4, 4.5, '3. Return candidates', fontsize=10, ha='center', color='white')
    
    # Search Visualization
    # Target point
    ax.plot(8, 5.5, 'ro', markersize=8, markeredgecolor='black', markeredgewidth=2)
    ax.text(8, 5.8, 'Target', fontsize=9, ha='center', fontweight='bold')
    
    # Search radius
    search_radius = Circle((8, 5.5), 1.5, color='red', alpha=0.3, linewidth=2, fill=False)
    ax.add_patch(search_radius)
    ax.text(9.5, 5.5, 'Radius = 100', fontsize=9, ha='center', fontweight='bold')
    
    # Candidate points
    candidate_positions = [(7, 5), (7.5, 6), (8.5, 5.2), (9, 4.8), (6.5, 5.3)]
    for i, pos in enumerate(candidate_positions):
        ax.plot(pos[0], pos[1], 'go', markersize=6, markeredgecolor='black', markeredgewidth=1)
        ax.text(pos[0], pos[1]-0.3, f'C{i+1}', fontsize=8, ha='center', fontweight='bold')
    
    # Color Filtered Results
    results_box = FancyBboxPatch((10, 4.5), 7, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['brown'], alpha=0.8, 
                                edgecolor=colors['brown'], linewidth=2)
    ax.add_patch(results_box)
    ax.text(13.5, 5.7, 'COLOR-FILTERED CANDIDATES', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(13.5, 5.3, 'Pre-filtered images:', fontsize=10, ha='center', color='white')
    ax.text(13.5, 4.9, 'Reduced search space', fontsize=10, ha='center', color='white')
    ax.text(13.5, 4.5, 'Performance optimization', fontsize=10, ha='center', color='white')
    
    # Performance Metrics
    perf_box = FancyBboxPatch((18, 4.5), 5.5, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['gray'], alpha=0.8, 
                             edgecolor=colors['gray'], linewidth=2)
    ax.add_patch(perf_box)
    ax.text(20.75, 5.7, 'PERFORMANCE METRICS', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(20.75, 5.3, 'Search time: O(log n)', fontsize=10, ha='center', color='white')
    ax.text(20.75, 4.9, 'Filter efficiency: 70%', fontsize=10, ha='center', color='white')
    ax.text(20.75, 4.5, 'Memory usage: Optimized', fontsize=10, ha='center', color='white')
    
    # ============================================================================
    # PHASE 5: FINAL RETRIEVAL & RANKING
    # ============================================================================
    
    # Phase 5 Header
    phase5_box = FancyBboxPatch((0.5, 3.5), 23, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['purple'], alpha=0.8, 
                               edgecolor=colors['purple'], linewidth=3)
    ax.add_patch(phase5_box)
    ax.text(12, 3.9, 'PHASE 5: CLIP-BASED RETRIEVAL & FINAL RANKING', 
            fontsize=16, fontweight='bold', ha='center', color='white')
    
    # CLIP Processing
    clip_box = FancyBboxPatch((0.5, 1.5), 7, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['cyan'], alpha=0.8, 
                             edgecolor=colors['cyan'], linewidth=2)
    ax.add_patch(clip_box)
    ax.text(4, 2.7, 'CLIP FEATURE EXTRACTION', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(4, 2.3, '• Image encoding', fontsize=10, ha='center', color='white')
    ax.text(4, 1.9, '• Text encoding', fontsize=10, ha='center', color='white')
    ax.text(4, 1.5, '• Similarity computation', fontsize=10, ha='center', color='white')
    
    # Combiner Network
    combiner_box = FancyBboxPatch((8, 1.5), 7, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['pink'], alpha=0.8, 
                                 edgecolor=colors['pink'], linewidth=2)
    ax.add_patch(combiner_box)
    ax.text(11.5, 2.7, 'COMBINER NETWORK', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(11.5, 2.3, '• AttentionFusion', fontsize=10, ha='center', color='white')
    ax.text(11.5, 1.9, '• Feature fusion', fontsize=10, ha='center', color='white')
    ax.text(11.5, 1.5, '• Joint representation', fontsize=10, ha='center', color='white')
    
    # Final Results
    final_box = FancyBboxPatch((16, 1.5), 7, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['red'], alpha=0.8, 
                              edgecolor=colors['red'], linewidth=2)
    ax.add_patch(final_box)
    ax.text(19.5, 2.7, 'FINAL RANKED RESULTS', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(19.5, 2.3, '• Top 10 images', fontsize=10, ha='center', color='white')
    ax.text(19.5, 1.9, '• Similarity scores', fontsize=10, ha='center', color='white')
    ax.text(19.5, 1.5, '• User presentation', fontsize=10, ha='center', color='white')
    
    # ============================================================================
    # ARROWS CONNECTING PHASES
    # ============================================================================
    
    arrows = [
        ((12, 15.5), (12, 12.5)),  # Phase 1 to Phase 2
        ((12, 12.5), (12, 9.5)),   # Phase 2 to Phase 3
        ((12, 9.5), (12, 6.5)),    # Phase 3 to Phase 4
        ((12, 6.5), (12, 3.5)),    # Phase 4 to Phase 5
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=25, fc='#2c3e50', ec='#2c3e50', linewidth=4)
        ax.add_patch(arrow)
    
    # ============================================================================
    # TECHNICAL DETAILS & ALGORITHMS
    # ============================================================================
    
    # Technical Details Box
    tech_box = FancyBboxPatch((0.5, 0.2), 23, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightyellow', alpha=0.8, 
                             edgecolor='orange', linewidth=2)
    ax.add_patch(tech_box)
    
    tech_text = """
    TECHNICAL IMPLEMENTATION DETAILS:
    • K-Means Clustering: k=5 clusters per image, extracts dominant colors with RGB/HSV conversion
    • KD-Tree Indexing: 3D spatial index (R,G,B) for O(log n) color search complexity
    • Extended Color Map: 80+ colors with regex patterns for "greenish-yellow", "light blue", etc.
    • Adaptive Radius: Dynamic search tolerance (60-100) based on query complexity
    • Pre-filtering: Color-based candidate selection reduces CLIP search space by ~70%
    • Performance: Sub-second response time for 37K+ image dataset
    """
    
    ax.text(12, 0.7, tech_text, fontsize=9, ha='center', va='center', color='#2c3e50',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # ============================================================================
    # LEGEND
    # ============================================================================
    
    legend_elements = [
        patches.Patch(color=colors['blue'], alpha=0.8, label='Data Preparation'),
        patches.Patch(color=colors['orange'], alpha=0.8, label='Storage & Indexing'),
        patches.Patch(color=colors['green'], alpha=0.8, label='Query Processing'),
        patches.Patch(color=colors['red'], alpha=0.8, label='Color Filtering'),
        patches.Patch(color=colors['purple'], alpha=0.8, label='Final Retrieval'),
        patches.Patch(color=colors['cyan'], alpha=0.7, label='Metadata'),
        patches.Patch(color=colors['pink'], alpha=0.7, label='Spatial Index'),
        patches.Patch(color=colors['yellow'], alpha=0.8, label='Text Input'),
        patches.Patch(color=colors['lime'], alpha=0.8, label='Color Parsing'),
        patches.Patch(color=colors['coral'], alpha=0.8, label='Target Color'),
        patches.Patch(color=colors['lavender'], alpha=0.8, label='Adaptive Radius'),
        patches.Patch(color=colors['olive'], alpha=0.8, label='Spatial Search'),
        patches.Patch(color=colors['brown'], alpha=0.8, label='Filtered Results'),
        patches.Patch(color=colors['gray'], alpha=0.8, label='Performance')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98),
             fontsize=7, framealpha=0.9, ncol=3)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Creating comprehensive color processing diagram...")
    
    # Create the comprehensive diagram
    fig = create_comprehensive_color_diagram()
    fig.savefig('comprehensive_color_processing.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ Saved: comprehensive_color_processing.png")
    
    plt.show()
    print("🎉 Comprehensive color processing diagram created successfully!") 