#!/usr/bin/env python3
"""
Create a detailed diagram explaining the HNSW search process
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle, Rectangle, Arrow
import numpy as np

# Set style for beautiful diagrams
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

def create_hnsw_diagram():
    """Create a comprehensive diagram explaining HNSW search process"""
    
    # Create figure with high DPI for crisp output
    fig, ax = plt.subplots(1, 1, figsize=(20, 16), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Colors
    colors = {
        'blue': '#1f77b4', 'orange': '#ff7f0e', 'green': '#2ca02c', 'red': '#d62728',
        'purple': '#9467bd', 'brown': '#8c564b', 'pink': '#e377c2', 'gray': '#7f7f7f',
        'olive': '#bcbd22', 'cyan': '#17becf', 'yellow': '#f7dc6f', 'lime': '#7dce94',
        'coral': '#f1948a', 'lavender': '#bb8fce', 'lightblue': '#87ceeb'
    }
    
    # Title
    ax.text(10, 15.5, 'HNSW (Hierarchical Navigable Small World) SEARCH PROCESS', 
            fontsize=22, fontweight='bold', ha='center', color='#2c3e50')
    ax.text(10, 15, 'Composed Image Retrieval System - Detailed HNSW Implementation', 
            fontsize=14, ha='center', color='#34495e')
    
    # ============================================================================
    # SECTION 1: HNSW INDEX CONSTRUCTION
    # ============================================================================
    
    # Section 1 Header
    section1_box = FancyBboxPatch((0.5, 13.5), 19, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['blue'], alpha=0.8, 
                                 edgecolor=colors['blue'], linewidth=3)
    ax.add_patch(section1_box)
    ax.text(10, 14, 'SECTION 1: HNSW INDEX CONSTRUCTION & HIERARCHY', 
            fontsize=16, fontweight='bold', ha='center', color='white')
    
    # Index Creation Process
    creation_box = FancyBboxPatch((0.5, 11.5), 9, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['cyan'], alpha=0.7, 
                                 edgecolor=colors['cyan'], linewidth=2)
    ax.add_patch(creation_box)
    ax.text(5, 12.7, 'HNSW INDEX CREATION', fontsize=14, fontweight='bold', ha='center', color='white')
    ax.text(5, 12.3, '1. Initialize index with cosine similarity', fontsize=10, ha='center', color='white')
    ax.text(5, 11.9, '2. Set parameters: M=16, ef_construction=200', fontsize=10, ha='center', color='white')
    ax.text(5, 11.5, '3. Add embeddings with hierarchical structure', fontsize=10, ha='center', color='white')
    
    # Parameters Explanation
    params_box = FancyBboxPatch((10, 11.5), 9, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['purple'], alpha=0.7, 
                               edgecolor=colors['purple'], linewidth=2)
    ax.add_patch(params_box)
    ax.text(14.5, 12.7, 'HNSW PARAMETERS', fontsize=14, fontweight='bold', ha='center', color='white')
    ax.text(14.5, 12.3, 'M=16: Max connections per layer', fontsize=10, ha='center', color='white')
    ax.text(14.5, 11.9, 'ef_construction=200: Search depth during build', fontsize=10, ha='center', color='white')
    ax.text(14.5, 11.5, 'ef=50: Search depth during query', fontsize=10, ha='center', color='white')
    
    # Hierarchical Structure Visualization
    hierarchy_box = FancyBboxPatch((0.5, 9.5), 19, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['green'], alpha=0.7, 
                                  edgecolor=colors['green'], linewidth=2)
    ax.add_patch(hierarchy_box)
    ax.text(10, 10.7, 'HIERARCHICAL LAYER STRUCTURE', fontsize=14, fontweight='bold', ha='center', color='white')
    
    # Draw hierarchical layers
    layers = [
        (10, 10.2, 'Layer 0 (Top)', 0.8, colors['red']),      # Top layer
        (10, 9.8, 'Layer 1', 1.2, colors['orange']),          # Middle layer
        (10, 9.4, 'Layer 2 (Bottom)', 1.6, colors['yellow'])  # Bottom layer
    ]
    
    for x, y, label, width, color in layers:
        layer_rect = FancyBboxPatch((x-width/2, y-0.1), width, 0.2, 
                                   boxstyle="round,pad=0.02", 
                                   facecolor=color, alpha=0.8, 
                                   edgecolor='black', linewidth=1)
        ax.add_patch(layer_rect)
        ax.text(x, y, label, fontsize=9, ha='center', va='center', fontweight='bold', color='black')
    
    # Layer connections
    ax.plot([10, 10], [10.1, 9.9], 'k-', linewidth=2, alpha=0.7)
    ax.plot([10, 10], [9.7, 9.5], 'k-', linewidth=2, alpha=0.7)
    
    # ============================================================================
    # SECTION 2: SEARCH PROCESS
    # ============================================================================
    
    # Section 2 Header
    section2_box = FancyBboxPatch((0.5, 8.5), 19, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['orange'], alpha=0.8, 
                                 edgecolor=colors['orange'], linewidth=3)
    ax.add_patch(section2_box)
    ax.text(10, 8.9, 'SECTION 2: HNSW SEARCH ALGORITHM', 
            fontsize=16, fontweight='bold', ha='center', color='white')
    
    # Search Steps
    steps = [
        (2, 7.5, '1. START AT TOP LAYER', 'Find entry point in highest layer', colors['red']),
        (6, 7.5, '2. GREEDY SEARCH', 'Find nearest neighbor in current layer', colors['orange']),
        (10, 7.5, '3. DESCEND LAYERS', 'Move to lower layer and continue search', colors['yellow']),
        (14, 7.5, '4. BOTTOM SEARCH', 'Exhaustive search in bottom layer', colors['green']),
        (18, 7.5, '5. RETURN RESULTS', 'Top-k nearest neighbors', colors['blue'])
    ]
    
    for x, y, title, desc, color in steps:
        step_box = FancyBboxPatch((x-1.5, y-0.4), 3, 0.8, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=color, alpha=0.8, 
                                 edgecolor=color, linewidth=2)
        ax.add_patch(step_box)
        ax.text(x, y+0.2, title, fontsize=10, ha='center', fontweight='bold', color='white')
        ax.text(x, y-0.1, desc, fontsize=8, ha='center', color='white')
    
    # Connect steps with arrows
    for i in range(len(steps)-1):
        start_x = steps[i][0] + 1.5
        end_x = steps[i+1][0] - 1.5
        y = steps[i][1]
        ax.annotate('', xy=(end_x, y), xytext=(start_x, y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))
    
    # ============================================================================
    # SECTION 3: VISUAL SEARCH EXAMPLE
    # ============================================================================
    
    # Section 3 Header
    section3_box = FancyBboxPatch((0.5, 6.5), 19, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['green'], alpha=0.8, 
                                 edgecolor=colors['green'], linewidth=3)
    ax.add_patch(section3_box)
    ax.text(10, 6.9, 'SECTION 3: VISUAL SEARCH EXAMPLE', 
            fontsize=16, fontweight='bold', ha='center', color='white')
    
    # Search visualization
    # Query point
    ax.plot(3, 5.5, 'ro', markersize=12, markeredgecolor='black', markeredgewidth=2)
    ax.text(3, 5.8, 'Query', fontsize=11, ha='center', fontweight='bold', color='red')
    
    # Layer 0 (Top) - few points
    top_points = [(2, 5.2), (4, 5.3), (3.5, 5.1)]
    for i, pos in enumerate(top_points):
        ax.plot(pos[0], pos[1], 'o', markersize=8, color=colors['red'], alpha=0.8, markeredgecolor='black')
        ax.text(pos[0], pos[1]-0.2, f'T{i+1}', fontsize=8, ha='center', fontweight='bold')
    
    # Layer 1 (Middle) - more points
    mid_points = [(1.5, 4.8), (2.5, 4.9), (3.2, 4.7), (4.2, 4.8), (3.8, 4.6)]
    for i, pos in enumerate(mid_points):
        ax.plot(pos[0], pos[1], 'o', markersize=6, color=colors['orange'], alpha=0.8, markeredgecolor='black')
        ax.text(pos[0], pos[1]-0.15, f'M{i+1}', fontsize=7, ha='center', fontweight='bold')
    
    # Layer 2 (Bottom) - many points
    bottom_points = [(1, 4.3), (1.8, 4.4), (2.3, 4.2), (2.8, 4.3), (3.3, 4.1), 
                     (3.8, 4.2), (4.3, 4.3), (4.8, 4.1), (5.2, 4.2)]
    for i, pos in enumerate(bottom_points):
        ax.plot(pos[0], pos[1], 'o', markersize=4, color=colors['yellow'], alpha=0.8, markeredgecolor='black')
        ax.text(pos[0], pos[1]-0.1, f'B{i+1}', fontsize=6, ha='center', fontweight='bold')
    
    # Search path visualization
    search_path = [(3, 5.5), (3.5, 5.1), (3.2, 4.7), (3.3, 4.1)]
    for i in range(len(search_path)-1):
        start = search_path[i]
        end = search_path[i+1]
        ax.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=3, alpha=0.8)
        ax.plot([start[0], end[0]], [start[1], end[1]], 'r--', linewidth=1, alpha=0.5)
    
    # Layer labels
    ax.text(6, 5.3, 'Layer 0 (Top)', fontsize=12, ha='center', fontweight='bold', color=colors['red'])
    ax.text(6, 4.8, 'Layer 1 (Middle)', fontsize=12, ha='center', fontweight='bold', color=colors['orange'])
    ax.text(6, 4.3, 'Layer 2 (Bottom)', fontsize=12, ha='center', fontweight='bold', color=colors['yellow'])
    
    # ============================================================================
    # SECTION 4: IMPLEMENTATION DETAILS
    # ============================================================================
    
    # Section 4 Header
    section4_box = FancyBboxPatch((0.5, 3.5), 19, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['purple'], alpha=0.8, 
                                 edgecolor=colors['purple'], linewidth=3)
    ax.add_patch(section4_box)
    ax.text(10, 3.9, 'SECTION 4: IMPLEMENTATION IN OUR SYSTEM', 
            fontsize=16, fontweight='bold', ha='center', color='white')
    
    # Code Implementation
    code_box = FancyBboxPatch((0.5, 1.5), 9, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightgray', alpha=0.8, 
                             edgecolor='black', linewidth=2)
    ax.add_patch(code_box)
    ax.text(5, 2.7, 'HNSW INDEX CREATION', fontsize=12, fontweight='bold', ha='center', color='#2c3e50')
    
    code_lines = [
        "index = hnswlib.Index(space='cosine', dim=640)",
        "index.init_index(max_elements=37825, ef_construction=200, M=16)",
        "index.add_items(embeddings_array, np.arange(37825))",
        "index.set_ef(50)"
    ]
    
    for i, line in enumerate(code_lines):
        ax.text(5, 2.3 - i*0.2, line, fontsize=9, ha='center', fontfamily='monospace', color='#2c3e50')
    
    # Search Implementation
    search_code_box = FancyBboxPatch((10, 1.5), 9, 1.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='lightblue', alpha=0.8, 
                                    edgecolor='black', linewidth=2)
    ax.add_patch(search_code_box)
    ax.text(14.5, 2.7, 'HNSW SEARCH EXECUTION', fontsize=12, fontweight='bold', ha='center', color='#2c3e50')
    
    search_lines = [
        "labels, distances = index.knn_query(query_vector, k=10)",
        "top_indices = labels[0]",
        "top_img_ids = [img_ids[idx] for idx in top_indices]",
        "return top_img_ids"
    ]
    
    for i, line in enumerate(search_lines):
        ax.text(14.5, 2.3 - i*0.2, line, fontsize=9, ha='center', fontfamily='monospace', color='#2c3e50')
    
    # ============================================================================
    # SECTION 5: PERFORMANCE CHARACTERISTICS
    # ============================================================================
    
    # Performance Box
    perf_box = FancyBboxPatch((0.5, 0.2), 19, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightyellow', alpha=0.8, 
                             edgecolor='orange', linewidth=2)
    ax.add_patch(perf_box)
    
    perf_text = """
    PERFORMANCE CHARACTERISTICS:
    • Time Complexity: O(log n) for search, O(n log n) for construction
    • Space Complexity: O(n) for storing the index
    • Search Speed: Sub-second response for 37K+ embeddings
    • Accuracy: High recall with cosine similarity metric
    • Scalability: Efficient for large-scale image retrieval
    • Memory Usage: Optimized with hierarchical structure
    """
    
    ax.text(10, 0.7, perf_text, fontsize=10, ha='center', va='center', color='#2c3e50',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # ============================================================================
    # ARROWS CONNECTING SECTIONS
    # ============================================================================
    
    arrows = [
        ((10, 13.5), (10, 8.5)),  # Section 1 to Section 2
        ((10, 8.5), (10, 6.5)),   # Section 2 to Section 3
        ((10, 6.5), (10, 3.5)),   # Section 3 to Section 4
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=25, fc='#2c3e50', ec='#2c3e50', linewidth=4)
        ax.add_patch(arrow)
    
    # ============================================================================
    # LEGEND
    # ============================================================================
    
    legend_elements = [
        patches.Patch(color=colors['blue'], alpha=0.8, label='Index Construction'),
        patches.Patch(color=colors['orange'], alpha=0.8, label='Search Algorithm'),
        patches.Patch(color=colors['green'], alpha=0.8, label='Visual Example'),
        patches.Patch(color=colors['purple'], alpha=0.8, label='Implementation'),
        patches.Patch(color=colors['red'], alpha=0.8, label='Top Layer'),
        patches.Patch(color=colors['orange'], alpha=0.8, label='Middle Layer'),
        patches.Patch(color=colors['yellow'], alpha=0.8, label='Bottom Layer'),
        patches.Patch(color='lightgray', alpha=0.8, label='Code Example'),
        patches.Patch(color='lightblue', alpha=0.8, label='Search Code'),
        patches.Patch(color='lightyellow', alpha=0.8, label='Performance')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98),
             fontsize=8, framealpha=0.9, ncol=2)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Creating HNSW search process diagram...")
    
    # Create the HNSW diagram
    fig = create_hnsw_diagram()
    fig.savefig('hnsw_search_process.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ Saved: hnsw_search_process.png")
    
    plt.show()
    print("🎉 HNSW search process diagram created successfully!") 