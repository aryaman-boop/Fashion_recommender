import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import heapq

# Step 1: Generate random RGB data points
np.random.seed(42)
num_points = 500
data_points = np.random.randint(0, 256, size=(num_points, 3))

# Step 2: Define voxel binning
bins = [0, 51, 102, 153, 204, 256]

def get_cell_index(value):
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return i
    return 4  # for value = 255

def get_voxel_coords(r, g, b):
    return (
        get_cell_index(r),
        get_cell_index(g),
        get_cell_index(b)
    )

# Step 3: Build the voxel grid
voxel_grid = defaultdict(list)
for point in data_points:
    voxel = get_voxel_coords(*point)
    voxel_grid[voxel].append(tuple(point))

# Step 4: Query function for finding nearest neighbors
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_neighboring_voxels(voxel, max_distance=1):
    x, y, z = voxel
    neighbors = []
    for dx in range(-max_distance, max_distance + 1):
        for dy in range(-max_distance, max_distance + 1):
            for dz in range(-max_distance, max_distance + 1):
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < 5 and 0 <= ny < 5 and 0 <= nz < 5:
                    neighbors.append((nx, ny, nz))
    return neighbors

def find_closest_points(query_rgb, k=10):
    voxel = get_voxel_coords(*query_rgb)
    candidates = []
    visited = set()
    distance = 0
    while len(candidates) < k and distance <= 5:
        for neighbor in get_neighboring_voxels(voxel, distance):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            candidates.extend(voxel_grid.get(neighbor, []))
        distance += 1
    heap = [(euclidean(query_rgb, pt), pt) for pt in candidates]
    return heapq.nsmallest(k, heap)

# Example query
query_point = [100, 150, 200]
closest_points = find_closest_points(query_point, k=10)

# Step 5: Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Voxel Cube with RGB Points")

# Plot all points
ax.scatter(data_points[:, 0], data_points[:, 1], data_points[:, 2],
           c=data_points / 255.0, s=20, alpha=0.5, label='Data Points')

# Highlight query point
ax.scatter(*query_point, c='black', s=100, marker='X', label='Query Point')

# Highlight closest points
closest_coords = np.array([pt for _, pt in closest_points])
ax.scatter(closest_coords[:, 0], closest_coords[:, 1], closest_coords[:, 2],
           c='red', s=60, marker='o', label='Closest Points')

# Cube grid lines for voxel visualization
for x in bins[:-1]:
    for y in bins[:-1]:
        for z in bins[:-1]:
            ax.plot([x, x], [y, y], [0, 255], color='gray', alpha=0.1)
            ax.plot([x, x], [0, 255], [z, z], color='gray', alpha=0.1)
            ax.plot([0, 255], [y, y], [z, z], color='gray', alpha=0.1)

ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.set_zlim(0, 255)
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.legend()
plt.tight_layout()
plt.savefig("voxel_rgb_plot.png", dpi=300)