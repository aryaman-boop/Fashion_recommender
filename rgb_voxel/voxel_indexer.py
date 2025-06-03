import pandas as pd
import numpy as np
import ast  # for safely parsing string lists

class RGBVoxelIndexer:
    def __init__(self, csv_path, divisions=5):
        self.divisions = divisions
        self.data = pd.read_csv(csv_path)

        # Parse the 'rgb' column from string to [R, G, B]
        self.data[['R', 'G', 'B']] = self.data['rgb'].apply(lambda x: pd.Series(ast.literal_eval(x)))
        
        self.voxel_size = 256 // divisions
        self.voxel_map = {}
        self._index_data()

    def _index_data(self):
        for idx, row in self.data.iterrows():
            voxel = self._get_voxel(row['R'], row['G'], row['B'])
            self.voxel_map.setdefault(voxel, []).append((row['R'], row['G'], row['B']))

    def _get_voxel(self, r, g, b):
        return (int(r) // self.voxel_size, int(g) // self.voxel_size, int(b) // self.voxel_size)

    def find_neighbors(self, r, g, b, k=10):
    from math import dist

    target_voxel = self._get_voxel(r, g, b)
    searched_cubes = set()
    neighbors = []

    # Helper to collect neighbors from a cube
    def collect_from_cube(cube):
        if cube in self.voxel_map:
            searched_cubes.add(cube)
            neighbors.extend(self.voxel_map[cube])

    # Step 1: Check current voxel
    collect_from_cube(target_voxel)

    # Step 2: Expand until we have enough points
    radius = 1
    max_radius = 4  # Prevent infinite expansion in sparse data
    while len(neighbors) < k and radius <= max_radius:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    neighbor_voxel = (
                        target_voxel[0] + dx,
                        target_voxel[1] + dy,
                        target_voxel[2] + dz
                    )
                    if neighbor_voxel not in searched_cubes:
                        collect_from_cube(neighbor_voxel)
        radius += 1

    # Log searched cubes and total found
    print(f"[DEBUG] Query RGB=({r},{g},{b}) → Voxel={target_voxel}")
    print(f"[DEBUG] Cubes searched: {len(searched_cubes)}")
    print(f"[DEBUG] Points found: {len(neighbors)}")

    # Step 3: Sort by Euclidean distance
    neighbors.sort(key=lambda p: dist((r, g, b), p))
    return neighbors[:k]