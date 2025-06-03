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
        voxel = self._get_voxel(r, g, b)
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_voxel = (voxel[0]+dx, voxel[1]+dy, voxel[2]+dz)
                    neighbors.extend(self.voxel_map.get(neighbor_voxel, []))
        distances = [((nr - r)**2 + (ng - g)**2 + (nb - b)**2, (nr, ng, nb)) for nr, ng, nb in neighbors]
        distances.sort(key=lambda x: x[0])
        return [color for _, color in distances[:k]]