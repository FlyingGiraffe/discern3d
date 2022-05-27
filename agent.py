import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski

class Agent(object):
    def __init__(self, kwargs):
        self.grid_coarse = kwargs['grid_coarse']
        self.grid_fine = kwargs['grid_fine']
        self.num_points_view = kwargs['num_points_view']
        self.num_points_scan = kwargs['num_points_scan']
        
        self.repn_coarse = np.zeros((kwargs['grid_coarse'], kwargs['grid_coarse'], kwargs['grid_coarse']), dtype=bool)
        self.repn_fine_idx = -np.ones((kwargs['grid_coarse'], kwargs['grid_coarse'], kwargs['grid_coarse']), dtype=int)
        self.repn_fine = []
        
    def scan(self, shape_data, view):
        '''
        Args:
          shape_data: full object pointcloud, size (num_points, 3)
          view: a view point on the unit sphere, size (1, 3)
        Output:
          repn_coarse_new: new coarse voxel grids, size (grid_coarse, grid_coarse, grid_coarse)
          repn_fine_idx_tmp: temporary indices for new fine voxel grids, size (grid_coarse, grid_coarse, grid_coarse)
          repn_fine_new: new fine voxel grids, list [elements of size (grid_fine, grid_fine, grid_fine)]
        '''
        points_view = farthest_subsample_points(shape_data, view, self.num_points_view)
        scan_idx = np.random.choice(self.num_points_view, self.num_points_scan)
        points_scan = points_view[scan_idx]
        
        repn_coarse_new = np.zeros(self.repn_coarse.shape)
        coarse_new_idx = np.floor((points_scan + 1) * self.grid_coarse / 2).astype(int)
        repn_coarse_new[coarse_new_idx[:, 0], coarse_new_idx[:, 1], coarse_new_idx[:, 2]] = 1
        
        num_vox_coarse_new = (repn_coarse_new != 0).sum()
        repn_fine_idx_tmp = -np.ones(self.repn_fine_idx.shape)
        repn_fine_idx_tmp[repn_coarse_new != 0] = np.arange(num_vox_coarse_new)
        
        repn_fine_new = []
        for i in range(num_vox_coarse_new):
            pass
        
        return repn_coarse_new, 


def farthest_subsample_points(pointcloud, view, num_subsampled_points=768):
    num_points = pointcloud.shape[0]
    nbrs = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud)
    idx = nbrs.kneighbors(view, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud[idx, :]