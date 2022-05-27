import json
path_data = json.load(open('paths.json'))
from .utils import *

Discern3D = {
    'dataset_kwargs': {
        'path': path_data['shapenet_path'],
        'partition': 'trainval',
        'num_points': 2048,
    },
    
    'num_agents': 5,
    'agent_kwargs': {
        # scene representation
        'grid_coarse': 8,
        'grid_fine': 4,
        
        # scanner simulation
        'num_points_view': 512, # number of points that can be seen from a view
        'num_points_scan': 128, # subsample points to be the actural scan
    },
}