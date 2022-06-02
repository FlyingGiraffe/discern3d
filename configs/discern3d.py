import json
import os

orig_dir = os.getcwd()
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
path_data = json.load(open('paths.json'))
os.chdir(orig_dir)
from .utils import *

Discern3D = {
    'dataset_kwargs': {
        'path': path_data['shapenet_path'],
        'partition': 'trainval',
        'num_points': 2048,
    },
    
    'num_agents': 3,
    'agent_kwargs': {
        # scene representation
        'grid_coarse': 32,
        'grid_fine': 4,
        
        # scanner simulation
        'num_points_view': 512, # number of points that can be seen from a view
        'num_points_scan': 16, # subsample points to be the actural scan

        # for the protocol
        'T': 3, # the target number of agents that each voxel block should be stored in.
        'K': 2, # the max number of voxel blocks that each agent can store
        'allocation_discrepancy_threshold': 2,
        'agent_ips': [('127.0.0.1', 9000),
                        ('127.0.0.1', 9001),
                        ('127.0.0.1', 9002)]
                        # ('127.0.0.1', 9003)]
                        # ('127.0.0.1', 9004)]
    },
}