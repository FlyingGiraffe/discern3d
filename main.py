import argparse
import os
import numpy as np

import configs
import dataset
import agent
from network import Router

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, default='Discern3D', help='config file name')
    parser.add_argument('--class_choice', type=str, default='airplane', help='dataset class choice')
    parser.add_argument('--shape_idx', type=int, default=4, help='dataset shape idx')
    # ~500 for cars, 0-100 for airplanes
    args = parser.parse_args()

    config = getattr(configs, args.config)
    metadata = configs.extract_metadata(config, 0)
    metadata['dataset_kwargs']['class_choice'] = args.class_choice
    
    # shapenet = dataset.ShapeNetPart(metadata['dataset_kwargs'])
    # shape_data = shapenet[args.shape_idx][0]
    modelnet = dataset.ModelNet10HighRes(metadata['dataset_kwargs'])
    shape_data = modelnet[args.shape_idx][0].pos
    shape_data = shape_data.numpy()
    
    UniversalRouter = Router(metadata['agent_kwargs']['agent_ips'],
                            intragroup_link_failure_prob=metadata['simulation']['intragroup_link_failure_prob'], 
                            max_partitions=metadata['simulation']['max_partitions'], 
                            refresh_hz=metadata['simulation']['refresh_hz'])

    agents = [agent.Agent(UniversalRouter, i, metadata['agent_kwargs']) for i in range(metadata['num_agents'])]

    if not os.path.isdir( metadata['agent_kwargs']['output_dir'] ):
        os.makedirs(metadata['agent_kwargs']['output_dir'])
    
    # per-agent setup.
    # voxel_ids = list(range(metadata['agent_kwargs']['grid_coarse']**3))
    # ids2agents = {}
    # for i, ag in enumerate(agents):
    #     ids2agents[i] = ag
    # for i, ag in enumerate(agents):
    #     ag.setup({'id': i, 'id2agents': ids2agents, 'voxel_ids': voxel_ids})

    # Runs scanning loop
    
    viewpoint_per_agent = np.random.randn(len(agents),3)
    viewpoint_per_agent = viewpoint_per_agent / np.linalg.norm(viewpoint_per_agent, axis=1, keepdims=True)
    identity_transition = lambda x: np.random.randn(len(agents),3)
    finished_callback = lambda: False


    for i in range(len(agents)):
        agents[i].run(shape_data, viewpoint_per_agent[i:i+1], identity_transition, gossip_hz=6, update_retry_hz=6, scan_hz=2, clean_hz=6, lowres_hz=6)
    # agents[0].scan_loop(shape_data, np.array([[0.0, 0.0, 1.0]]), identity_transition, finished_callback, scan_hz=20, vis=True)
