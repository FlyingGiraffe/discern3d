import argparse
import os
import numpy as np

import configs
import dataset
import agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, default='Discern3D', help='config file name')
    parser.add_argument('--class_choice', type=str, default='airplane', help='dataset class choice')
    parser.add_argument('--shape_idx', type=int, default=4, help='dataset shape idx')
    args = parser.parse_args()
    
    config = getattr(configs, args.config)
    metadata = configs.extract_metadata(config, 0)
    metadata['dataset_kwargs']['class_choice'] = args.class_choice
    
    shapenet = dataset.ShapeNetPart(metadata['dataset_kwargs'])
    shape_data = shapenet[args.shape_idx][0]
    agents = [agent.Agent(i, metadata['agent_kwargs']) for i in range(metadata['num_agents'])]

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
        agents[i].run(shape_data, viewpoint_per_agent[i:i+1], identity_transition, gossip_hz=20, update_retry_hz=10)
    # agents[0].scan_loop(shape_data, np.array([[0.0, 0.0, 1.0]]), identity_transition, finished_callback, scan_hz=20, vis=True)
