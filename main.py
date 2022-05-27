import argparse
import os
import numpy as np

import configs
import dataset
import agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, default='Discern3D', help='config file name')
    parser.add_argument('--class_choice', type=str, default='chair', help='dataset class choice')
    parser.add_argument('--shape_idx', type=int, default=0, help='dataset shape idx')
    args = parser.parse_args()
    
    config = getattr(configs, args.config)
    metadata = configs.extract_metadata(config, 0)
    metadata['dataset_kwargs']['class_choice'] = args.class_choice
    
    shapenet = dataset.ShapeNetPart(metadata['dataset_kwargs'])
    shape_data = shapenet[args.shape_idx][0]
    agents = [agent.Agent(metadata['agent_kwargs']) for i in range(metadata['num_agents'])]
    
    agents[0].scan(shape_data, np.array([[0.0, 0.0, 1.0]]))