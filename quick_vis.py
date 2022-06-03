import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import open3d.visualization as vis


# vis_directory = "/mnt/c/Users/colton/Documents/discern3d/tmp_viz"
# vis_directory = "/C/Users/colton/Documents/discern3d/tmp_viz"
# vis_directory = "/home/colton/Documents/discern3d/beach-ball-viz"
vis_directory = "/home/colton/Documents/discern3d/tmp_viz"

viz_files = os.listdir(vis_directory)
a0_viz_files = sorted([f for f in viz_files if f.startswith("agent0")])
a1_viz_files = sorted([f for f in viz_files if f.startswith("agent1")])
a2_viz_files = sorted([f for f in viz_files if f.startswith("agent2")])
print("Num files: {0}".format(len(a0_viz_files)))

# for i, fname in enumerate(a0_viz_files):
# print(i)
# if i < 40:
#     continue
with open(os.path.join(vis_directory, a0_viz_files[-2]), "rb") as f:
    voxel_data_0 = pickle.load(f)
with open(os.path.join(vis_directory, a1_viz_files[-2]), "rb") as f:
    voxel_data_1 = pickle.load(f)
with open(os.path.join(vis_directory, a2_viz_files[-2]), "rb") as f:
    voxel_data_2 = pickle.load(f)

coarse_grid, fine_grid0 = voxel_data_0['coarse'], voxel_data_0['fine']
fine_grid1 = voxel_data_1['fine']
fine_grid2 = voxel_data_2['fine']

complete_grid = fine_grid0 | fine_grid1 | fine_grid2
colors = np.zeros((fine_grid0.shape[0], fine_grid0.shape[0], fine_grid0.shape[0], 3))
colors[fine_grid0] += np.array([1, 0, 0])
colors[fine_grid1] += np.array([0, 1, 0])
colors[fine_grid2] += np.array([0, 0, 1])


pc = o3d.geometry.PointCloud()
print("Number of populated voxels: {0}".format(np.sum(complete_grid)))


voxel_idxs = np.stack(np.where(complete_grid), axis=1)
print(voxel_idxs.shape)
voxel_clrs = colors[voxel_idxs[:, 0], voxel_idxs[:, 1], voxel_idxs[:, 2]]
print(">>>>>>>>>>>")
# print(voxel_clrs)
print(voxel_clrs.shape)
pc.points = o3d.utility.Vector3dVector(voxel_idxs)
pc.colors = o3d.utility.Vector3dVector(voxel_clrs)
o3d_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=1.0)
# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
o3d.visualization.draw_geometries([o3d_voxel])

# visualize coarse grid
pc = o3d.geometry.PointCloud()
voxel_idxs = np.stack(np.where(coarse_grid), axis=1)
pc.points = o3d.utility.Vector3dVector(voxel_idxs)
voxel_clrs = np.ones((voxel_idxs.shape[0], 3)) / 2
pc.colors = o3d.utility.Vector3dVector(voxel_clrs)
o3d_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=1.0)
# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
o3d.visualization.draw_geometries([o3d_voxel])

# visualize fine grid


