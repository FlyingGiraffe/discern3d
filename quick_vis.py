import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import open3d.visualization as vis


vis_directory = "/mnt/c/Users/colton/Documents/discern3d/tmp_viz"
# vis_directory = "/C/Users/colton/Documents/discern3d/tmp_viz"
viz_files = os.listdir(vis_directory)
a0_viz_files = sorted([f for f in viz_files if f.startswith("agent1")])
print("Num files: {0}".format(len(a0_viz_files)))

for i, fname in enumerate(a0_viz_files):
    print(i)
    if i < 85:
        continue
    with open(os.path.join(vis_directory, fname), "rb") as f:
        voxel_data = pickle.load(f)
    coarse_grid, fine_grid = voxel_data['coarse'], voxel_data['fine']

    # visualize coarse grid
    pc = o3d.geometry.PointCloud()
    voxel_idxs = np.stack(np.where(coarse_grid), axis=1)
    pc.points = o3d.utility.Vector3dVector(voxel_idxs)
    o3d_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=1.0)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
    o3d.visualization.draw_geometries([o3d_voxel, frame])

    # visualize fine grid
    pc = o3d.geometry.PointCloud()
    print("Number of populated voxels: {0}".format(np.sum(fine_grid)))
    voxel_idxs = np.stack(np.where(fine_grid), axis=1)
    pc.points = o3d.utility.Vector3dVector(voxel_idxs)
    o3d_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=1.0)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
    o3d.visualization.draw_geometries([o3d_voxel, frame])
