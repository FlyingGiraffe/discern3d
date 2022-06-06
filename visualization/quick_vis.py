import os
import pickle
import numpy as np
import open3d as o3d
import seaborn as sns
import viz_utils as utils

PALLETTE = sns.color_palette("bright")
c1, c3 = PALLETTE[1], PALLETTE[3]
PALLETTE[1] = c3
PALLETTE[3] = c1

# airplane viewpoints
VIEWPOINT = 'viewpoint_1654299775.json'  # fine resolution
# VIEWPOINT = 'viewpoint_1654299917.json'  # course resolution

# car viewpoints
# VIEWPOINT = 'viewpoint_1654300727.json'
# VIEWPOINT = 'viewpoint_1654300808.json'


def visualize(voxels, colors):
    print("Number of populated voxels: {0}".format(np.sum(voxels)))

    # convert to point cloud and visualize
    voxel_idxs = np.stack(np.where(voxels), axis=1)
    voxel_clrs = colors[voxel_idxs[:, 0], voxel_idxs[:, 1], voxel_idxs[:, 2]]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(voxel_idxs)
    pc.colors = o3d.utility.Vector3dVector(voxel_clrs)
    o3d_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=1.0)
    # o3d_voxel.material = o3d.visualization.Material('defaultLit')
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
    # material = o3d.visualization.rendering.MaterialRecord()

    vis = utils.create_visualizer(1920, 1080)
    vis.add_geometry(o3d_voxel)
    utils.render(vis, 'vis_out.png',
                 point_size=1.0,
                 viewpoint=VIEWPOINT,
                 show=True)

    vis.clear_geometries()
    utils.destroy_visualizer(vis)






def place_camera(vis, cam_trans=[2.0, 0.0, 2.0]):
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    extrins = np.eye(4)
    extrins[:3,2] = [0.0, -1.0, 0.0]
    extrins[:3,1] = [0.0, 0.0, 1.0]
    extrins[:3, 3] = cam_trans
    params.extrinsic = o3d.utility.Matrix4dVector(extrins[np.newaxis])[0]
    ctr.convert_from_pinhole_camera_parameters(params)

def show_final_reps(dir, n_agents):
    all_agent_files = os.listdir(dir)
    per_agent_files = []
    for aidx in range(n_agents):
        per_agent_files.append(sorted([f for f in all_agent_files if f.startswith("agent%s" % aidx)]))

    per_agent_fine_rep = []
    for aidx in range(n_agents):
        with open(os.path.join(vis_directory, per_agent_files[aidx][-1]), "rb") as f:
            voxel_data = pickle.load(f)
            per_agent_fine_rep.append(voxel_data['fine'])

    # get shared grid
    complete_grid = np.zeros(per_agent_fine_rep[0].shape, dtype=np.bool)
    for fine_rep in per_agent_fine_rep:
        complete_grid |= fine_rep

    # get coloring
    n_voxels = per_agent_fine_rep[0].shape[0]
    colors = np.zeros((n_voxels, n_voxels, n_voxels, 3))
    for i, fine_rep in enumerate(per_agent_fine_rep):
        colors[fine_rep] += np.array(PALLETTE[i][:3])
    white = np.sum(colors, axis=3) >= 3
    colors[white] /= 2 # np.array([0.7, 0.7, 0.7])

    # visualize
    visualize(complete_grid, colors)

def clean_final_rep_dir(dir, n_agents):
    all_agent_files = os.listdir(dir)
    per_agent_files = []
    for aidx in range(n_agents):
        agent_files = sorted([f for f in all_agent_files if f.startswith("agent%s" % aidx)])
        if len(agent_files) > 3:
            per_agent_files.append(agent_files[-2])
        else:
            per_agent_files.append(agent_files[0])

    for f in all_agent_files:
        if f not in per_agent_files:
            os.remove(os.path.join(dir, f))


def show_time_series(dir, n_agents, idxs, resolution="fine"):
    num_steps = len(idxs)
    all_agent_files = os.listdir(dir)
    per_agent_files = []
    for aidx in range(n_agents):
        per_agent_files.append(sorted([f for f in all_agent_files if f.startswith("agent%s" % aidx)]))
    print(per_agent_files)

    per_agent_fine_rep = []
    for aidx in range(n_agents):
        per_agent_fine_rep.append([])
        for j in range(num_steps):
            with open(os.path.join(vis_directory, per_agent_files[aidx][idxs[j]]), "rb") as f:
                voxel_data = pickle.load(f)['fine'] if resolution == "fine" else pickle.load(f)['coarse']
                per_agent_fine_rep[aidx].append(voxel_data)


    for j in range(num_steps):
        # get shared grid
        complete_grid = np.zeros(per_agent_fine_rep[0][0].shape, dtype=np.bool)
        for aidx in range(n_agents):
            complete_grid |= per_agent_fine_rep[aidx][j]
            if resolution == "coarse":
                break

        # get coloring
        n_voxels = per_agent_fine_rep[0][0].shape[0]
        colors = np.zeros((n_voxels, n_voxels, n_voxels, 3))
        for aidx in range(n_agents):
            colors[per_agent_fine_rep[aidx][j]] += np.array(PALLETTE[aidx][:3])
            if resolution == "coarse":
                colors[per_agent_fine_rep[aidx][j]] = np.array([0.7, 0.7, 0.7])
                break
        white = np.sum(colors, axis=3) >= 3
        colors[white] /= 2 # np.array([0.7, 0.7, 0.7])
        print("At step %s" % (idxs[j]))
        visualize(complete_grid, colors)






# visualize fine grid

if __name__ == "__main__":
    num_agents = 3
    vis_directory = "/home/colton/Documents/discern3d/tmp_viz"
    # clean_final_rep_dir(vis_directory, num_agents)
    show_final_reps(vis_directory, num_agents)
    # show_time_series(vis_directory, num_agents, idxs=[140], resolution="fine")

