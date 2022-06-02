import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import open3d.visualization as vis

def vis_voxel_grid(voxels):
    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    colors[voxels] = 'blue'
    # colors[cube1] = 'blue'
    # colors[cube2] = 'green'
    print("Rendering Voxel Grid!")

    voxel_idxs = np.stack(np.where(voxels), axis=1)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(voxel_idxs)
    o3d_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=1.0)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
    o3d.visualization.draw_geometries([o3d_voxel, frame])

    # # and plot everything
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.voxels(voxels, facecolors=colors, edgecolor='k')
    # plt.show()

    #     material = o3d.visualization.rendering.MaterialRecord()
    #
    #     render = rendering.OffscreenRenderer(640, 480)
    #     render.scene.add_geometry("voxels", o3d_voxel, material)
    #     render.scene.camera.look_at([0, 0, 0], [0, 1, 1], [0, 0, 1])
    #     img = render.render_to_image()
    #     print("=====")
    #     print(img)
    #     print(type(img))