
import os, time

import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt


#
# Parsing and Saving
#

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_npz(path):
    print('Loading ' + path + '...')
    data = np.load(path, allow_pickle=True)
    data_dict = dict()
    for f in data.files:
        data_dict[f] = data[f]
    return data_dict

def parse_obj_points(points, point2frame):
    num_frames = np.amax(point2frame)+1 #len(np.unique(point2frame))
    object_seq = [points[point2frame == i] for i in range(num_frames)]

    return object_seq

def parse_scene_points(points, point2frame, frame2sequence):
    num_obj_seqs = len(np.unique(frame2sequence))
    ptr = [0] + [np.argmax(frame2sequence == i) for i in range(1, num_obj_seqs)] + [len(frame2sequence)]
    objects = [] # list of lists of np arrays that are point clouds
    for obj_id in range(num_obj_seqs):
        point_mask = frame2sequence[point2frame] == obj_id
        obj_points = points[point_mask]
        local_frame_ids = point2frame[point_mask] - ptr[obj_id]
        pc_seq = []
        for frame_id in range(ptr[obj_id+1] - ptr[obj_id]):
            pc_seq.append(obj_points[local_frame_ids == frame_id])
        objects.append(pc_seq)

    return objects

def parse_scene_bboxes(bboxes, frame2sequence):
    num_obj_seqs = len(np.unique(frame2sequence))
    obj_bboxes = [bboxes[frame2sequence == i] for i in range(num_obj_seqs)]
    return obj_bboxes

def parse_scene_conf(confidences, frame2sequence):
    num_obj_seqs = len(np.unique(frame2sequence))
    bbox_conf = [confidences[frame2sequence == i] for i in range(num_obj_seqs)]
    return bbox_conf

#
# Visualization
#

SEQ_CM = ['Purples', 'Oranges', 'Blues', 'Reds', 'Greens', 'Greys']
# SEQ_CM = ['YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
                    # 'Greys'

COLORS = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

COLOR_BY_OPTS = {'time', 'error'}

def save_viewpoint(vis):
    print('Saving viewpoint...')
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    trajectory = o3d.camera.PinholeCameraTrajectory()
    trajectory.parameters = [param]
    o3d.io.write_pinhole_camera_trajectory('./viewpoint_%d.json' % (int(time.time())), trajectory)

def load_viewpoint(vis, filename):
    ctr = vis.get_view_control()
    trajectory = o3d.io.read_pinhole_camera_trajectory(filename)
    ctr.convert_from_pinhole_camera_parameters(trajectory.parameters[0])

def create_visualizer(width=1920, height=1080):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(86, save_viewpoint) # v key
    vis.create_window(width=width, height=height)
    print('Press `v` to save viewpoint...')

    return vis

def draw_axis(vis,
              origin=[0.0, 0.0, 0.0]
             ):
    origin = np.array(origin)
    points = np.array([
        origin,
        origin + np.array([1.0, 0.0, 0.0]),
        origin + np.array([0.0, 1.0, 0.0]),
        origin + np.array([0.0, 0.0, 1.0]),
    ])
    lines = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]
    colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

def bbox2corners(trans, lwh, yaw):
    '''
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1

      + heading vector from top middle to between 4-5 are index 8,9
    '''
    l, w, h = lwh
    hl, hw, hh = l / 2., w / 2., h / 2.
    local_corners = np.array([
        [hl, hw, -hh],
        [hl, -hw, -hh],
        [-hl, -hw, -hh],
        [-hl, hw, -hh],
        [hl, hw, hh],
        [hl, -hw, hh],
        [-hl, -hw, hh],
        [-hl, hw, hh],
        # heading vector
        [0.0, 0.0, hh],
        [hl, 0.0, hh]
    ])

    R = np.eye(3)
    yaw_x, yaw_y = np.cos(yaw), np.sin(yaw)
    R[:,0] = np.array([yaw_x, yaw_y, 0.0])
    R[:,1] = np.array([-yaw_y, yaw_x, 0.0])
    corners = np.dot(R, local_corners.T).T + trans[np.newaxis]
    return corners


def draw_point_seq(vis, point_seq,
                    colormap='rainbow',
                    subsamp=1):
    '''
    :param point_seq: [optional] list of np arrays that are (M,3)
    :param colormap: any of the matplotlib colormaps
    :param subsamp: subsamples the sequence by this much
    '''
    # other possible cmaps: rainbow, nipy_spectral, YlGnBu, GnBu, gist_rainbow
    # plot points
    T = len(point_seq)
    cmap = plt.get_cmap(colormap)
    color_list = [list(cmap(float(t) / (T)))[:3] for t in range(T)]

    for t, step_pts in enumerate(point_seq):
        if not t % subsamp == 0:
            continue
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(step_pts[:,:3]))
        pc.paint_uniform_color(color_list[t])
        vis.add_geometry(pc)

def draw_bbox_seq(vis, bboxes,
                    bbox_valid=None,
                    bbox_center=False,
                    bbox_errors=None,
                    colormap='rainbow',
                    color_by='time',
                    line_width=0.01,
                    max_err=0.5,
                    subsamp=1):
    '''
    :param bboxes: (N, 7) with [x, y, z, w, l, h, \yaw]
    :param bbox_valid: (N,) True, where entry in bboxes is valid and should be visaulized
    :param bbox_center: if true, displays a point at bbox centers
    :param bbox_errors: (N,) the error for each bbox to display
    :param colormap: any of the matplotlib colormaps
    :param color_by: 'time' or 'error'
    :param subsamp: subsamples the sequence by this much
    '''
    assert color_by in COLOR_BY_OPTS
    # other possible cmaps: rainbow, nipy_spectral, YlGnBu, GnBu, gist_rainbow

    # plot bboxes
    T = bboxes.shape[0]
    if isinstance(colormap, str):
        cmap = plt.get_cmap(colormap)
    else:
        cmap = colormap

    if color_by == 'time':
        color_list = [list(cmap(0.6*float(t) / (T) + 0.2))[:3] for t in range(T)]
    elif color_by == 'error':
        color_list = [list(cmap(bbox_errors[t] / max_err))[:3] for t in range(T)]

    for t in range(T):
        if not t % subsamp == 0:
            continue
        if bbox_valid is not None and not bbox_valid[t]:
            continue
        draw_bbox(vis, bboxes[t], color_list[t],
                    viz_center=bbox_center,
                    line_width=line_width)

def place_camera(vis,
                  cam_trans=[0.0, 0.0, 0.0]):
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    extrins = np.eye(4)
    extrins[:3,2] = [0.0, -1.0, 0.0]
    extrins[:3,1] = [0.0, 0.0, 1.0]
    extrins[:3, 3] = cam_trans
    params.extrinsic = o3d.utility.Matrix4dVector(extrins[np.newaxis])[0]
    ctr.convert_from_pinhole_camera_parameters(params)

def set_render_options(vis,
                       point_size=5.0):
    opt = vis.get_render_option()
    opt.point_size = point_size

def render(vis, out_path,
            point_size=5.0,
            viewpoint=None,
            show=False):
    if viewpoint is not None:
        load_viewpoint(vis, viewpoint)
    # else:
    #     place_camera(vis, [0.0, 0.0, 0.0])
    set_render_options(vis, point_size)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path)
    if show:
        vis.run()

def destroy_visualizer(vis):
    vis.destroy_window()