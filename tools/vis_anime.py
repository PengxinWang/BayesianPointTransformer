import os
import numpy as np
import open3d as o3d
import imageio

from pointbnn.datasets import build_dataset
from pointbnn.engines.defaults import (
    default_argument_parser,
    default_config_parser,
)

def build_color_map(dataset="S3DISDataset", class_names=None, save_color_legend=False):
    if dataset == "S3DISDataset":
        color_map = [[128,64,128], [244,35,232], [70,70,70], [102,102,156],
                       [190,153,153], [153,153,153], [250,170,30], [220,220,0],
                       [107,142,35], [151,251,152], [70,160,180], [220,20,60],
                       [255,0,0],]
        if save_color_legend:
            assert class_names is not None
    else:
        raise NotImplementedError
    return color_map

def normalize(coord, color):
    coord, color = np.array(coord), np.array(color)
    centroid = np.mean(coord, axis=0)
    coord_centered = coord - centroid
    max_distance = np.max(np.linalg.norm(coord_centered, axis=1))
    coord_normalized = coord_centered / max_distance
    color_normalized = np.array(color) / 255.0
    return coord_normalized, color_normalized

def plot_rotating_pc(data_dict, vis_save_path):
    vis_save_path = os.path.join(vis_save_path, 'frames')
    if not os.path.exists(vis_save_path):
        os.makedirs(vis_save_path)
    pc = o3d.geometry.PointCloud()
    coord_normalized, color_normalized = normalize(data_dict["coord"], data_dict["color"])
    pc.points=o3d.utility.Vector3dVector(coord_normalized)
    pc.colors=o3d.utility.Vector3dVector(color_normalized)
    def rotate_view(vis):
        nonlocal frame_idx
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        frame_path=os.path.join(vis_save_path, f'frame_{frame_idx:04d}.png')
        vis.capture_screen_image(frame_path)
        frame_idx += 1
        if frame_idx > 120:
            return True
        else:
            return False
    frame_idx=0
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="original_pc_{}".format(data_dict["name"]), width=800, height=600)
    vis.add_geometry(pc)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_up([0.,-1.,0.])
    o3d.visualization.draw_geometries_with_animation_callback([pc], rotate_view)
    return vis_save_path    

def create_gif_from_frames(frame_folder, gif_save_path):
    images = []
    for filename in sorted(os.listdir(frame_folder)):
        if filename.endswith(".png"):
            images.append(imageio.imread(os.path.join(frame_folder, filename)))
    imageio.mimsave(gif_save_path, images, duration=5)

def main():
    args = default_argument_parser().parse_args()
    args.config_file = f'D:\PointNet\PointNet\configs\S3DIS/bnn_balanced_large_sphere_rpe.py'
    args.options = {"save_path": f'D:\PointNet\PointNet\exp\S3DIS/bnn_balanced_large_sphere'}
    cfg = default_config_parser(args.config_file, args.options)
    test_dataset = build_dataset(cfg.data.test)
    save_path = os.path.join(cfg.save_path, "result")
    vis_save_path = os.path.join(cfg.save_path, "vis_result")
    gif_save_path = os.path.join(vis_save_path, 'origin_pc_{}.gif'.format(data_dict["name"]))
    os.makedirs(vis_save_path, exist_ok=True)

    idx = 4
    data_dict = test_dataset.get_data(idx)
    # frame_folder = plot_rotating_pc(data_dict, vis_save_path)
    frame_folder = f'D:\PointNet\PointNet\exp\S3DIS/bnn_balanced_large_sphere/vis_result/frames'
    create_gif_from_frames(frame_folder, gif_save_path)
if __name__ == "__main__":
    main()