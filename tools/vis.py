import os
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt

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

def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)

def plot_original_pc(data_dict, vis_save_path, return_camera_params=False):
    vis_save_path = os.path.join(vis_save_path, 'original_pc_{}.png'.format(data_dict["name"]))
    pc = o3d.geometry.PointCloud()
    coord_normalized, color_normalized = normalize(data_dict["coord"], data_dict["color"])
    pc.points=o3d.utility.Vector3dVector(coord_normalized)
    pc.colors=o3d.utility.Vector3dVector(color_normalized)

    vis=o3d.visualization.Visualizer()
    vis.create_window(window_name='original_pc_{}'.format(data_dict["name"]),
                      width=800,
                      height=600)
    
    vis.add_geometry(pc)
    vis.run()
    vis.capture_screen_image(vis_save_path)
    if return_camera_params:
        ctr=vis.get_view_control()
        camera_params = ctr.convert_to_pinhole_camera_parameters()
    vis.destroy_window()
    print(f'saved image to: {vis_save_path}')
    if return_camera_params:
        return camera_params
 
def plot_seg_gd(data_dict, vis_save_path, color_map, camera_params=None):
    vis_save_path = os.path.join(vis_save_path, 'seg_gd_{}.png'.format(data_dict["name"]))
    color = [color_map[idx] for idx in data_dict['segment']]
    pc = o3d.geometry.PointCloud()
    coord_normalized, color_normalized = normalize(data_dict["coord"], color)
    pc.points=o3d.utility.Vector3dVector(coord_normalized)
    pc.colors=o3d.utility.Vector3dVector(color_normalized)

    vis=o3d.visualization.Visualizer()
    vis.create_window(window_name='seg_gd_{}'.format(data_dict["name"]),
                      width=800,
                      height=600)
    vis.add_geometry(pc)
    if camera_params:
        ctr=vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_params)
    else:
        vis.run()
    vis.capture_screen_image(vis_save_path)
    vis.destroy_window()
    print(f'saved image to: {vis_save_path}')

def plot_seg_pred(data_dict, save_path, vis_save_path, color_map, camera_params=None):
    vis_save_path = os.path.join(vis_save_path, 'seg_pred_{}.png'.format(data_dict["name"]))
    save_path = os.path.join(save_path, '{}_pred.npy'.format(data_dict["name"]))
    seg_pred = np.load(save_path)
    color = [color_map[idx] for idx in seg_pred]
    pc = o3d.geometry.PointCloud()
    coord_normalized, color_normalized = normalize(data_dict["coord"], color)
    pc.points=o3d.utility.Vector3dVector(coord_normalized)
    pc.colors=o3d.utility.Vector3dVector(color_normalized)

    vis=o3d.visualization.Visualizer()
    vis.create_window(window_name='seg_pred_{}'.format(data_dict["name"]),
                      width=800,
                      height=600)
    vis.add_geometry(pc)
    if camera_params:
        ctr=vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_params)
    else:
        vis.run()
    vis.capture_screen_image(vis_save_path)
    vis.destroy_window()
    print(f'saved image to: {vis_save_path}')

def plot_uc(data_dict, save_path, vis_save_path, uc_type='aleatoric', camera_params=None):
    vis_save_path = os.path.join(vis_save_path, '{}_{}.png'.format(uc_type, data_dict["name"]))
    save_path = os.path.join(save_path, '{}_uncertainty.pt'.format(data_dict["name"]))
    uncertainty = torch.load(save_path)[uc_type].numpy()
    uncertainty_normalized = (uncertainty - np.min(uncertainty))/(np.max(uncertainty)- np.min(uncertainty))
    color = np.zeros((len(uncertainty_normalized), 3))
    color[:, 2] = uncertainty_normalized
    pc = o3d.geometry.PointCloud()
    coord_normalized, _ = normalize(data_dict["coord"], color)
    pc.points=o3d.utility.Vector3dVector(coord_normalized)
    pc.colors=o3d.utility.Vector3dVector(color)

    vis=o3d.visualization.Visualizer()
    vis.create_window(window_name='{}_{}'.format(uc_type, data_dict["name"]),
                      width=800,
                      height=600)
    vis.add_geometry(pc)
    if camera_params:
        ctr=vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_params)
    else:
        vis.run()
    vis.capture_screen_image(vis_save_path)
    vis.destroy_window()
    print(f'saved image to: {vis_save_path}')

def main():
    args = default_argument_parser().parse_args()
    args.config_file = f'D:\PointNet\PointNet\configs\S3DIS/bnn_balanced_large_sphere_rpe.py'
    args.options = {"save_path": f'D:\PointNet\PointNet\exp\S3DIS/bnn_balanced_large_sphere'}
    cfg = default_config_parser(args.config_file, args.options)
    test_dataset = build_dataset(cfg.data.test)
    save_path = os.path.join(cfg.save_path, "result")
    vis_save_path = os.path.join(cfg.save_path, "vis_result")
    os.makedirs(vis_save_path, exist_ok=True)

    idx = 4
    data_dict = test_dataset.get_data(idx)
    # camera_params = plot_original_pc(data_dict, vis_save_path, return_camera_params=True)
    # color_map = build_color_map(dataset=cfg.dataset_type)
    # plot_seg_gd(data_dict, vis_save_path, color_map, camera_params)
    # plot_seg_pred(data_dict, save_path, vis_save_path, color_map, camera_params)
    # plot_uc(data_dict, save_path, vis_save_path, 'epistemic', camera_params)

    plot_original_pc(data_dict, vis_save_path, return_camera_params=False, rotating=True)
    color_map = build_color_map(dataset=cfg.dataset_type)
    plot_seg_gd(data_dict, vis_save_path, color_map)
    plot_seg_pred(data_dict, save_path, vis_save_path, color_map)
    plot_uc(data_dict, save_path, vis_save_path, 'aleatoric')
    plot_uc(data_dict, save_path, vis_save_path, 'epistemic')

if __name__ == "__main__":
    main()