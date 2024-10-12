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

def plot_original_pc(coord, color, vis_save_path, data_name, camera_cfg):
    vis_save_path = os.path.join(vis_save_path, 'original_pc_{}.png'.format(data_name))
    pc = o3d.geometry.PointCloud()
    pc.points=o3d.utility.Vector3dVector(coord)
    pc.colors=o3d.utility.Vector3dVector(color)

    vis=o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pc)

    ctr = vis.get_view_control()
    ctr.set_front(camera_cfg["front_vector"])
    ctr.set_up(camera_cfg["up_vector"])
    ctr.set_zoom(camera_cfg["zoom_factor"])
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(vis_save_path)
    vis.destroy_window()
    print(f'Saved image to: {vis_save_path}')
    

def plot_seg_gd(coord, seg_gd, vis_save_path, data_name):
    pass

def plot_seg_pred(coord, seg_pred_path, vis_save_path, data_name):
    pass

def plot_uc(coord, uc_path, vis_save_path, data_name):
    pass

def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    test_dataset = build_dataset(cfg.data.test)
    save_path = os.path.join(cfg.save_path, "result")
    vis_save_path = os.path.join(cfg.save_path, "vis_result")
    os.makedirs(vis_save_path, exist_ok=True)
    
    camera_config = {"front_vector": [0.2, 0, 0.8],
                     "up_vector": [0., 1., 0.],
                     "zoom_vector": 0.8}

    idx = 0
    data_dict = test_dataset.get_data(idx)
    seg_pred_path = os.path.join(save_path, "{}_pred.npy".format(data_dict["name"]))
    uc_path = os.path.join(save_path, "{}_uncertainty.pt".format(data_dict["name"]))
    plot_original_pc(data_dict["coord"], data_dict["color"], vis_save_path, data_dict["name"], camera_config)


if __name__ == "__main__":
    main()