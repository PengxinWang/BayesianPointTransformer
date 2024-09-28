import os
import numpy as np
import torch
import torch.utils.data
import open3d as o3d
import matplotlib.pyplot as plt

from weaver.datasets import build_dataset
from weaver.utils.logger import get_root_logger
from weaver.utils.registry import Registry
from weaver.utils.misc import intersection_and_union

VISUALIZERS = Registry("visualizers")

class VisualizerBase:
    def __init__(self, cfg) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(log_file=os.path.join(cfg.save_path, "vis.log"),
                                      file_mode="a" if cfg.resume else "w",
                                      )
        self.cfg = cfg
        self.vis_dataset = build_dataset(self.cfg.data.vis)

    def vis(self):
        raise NotImplementedError

@VISUALIZERS.register_module("SemSegVisualizer")
class SemSegVisualizer(VisualizerBase):
    def vis(self, idx):
        self.logger.info(">>>>>>>>>>>>>>>> Start Visualization >>>>>>>>>>>>>>>>")
        save_path = os.path.join(self.cfg.save_path, "result")
        self.logger.info(f"pred path: {self.cfg.save_path}")
        data_dict = self.vis_dataset.get_data(idx)
        segment = data_dict.pop("segment")
        data_name = data_dict.pop("name")
        coord = data_dict.pop("coord")
        self.logger.info(f'name: {data_name}')
        self.logger.info(f'segment: {segment.shape}')
        self.logger.info(f'coord: {coord.shape}')
        pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
        assert os.path.isfile(pred_save_path)
        pred = np.load(pred_save_path)
        self.plot(pred, segment, coord, self.cfg.data.num_classes)
        self.logger.info("<<<<<<<<<<<<<<<<< End Visualization <<<<<<<<<<<<<<<<<")
    
    def plot(self, pred, gt, points, num_classes):
        colormap = plt.get_cmap('tab20', num_classes)
        colormap = np.array([colormap(i)[:3] for i in range(num_classes)])
        pred_pc = o3d.geometry.PointCloud()
        gt_pc = o3d.geometry.PointCloud()
        pred_pc.points=o3d.utility.Vector3dVector(points)
        gt_pc.points=o3d.utility.Vector3dVector(points)
        pred_colors = np.array([colormap[label] for label in pred])
        gt_colors = np.array([colormap[label] for label in gt])
        pred_pc.colors = o3d.utility.Vector3dVector(pred_colors)
        gt_pc.colors = o3d.utility.Vector3dVector(gt_colors)
        self.logger.info(f'displaying prediction vs ground truth')
        o3d.visualization.draw_geometries([pred_pc], window_name='pred')
        o3d.visualization.draw_geometries([gt_pc], window_name='ground truth')
