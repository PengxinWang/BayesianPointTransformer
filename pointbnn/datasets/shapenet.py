import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointbnn.utils.logger import get_root_logger
from pointbnn.utils.cache import shared_dict

from .builder import DATASETS, build_dataset
from .transform import Compose, TRANSFORMS

@DATASETS.register_module()
class ShapeNetDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/shape_data",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
        loop=1,
        data_type=None,
    ):
        super(ShapeNetDataset, self).__init__()
        self.data_type=data_type
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.cache = cache
        self.ignore_index = ignore_index
        self.loop = loop if not test_mode else 1  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = self.test_cfg.voxelize
            self.test_crop = self.test_cfg.crop if self.test_cfg.crop else None
            self.post_transform = self.test_cfg.post_transform
            self.aug_transform = [aug for aug in self.test_cfg.aug_transform]

        self.data_dict_list = self.get_data_list()
        #self.logger = self.get_root_logger()
        #self.logger.info("Totally {} x {} samples in {} set.".format(len(self.data_dict_list), self.loop, split))

    def get_data_list(self):
        data_dict_list = {}

        for folder_name in os.listdir(self.data_root):
            folder_path = os.path.join(self.data_root, folder_name)
            
            if os.path.isdir(folder_path):
                points_path = os.path.join(folder_path, 'points')
                points_label_path = os.path.join(folder_path, 'points_label')
                
                if os.path.exists(points_path) and os.path.exists(points_label_path):
                    points_files = sorted(os.listdir(points_path))
                    points_label_files = sorted(os.listdir(points_label_path))
                    
                    if len(points_files) == len(points_label_files):
                        data_list = []
                        for i in range(len(points_files)):
                            points_file_path = os.path.join(points_path, points_files[i])
                            points_label_file_path = os.path.join(points_label_path, points_label_files[i])
                            data_list.append((points_file_path, points_label_file_path))
                        
                        data_dict_list[folder_name] = data_list

        return data_dict_list

    def get_data(self, index):
        if self.data_type not in self.data_dict_list:
            raise ValueError(f"Data type {self.data_type} not found in data_dict_list.")
        
        data_list = self.data_dict_list[self.data_type]
        
        if index >= len(data_list):
            raise ValueError(f"Index {index} is out of range for data type {self.data_type}.")
        
        points_file_path, points_label_file_path = data_list[index]
        
        with open(points_file_path, 'r') as f:
            coords = np.array([list(map(float, line.split())) for line in f])
        
        with open(points_label_file_path, 'r') as f:
            segments = np.array([int(line.strip()) for line in f])
        
        data_dict = {
            "coord": coords,
            "segment": segments
        }
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)        
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        data_dict_list.append(deepcopy(data_dict))
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list                        
        return result_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
