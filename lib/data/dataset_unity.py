import torch
import numpy as np
import glob
import os
import io
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from lib.data.augmentation import Augmenter3D
from lib.utils.tools import read_pkl
from lib.utils.utils_data import flip_data, crop_scale
from lib.utils.utils_mesh import flip_thetas
from lib.utils.utils_mesh import unity_rot_to_angle_axis


class MotionDataset(Dataset):
    def __init__(self, args, subset_list, data_split): # data_split: train/test
        np.random.seed(0)
        self.data_root = args.data_root
        self.subset_list = subset_list
        self.data_split = data_split
        file_list_all = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list_all.append(os.path.join(data_path, i))
        self.file_list = file_list_all
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError 

# if gt_2d == true , then just project the 3d pose to 2D (set pos.z = 1)
class UnityDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(UnityDataset3D, self).__init__(args, subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_3d17 = motion_file["data_label"]  
        motion_3d17 = torch.FloatTensor(motion_3d17)
        motion_2d = motion_file["data_input"]
        motion_2d = torch.FloatTensor(motion_2d)

        # extra 24 bone's forward and up dir vector
        bone_forward_up_dir = motion_file["data_dirs"]  
        bone_forward_up_dir = torch.FloatTensor(bone_forward_up_dir)
        motion_theta = unity_rot_to_angle_axis(bone_forward_up_dir)  

        # motion_3d24 = motion_file["data_kp3d"]  
        # motion_3d24 = torch.FloatTensor(motion_3d24)
        dir_fu = torch.FloatTensor([])

        # convert unity forward_dir and up_dir to angle_axis
        motion_smpl_3d = {
            'theta': motion_theta,       # bone rotation forward up
            'kp_3d': motion_3d17,       # 3D mesh vertices
            'dir_fu':dir_fu
            # 'kp_3d24': motion_3d24,        # 3D keypoints 24
        }
        return motion_2d, motion_smpl_3d
    