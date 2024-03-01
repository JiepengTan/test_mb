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
from lib.utils.utils_data import flip_data
    
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
class MotionDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(MotionDataset3D, self).__init__(args, subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        print(file_path)
        motion_file = read_pkl(file_path)
        motion_3d = motion_file["data_label"]  
        motion_3d = torch.FloatTensor(motion_3d)
        motion_2d = motion_file["data_input"]
        motion_2d = torch.FloatTensor(motion_2d)
        if self.data_split=="train":
            if self.flip and random.random() > 0.5:# Training augmentation - random flipping                  
                motion_2d = flip_data(motion_2d)
                motion_3d = flip_data(motion_3d)
        elif not self.data_split=="test":       
            raise ValueError('Data split unknown.')    
        return motion_2d, motion_3d