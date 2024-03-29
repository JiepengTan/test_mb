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
    
class MotionDataset36(Dataset):
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
class MotionDataset3D36(MotionDataset36):
    def __init__(self, args, subset_list, data_split):
        super(MotionDataset3D36, self).__init__(args, subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d

    def do_convert_pkl_unity(self,file_path):
        # Load the .pkl file
        with open(file_path, 'rb') as f:
            motion_data = pickle.load(f)

        # Extract data_input and data_label
        data_input = motion_data.get("data_input", [])
        data_label = motion_data.get("data_label", [])

        # Flatten the data arrays into lists of floats
        data_input_flat = data_input.flatten().tolist() if data_input is not None else []
        data_label_flat = data_label.flatten().tolist() if data_label is not None else []

        # Create the dictionary for the MotionBERTClip class structure
        motion_bert_clip_dict = {
            'data_input': data_input_flat,
            'data_label': data_label_flat
        }

        # Convert the dictionary to JSON
        json_data = json.dumps(motion_bert_clip_dict, indent=4)

        # Save the JSON data to a file with the same name but with a .json extension
        json_file_path = os.path.splitext(file_path)[0] + '.json'
        with open(json_file_path, 'w') as json_file:
            json_file.write(json_data)

        print(f"Converted {file_path} to {json_file_path}")


    #each pkl contain ( Normalized in windows space)
    #     "data_input": //(243, 17, 3)
    #     "data_label": //(243, 17, 3)
    def convert_pkl_unity(self,index):
        file_path = self.file_list[index]
        self.do_convert_pkl_unity(file_path)
        # TODO 
        return
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_3d = motion_file["data_label"]  
        if self.data_split=="train":
            if self.synthetic or self.gt_2d:
                motion_3d = self.aug.augment3D(motion_3d)
                motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1                        # No 2D detection, use GT xy and c=1.
            elif motion_file["data_input"] is not None:     # Have 2D detection 
                motion_2d = motion_file["data_input"]
                if self.flip and random.random() > 0.5:                        # Training augmentation - random flipping
                    motion_2d = flip_data(motion_2d)
                    motion_3d = flip_data(motion_3d)
            else:
                raise ValueError('Training illegal.') 
        elif self.data_split=="test":                                           
            motion_2d = motion_file["data_input"]
            if self.gt_2d:
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1
        else:
            raise ValueError('Data split unknown.')    
        return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)