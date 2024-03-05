import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.utils.utils_smpl import SMPL
from lib.utils.utils_mesh import unity_rot_to_angle_axis

class MeshRegressor(nn.Module):
    def __init__(self, args, dim_rep=512, num_joints=17, hidden_dim=2048, dropout_ratio=0.):
        super(MeshRegressor, self).__init__()
        param_pose_dim = 24 * 6
        self.head_kp_3d = nn.Linear(dim_rep, 3) 

        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(num_joints*dim_rep, hidden_dim)
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.fc2 = nn.Linear(num_joints*dim_rep, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.head_pose = nn.Linear(hidden_dim, param_pose_dim)
        nn.init.xavier_uniform_(self.head_pose.weight, gain=0.01)

    def forward(self, feat, init_pose=None, init_shape=None):
        N, T, J, C = feat.shape    # C == arg.dim_rep
        NT = N * T

        kp_3d = self.head_kp_3d(feat)    #N, T, J, 3  # 

        feat = feat.reshape(N, T, -1)

        feat_pose = feat.reshape(NT, -1)     # (N*T, J*C)

        feat_pose = self.dropout(feat_pose)
        feat_pose = self.fc1(feat_pose)
        feat_pose = self.bn1(feat_pose)
        feat_pose = self.relu1(feat_pose)    # (NT, C)

        pred_pose = self.head_pose(feat_pose) # (NT, 24*6)

        # convert unity forward_dir and up_dir to angle_axis
        pose = unity_rot_to_angle_axis(pred_pose)  
        output = [{
            'kp_3d'  : kp_3d,                       # (N*T, J, 3)
            'theta'  : torch.cat([pose], dim=1),    # (N*T, 72)
            'dir_fu' : pred_pose,                   # (N*T, 24*6) #target dirs forward ,up

        }]
        return output
    
# return theta and kp_3d17
class UnityRegressor(nn.Module):
    def __init__(self, args, backbone, dim_rep=512, num_joints=17, hidden_dim=2048, dropout_ratio=0.5):
        super(UnityRegressor, self).__init__()
        self.backbone = backbone
        self.feat_J = num_joints
        self.head = MeshRegressor(args, dim_rep, num_joints, hidden_dim, dropout_ratio)
        
    def forward(self, x, init_pose=None, init_shape=None, n_iter=3):
        '''
            Input: (N x T x 17 x 3) 
        '''
        N, T, J, C = x.shape  
        feat = self.backbone.get_representation(x)  
        feat = feat.reshape([N, T, self.feat_J, -1])      # (N, T, J, C)
        unity_output = self.head(feat)
        for s in unity_output:
            s['theta'] = s['theta'].reshape(N, T, -1)
            s['kp_3d'] =  s['kp_3d'].reshape(N, T, -1, 3)
            s['dir_fu'] = s['dir_fu'].reshape(N, T, -1)  # (N, T, 24*6) #target dirs forward ,up
        return unity_output