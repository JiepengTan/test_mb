import torch
import torch.nn as nn
import ipdb
from lib.utils.utils_mesh import batch_rodrigues
from lib.model.loss import *


class UnityLoss(nn.Module):
    def __init__(
            self,
            loss_type='MSE',
            device='cuda',
    ):
        super(UnityLoss, self).__init__()
        self.device = device
        self.loss_type = loss_type
        if loss_type == 'MSE': 
            self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
            self.criterion_regr = nn.MSELoss().to(self.device)
        elif loss_type == 'L1': 
            self.criterion_keypoints = nn.L1Loss(reduction='none').to(self.device)
            self.criterion_regr = nn.L1Loss().to(self.device)


    def forward(
            self,
            unity_output,
            data_gt,
    ):
        #  unity_output
        #  theta  (N, T, 24*6)
        #  kp_3d (N, T, 17, 3)
        #  dir_fu# (N, T, 24*6)
        preds = unity_output[-1]
        pred_dirs = preds['dir_fu'].reshape(-1, 6)
        real_dirs = data_gt['dir_fu'].reshape(-1, 6)

        pred_forward, pred_up = pred_dirs[:, :3], pred_dirs[:, 3:]
        real_forward, real_up = real_dirs[:, :3], real_dirs[:, 3:]

        len_f = torch.norm(real_forward, dim=-1, keepdim=True)
        len_u = torch.norm(real_up, dim=-1, keepdim=True)

        pred_forward_norm = pred_forward / len_f
        pred_up_norm = pred_up / len_u

        # Calculate the dot products for forward and up vectors
        loss_dict = {}
        loss_dict['loss_a'] = torch.abs((pred_forward_norm * real_forward).sum(dim=1) -1).mean()
        loss_dict['loss_a_up'] = torch.abs((pred_up_norm * real_up).sum(dim=1) -1).mean()
        loss_dict['loss_norm'] = torch.abs(len_f - 1).mean() + torch.abs(len_u - 1).mean()

        # TODO: Implement loss_3d_pos and loss_av calculations
        loss_dict['loss_3d_pos'] = torch.tensor(0, device=self.device, dtype=torch.float32)
        loss_dict['loss_av'] = torch.tensor(0, device=self.device, dtype=torch.float32)
        
        return loss_dict
        