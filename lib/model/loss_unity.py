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
        real_dir_fu = data_gt['dir_fu']
        preds_dir_fu = unity_output[-1]['dir_fu']
        N, T , J = real_dir_fu.shape[0], real_dir_fu.shape[1], real_dir_fu.shape[2]

        pred_dirs = preds_dir_fu.reshape(-1, 6)
        real_dirs = real_dir_fu.reshape(-1, 6)

        pred_f, pred_u = pred_dirs[:, :3], pred_dirs[:, 3:]
        real_f, real_u = real_dirs[:, :3], real_dirs[:, 3:]

        len_f = torch.norm(pred_f, dim=-1, keepdim=True)
        len_u = torch.norm(pred_u, dim=-1, keepdim=True)

        pred_f = pred_f / len_f
        pred_u = pred_u / len_u

        # Calculate the dot products for forward and up vectors
        loss_dict = {}
        loss_dict['loss_a'] = torch.abs((pred_f * real_f).sum(dim=1) -1).mean()
        loss_dict['loss_a_up'] = torch.abs((pred_u * real_u).sum(dim=1) -1).mean()
        loss_dict['loss_norm'] = torch.abs(len_f - 1).mean() + torch.abs(len_u - 1).mean()

        pred_dir_f = pred_f.reshape(N,T,J,3)
        real_dir_f = real_f.reshape(N,T,J,3)
        loss_dict['loss_av'] = loss_dir_angle_velocity_unity(pred_dir_f,real_dir_f)

        #print(loss_dict['loss_av'])
        # TODO: Implement loss_3d_pos calculations
        loss_dict['loss_3d_pos'] = torch.tensor(0, device=self.device, dtype=torch.float32)
        
        return loss_dict
        