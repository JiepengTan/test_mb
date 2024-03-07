import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Numpy-based errors

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1), axis=1)

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1), axis=1)


# PyTorch-based errors (for losses)

def loss_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def loss_2d_weighted(predicted, target, conf):
    assert predicted.shape == target.shape
    predicted_2d = predicted[:,:,:,:2]
    target_2d = target[:,:,:,:2]
    diff = (predicted_2d - target_2d) * conf
    return torch.mean(torch.norm(diff, dim=-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    # 将标准姿势 投影到 预测姿势
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True) 
    # 获取预测姿势到目标姿势的 缩放值，归一化，希望让误差计算更加的合理
    scale = norm_target / norm_predicted
    
    return loss_mpjpe(scale * predicted, target)

def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length

def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length)/gt_3d_length, 2).mean()
    return loss_length

def get_limb_lens(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0,1], [1,2], [2,3],
         [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10],
         [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]
        ]
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    limb_lens = torch.norm(limbs, dim=-1)
    return limb_lens

def loss_limb_var(x):
    '''
        Input: (N, T, 17, 3)
    '''
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    limb_lens = get_limb_lens(x)
    limb_lens_var = torch.var(limb_lens, dim=1)
    limb_loss_var = torch.mean(limb_lens_var)
    return limb_loss_var

def loss_limb_gt(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_lens_x = get_limb_lens(x)
    limb_lens_gt = get_limb_lens(gt) # (N, T, 16)
    return nn.L1Loss()(limb_lens_x, limb_lens_gt)

def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:,1:] - predicted[:,:-1]
    velocity_target = target[:,1:] - target[:,:-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))

def loss_joint(predicted, target):
    assert predicted.shape == target.shape
    return nn.L1Loss()(predicted, target)

def get_angles(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0,1], [1,2], [2,3],
         [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10],
         [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]
        ]
    angle_id = [[ 0,  3],
                [ 0,  6],
                [ 3,  6],
                [ 0,  1],
                [ 1,  2],
                [ 3,  4],
                [ 4,  5],
                [ 6,  7],
                [ 7, 10],
                [ 7, 13],
                [ 8, 13],
                [10, 13],
                [ 7,  8],
                [ 8,  9],
                [10, 11],
                [11, 12],
                [13, 14],
                [14, 15] ]
    eps = 1e-7
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    angles = limbs[:,:,angle_id,:]
    angle_cos = F.cosine_similarity(angles[:,:,:,0,:], angles[:,:,:,1,:], dim=-1)
    return torch.acos(angle_cos.clamp(-1+eps, 1-eps))

def loss_angle(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_angles_x = get_angles(x)
    limb_angles_gt = get_angles(gt)
    return nn.L1Loss()(limb_angles_x, limb_angles_gt)

def loss_angle_velocity(x, gt):
    """
    Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert x.shape == gt.shape
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    x_a = get_angles(x)
    gt_a = get_angles(gt)
    x_av = x_a[:,1:] - x_a[:,:-1]
    gt_av = gt_a[:,1:] - gt_a[:,:-1]
    return nn.L1Loss()(x_av, gt_av)


def get_dir_angle_velocity_unity(pred_dir_nrom):
    '''
        Input: (N, T, 24, 3)
        Output: (N, T, 23)
    '''
    eps = 1e-7
    angle_cos = F.cosine_similarity(pred_dir_nrom[:, :-1, :, :], pred_dir_nrom[:, 1:, :, :], dim=-1)
    return torch.acos(angle_cos.clamp(-1+eps, 1-eps)) 
    
def loss_dir_angle_velocity_unity(x, gt):
    """
        Input: (N, T, 24, 3), (N, T, 24, 3)
        Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    x_av = get_dir_angle_velocity_unity(x)
    gt_av = get_dir_angle_velocity_unity(gt)
    return nn.L1Loss()(x_av, gt_av)


def get_angles_unity(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
        use unity's HumanBodyBones define
    '''
    # child , parent
    limbs_id = [
        [1,0 ], #LeftUpperLeg,Hips
        [2,0 ], #RightUpperLeg,Hips
        [3,1 ], #LeftLowerLeg,LeftUpperLeg
        [4,2 ], #RightLowerLeg,RightUpperLeg
        [5,3 ], #LeftFoot,LeftLowerLeg
        [6,4 ], #RightFoot,RightLowerLeg
        [7,0 ], #Spine,Hips
        [8,7 ], #Chest,Spine
        [9,8 ], #UpperChest,Chest
        [10,9 ], #Neck,UpperChest
        [11,10 ], #Head,Neck
        [12,9 ], #LeftShoulder,UpperChest
        [13,9 ], #RightShoulder,UpperChest
        [14,12 ], #LeftUpperArm,LeftShoulder
        [15,13 ], #RightUpperArm,RightShoulder
        [16,14 ], #LeftLowerArm,LeftUpperArm
        [17,15 ], #RightLowerArm,RightUpperArm
        [18,16 ], #LeftHand,LeftLowerArm
        [19,17 ], #RightHand,RightLowerArm
        [20,5 ], #LeftToes,LeftFoot
        [21,6 ], #RightToes,RightFoot
        [22,18 ], #LeftThumbProximal,LeftHand
        [23,22 ], #LeftThumbIntermediate,LeftThumbProximal
        [24,23 ], #LeftThumbDistal,LeftThumbIntermediate
        [25,18 ], #LeftIndexProximal,LeftHand
        [26,25 ], #LeftIndexIntermediate,LeftIndexProximal
        [27,26 ], #LeftIndexDistal,LeftIndexIntermediate
        [28,18 ], #LeftMiddleProximal,LeftHand
        [29,28 ], #LeftMiddleIntermediate,LeftMiddleProximal
        [30,29 ], #LeftMiddleDistal,LeftMiddleIntermediate
        [31,18 ], #LeftRingProximal,LeftHand
        [32,31 ], #LeftRingIntermediate,LeftRingProximal
        [33,32 ], #LeftRingDistal,LeftRingIntermediate
        [34,18 ], #LeftLittleProximal,LeftHand
        [35,34 ], #LeftLittleIntermediate,LeftLittleProximal
        [36,35 ], #LeftLittleDistal,LeftLittleIntermediate
        [37,19 ], #RightThumbProximal,RightHand
        [38,37 ], #RightThumbIntermediate,RightThumbProximal
        [39,38 ], #RightThumbDistal,RightThumbIntermediate
        [40,19 ], #RightIndexProximal,RightHand
        [41,40 ], #RightIndexIntermediate,RightIndexProximal
        [42,41 ], #RightIndexDistal,RightIndexIntermediate
        [43,19 ], #RightMiddleProximal,RightHand
        [44,43 ], #RightMiddleIntermediate,RightMiddleProximal
        [45,44 ], #RightMiddleDistal,RightMiddleIntermediate
        [46,19 ], #RightRingProximal,RightHand
        [47,46 ], #RightRingIntermediate,RightRingProximal
        [48,47 ], #RightRingDistal,RightRingIntermediate
        [49,19 ], #RightLittleProximal,RightHand
        [50,49 ], #RightLittleIntermediate,RightLittleProximal
        [51,50 ], #RightLittleDistal,RightLittleIntermediate
    ]

    angle_id = [
        [0,6 ],#[LeftUpperLeg <-- Spine --> Hips ],
        [1,6 ],#[RightUpperLeg <-- Spine --> Hips ],
        [2,0 ],#[LeftLowerLeg <-- LeftUpperLeg --> Hips ],
        [3,1 ],#[RightLowerLeg <-- RightUpperLeg --> Hips ],
        [4,2 ],#[LeftFoot <-- LeftLowerLeg --> LeftUpperLeg ],
        [5,3 ],#[RightFoot <-- RightLowerLeg --> RightUpperLeg ],
        [6,0 ],#[Spine <-- LeftUpperLeg --> Hips ],
        [7,6 ],#[Chest <-- Spine --> Hips ],
        [8,7 ],#[UpperChest <-- Chest --> Spine ],
        [9,8 ],#[Neck <-- UpperChest --> Chest ],
        [10,9 ],#[Head <-- Neck --> UpperChest ],
        [11,8 ],#[LeftShoulder <-- UpperChest --> Chest ],
        [12,8 ],#[RightShoulder <-- UpperChest --> Chest ],
        [13,11 ],#[LeftUpperArm <-- LeftShoulder --> UpperChest ],
        [14,12 ],#[RightUpperArm <-- RightShoulder --> UpperChest ],
        [15,13 ],#[LeftLowerArm <-- LeftUpperArm --> LeftShoulder ],
        [16,14 ],#[RightLowerArm <-- RightUpperArm --> RightShoulder ],
        [17,15 ],#[LeftHand <-- LeftLowerArm --> LeftUpperArm ],
        [18,16 ],#[RightHand <-- RightLowerArm --> RightUpperArm ],
        [19,4 ],#[LeftToes <-- LeftFoot --> LeftLowerLeg ],
        [20,5 ],#[RightToes <-- RightFoot --> RightLowerLeg ],
        [21,17 ],#[LeftThumbProximal <-- LeftHand --> LeftLowerArm ],
        [22,21 ],#[LeftThumbIntermediate <-- LeftThumbProximal --> LeftHand ],
        [23,22 ],#[LeftThumbDistal <-- LeftThumbIntermediate --> LeftThumbProximal ],
        [24,17 ],#[LeftIndexProximal <-- LeftHand --> LeftLowerArm ],
        [25,24 ],#[LeftIndexIntermediate <-- LeftIndexProximal --> LeftHand ],
        [26,25 ],#[LeftIndexDistal <-- LeftIndexIntermediate --> LeftIndexProximal ],
        [27,17 ],#[LeftMiddleProximal <-- LeftHand --> LeftLowerArm ],
        [28,27 ],#[LeftMiddleIntermediate <-- LeftMiddleProximal --> LeftHand ],
        [29,28 ],#[LeftMiddleDistal <-- LeftMiddleIntermediate --> LeftMiddleProximal ],
        [30,17 ],#[LeftRingProximal <-- LeftHand --> LeftLowerArm ],
        [31,30 ],#[LeftRingIntermediate <-- LeftRingProximal --> LeftHand ],
        [32,31 ],#[LeftRingDistal <-- LeftRingIntermediate --> LeftRingProximal ],
        [33,17 ],#[LeftLittleProximal <-- LeftHand --> LeftLowerArm ],
        [34,33 ],#[LeftLittleIntermediate <-- LeftLittleProximal --> LeftHand ],
        [35,34 ],#[LeftLittleDistal <-- LeftLittleIntermediate --> LeftLittleProximal ],
        [36,18 ],#[RightThumbProximal <-- RightHand --> RightLowerArm ],
        [37,36 ],#[RightThumbIntermediate <-- RightThumbProximal --> RightHand ],
        [38,37 ],#[RightThumbDistal <-- RightThumbIntermediate --> RightThumbProximal ],
        [39,18 ],#[RightIndexProximal <-- RightHand --> RightLowerArm ],
        [40,39 ],#[RightIndexIntermediate <-- RightIndexProximal --> RightHand ],
        [41,40 ],#[RightIndexDistal <-- RightIndexIntermediate --> RightIndexProximal ],
        [42,18 ],#[RightMiddleProximal <-- RightHand --> RightLowerArm ],
        [43,42 ],#[RightMiddleIntermediate <-- RightMiddleProximal --> RightHand ],
        [44,43 ],#[RightMiddleDistal <-- RightMiddleIntermediate --> RightMiddleProximal ],
        [45,18 ],#[RightRingProximal <-- RightHand --> RightLowerArm ],
        [46,45 ],#[RightRingIntermediate <-- RightRingProximal --> RightHand ],
        [47,46 ],#[RightRingDistal <-- RightRingIntermediate --> RightRingProximal ],
        [48,18 ],#[RightLittleProximal <-- RightHand --> RightLowerArm ],
        [49,48 ],#[RightLittleIntermediate <-- RightLittleProximal --> RightHand ],
        [50,49 ],#[RightLittleDistal <-- RightLittleIntermediate --> RightLittleProximal ],
    ]
    eps = 1e-7
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,1,:]-limbs[:,:,:,0,:]
    angles = limbs[:,:,angle_id,:]
    angle_cos = F.cosine_similarity(angles[:,:,:,0,:], angles[:,:,:,1,:], dim=-1)
    return torch.acos(angle_cos.clamp(-1+eps, 1-eps)) 

def loss_angle_unity(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_angles_x = get_angles_unity(x)
    limb_angles_gt = get_angles_unity(gt)
    return nn.L1Loss()(limb_angles_x, limb_angles_gt)

def loss_angle_velocity_unity(x, gt):
    """
    Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert x.shape == gt.shape
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    x_a = get_angles_unity(x)
    gt_a = get_angles_unity(gt)
    x_av = x_a[:,1:] - x_a[:,:-1]
    gt_av = gt_a[:,1:] - gt_a[:,:-1]
    return nn.L1Loss()(x_av, gt_av)