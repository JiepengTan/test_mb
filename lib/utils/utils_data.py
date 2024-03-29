import os
import torch
import torch.nn.functional as F
import numpy as np
import copy

# 将动作的所有坐标都被标准化到[-1, 1]的范围(基于当前动作的 boundbox )
def crop_scale(motion, scale_range=[1, 1]):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]!=0][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result

# 将动作的所有坐标都标准化到[-1, 1]的范围(基于当前动作的 boundbox )
def crop_scale_3d(motion, scale_range=[1, 1]):
    '''
        Motion: [T, 17, 3]. (x, y, z)
        Normalize to [-1, 1]
        Z is relative to the first frame's root.
    '''
    result = copy.deepcopy(motion)
    result[:,:,2] = result[:,:,2] - result[0,0,2]
    xmin = np.min(motion[...,0])
    xmax = np.max(motion[...,0])
    ymin = np.min(motion[...,1])
    ymax = np.max(motion[...,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) / ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
    result[...,2] = result[...,2] / scale
    result = (result - 0.5) * 2
    return result

# 水平翻转：  X *=-1 && 左右手脚index 交换
def flip_data(data):
    """
    horizontal flip
        data: [N, F, 17, D] or [F, 17, D]. X (horizontal coordinate) is the first channel in D.
    Return
        result: same
    """

    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]
    if(data.shape[-2]== 52) :
        left_joints = [
            1,# LeftUpperLeg
            3,# LeftLowerLeg
            5,# LeftFoot
            12,# LeftShoulder
            14,# LeftUpperArm
            16,# LeftLowerArm
            18,# LeftHand
            20,# LeftToes
            22,# LeftThumbProximal
            23,# LeftThumbIntermediate
            24,# LeftThumbDistal
            25,# LeftIndexProximal
            26,# LeftIndexIntermediate
            27,# LeftIndexDistal
            28,# LeftMiddleProximal
            29,# LeftMiddleIntermediate
            30,# LeftMiddleDistal
            31,# LeftRingProximal
            32,# LeftRingIntermediate
            33,# LeftRingDistal
            34,# LeftLittleProximal
            35,# LeftLittleIntermediate
            36,# LeftLittleDistal
        ]
        right_joints = [
            2,# RightUpperLeg
            4,# RightLowerLeg
            6,# RightFoot
            13,# RightShoulder
            15,# RightUpperArm
            17,# RightLowerArm
            19,# RightHand
            21,# RightToes
            37,# RightThumbProximal
            38,# RightThumbIntermediate
            39,# RightThumbDistal
            40,# RightIndexProximal
            41,# RightIndexIntermediate
            42,# RightIndexDistal
            43,# RightMiddleProximal
            44,# RightMiddleIntermediate
            45,# RightMiddleDistal
            46,# RightRingProximal
            47,# RightRingIntermediate
            48,# RightRingDistal
            49,# RightLittleProximal
            50,# RightLittleIntermediate
            51,# RightLittleDistal
        ]
    if(data.shape[-2]== 22) :
        left_joints = [
            1,# LeftUpperLeg
            3,# LeftLowerLeg
            5,# LeftFoot
            12,# LeftShoulder
            14,# LeftUpperArm
            16,# LeftLowerArm
            18,# LeftHand
            20,# LeftToes
        ]
        right_joints = [
            2,# RightUpperLeg
            4,# RightLowerLeg
            6,# RightFoot
            13,# RightShoulder
            15,# RightUpperArm
            17,# RightLowerArm
            19,# RightHand
            21,# RightToes
        ]

    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1                                               # flip x of all joints
    flipped_data[..., left_joints+right_joints, :] = flipped_data[..., right_joints+left_joints, :]
    return flipped_data


# 帧采样
def resample(ori_len, target_len, replay=False, randomness=True):
    if replay:
        # [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        if ori_len > target_len:
            st = np.random.randint(ori_len-target_len)
            return range(st, st+target_len)  # Random clipping from sequence
        else:
            return np.array(range(target_len)) % ori_len  # Replay padding
    else:
        if randomness:
            # eg: np.linspace 会生成 [0, 0.5, 1, 1.5, ..., 4.5]。
            # 然后，对于每个数，函数会随机选择其下界或上界，最终得到的序列可能是 
            # [0, 1, 1, 1, 2, 3, 3, 4, 4, 5]
            even = np.linspace(0, ori_len, num=target_len, endpoint=False)
            if ori_len < target_len:
                low = np.floor(even)
                high = np.ceil(even)
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel*low+(1-sel)*high)
            else:
                interval = even[1] - even[0]
                result = np.random.random(even.shape)*interval + even
                #[0.5, 2.3, 4.1, 6.7, 9.1]
                #[0, 2, 4, 6, 9]
            result = np.clip(result, a_min=0, a_max=ori_len-1).astype(np.uint32)
        else:
            # [0, 2, 4, 6, 8]
            result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result

# 数据切片，不够一个片段的，就采样补齐
def split_clips(vid_list, n_frames, data_stride):
    result = []
    n_clips = 0
    st = 0
    i = 0
    saved = set()
    while i<len(vid_list):
        i += 1
        if i-st == n_frames:
            result.append(range(st,i))
            saved.add(vid_list[i-1])
            st = st + data_stride
            n_clips += 1
        if i==len(vid_list):
            break
        if vid_list[i]!=vid_list[i-1]: 
            # 最后一组不够的时候 ，调用resample 进行帧补充
            if not (vid_list[i-1] in saved):
                resampled = resample(i-st, n_frames) + st
                result.append(resampled)
                saved.add(vid_list[i-1])
            st = i
    return result