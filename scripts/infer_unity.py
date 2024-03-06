import os
import numpy as np
import argparse
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
from lib.utils.vismo import render_and_save

from lib.model.loss import *
from lib.model.loss_unity import *
from lib.utils.utils_smpl import *
from lib.model.model_unity import UnityRegressor
import json

def run_3d_pose_estimation(opts):
    args = get_config(opts.config)
    model_backbone = load_backbone(args)
    model = UnityRegressor(args, backbone=model_backbone, dim_rep=args.dim_rep, hidden_dim=args.hidden_dim, dropout_ratio=args.dropout, num_joints=args.num_joints)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()

    checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    testloader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True,
            'drop_last': False
    }
    if opts.unity:
        fps_in = 60
        vid_size =[1920,1080]
    else:
        vid = imageio.get_reader(opts.vid_path,  'ffmpeg')
        fps_in = vid.get_meta_data()['fps']
        vid_size = vid.get_meta_data()['size']
        
    os.makedirs(opts.out_path, exist_ok=True)

    if opts.pixel:
        # Keep relative scale with pixel coornidates
        wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
    else:
        # Scale to [-1,1]
        wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)

    test_loader = DataLoader(wild_dataset, **testloader_params)

    results_all = []
    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            preds = model(batch_input)[-1]
            pred_dirs = preds['dir_fu'].reshape(-1, 6)
            results_all.append(pred_dirs.cpu().numpy())
            
    results_all = np.concatenate(results_all)
    if opts.render : 
        render_and_save(results_all, '%s/X3D.mp4' % (opts.out_path), keep_imgs=False, fps=fps_in)
    np.save('%s/X3D.npy' % (opts.out_path), results_all)
    with open('%s/X3D.json' % (opts.out_path), 'w') as f:
        json.dump(results_all.flatten().tolist(), f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    parser.add_argument('--render', type=bool, default=False, help='render it')
    parser.add_argument('--unity', type=bool, default=False, help='render it')
    opts = parser.parse_args()
    return opts

# This is how you would call the function from another script
if __name__ == "__main__":
    opts = parse_args()
    run_3d_pose_estimation(opts)