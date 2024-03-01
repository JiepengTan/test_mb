
import os
import numpy as np
import argparse
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
import sys
sys.path.insert(0, os.getcwd())
from lib.utils.vismo import render_and_save


fps_in = 30
def render_all_epoch():
    input_dir  = "./examples/train_result"
    output_dir = input_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    motion_list = [file for file in sorted(os.listdir(input_dir)) if file.endswith('.npy')]
    count = len(motion_list)
    idx = 0
    for flie_name in motion_list:
        idx += 1
        path = os.path.join(input_dir,flie_name)
        print(f"{idx}/{count} {path}")
        results_all = np.load(path)
        render_and_save(results_all, '%s/%s.mp4' % (output_dir, flie_name), keep_imgs=False, fps=fps_in)


# This is how you would call the function from another script
if __name__ == "__main__":
    render_all_epoch()
    