
import os
import numpy as np
import argparse
import time
import sys
sys.path.insert(0, os.getcwd())
from lib.utils.vismo import render_and_save

fps_in = 30
max_epoch = 60
def render_all_epoch():
    input_dir  = "./examples/train_result"
    output_dir = input_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    idx = 0
    for idx in range(max_epoch):
        path = os.path.join(input_dir,f"{idx}.npy")
        next_path = os.path.join(input_dir,f"{idx+1}.npy")
        if(idx == max_epoch-1):
            next_path = path
        while(not os.path.exists(next_path)) :
            print("wait " + next_path)
            time.sleep(10)

        results_all = np.load(path)
        render_and_save(results_all, '%s/%s.mp4' % (output_dir, str(idx)), keep_imgs=False, fps=fps_in)

# This is how you would call the function from another script
if __name__ == "__main__":
    render_all_epoch()
    