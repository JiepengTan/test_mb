import os
import pickle
import random
import struct
import sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
from lib.utils.tools import read_pkl
import shutil

def parse_unity_data(input_path, output_dir):
    with open(input_path, 'rb') as f:
        startIdx = struct.unpack('i', f.read(4))[0]
        length = struct.unpack('i', f.read(4))[0]
        idx = startIdx
        for _ in range(length):
            data_input_flat = []
            data_input_size = struct.unpack('i', f.read(4))[0]
            for _ in range(data_input_size):
                data_input_flat.append(struct.unpack('f', f.read(4))[0])
            data_label_size = struct.unpack('i', f.read(4))[0]
            data_label_flat = []
            for _ in range(data_label_size):
                data_label_flat.append(struct.unpack('f', f.read(4))[0])

            data_input = np.array(data_input_flat).reshape((243, 17, 3))
            data_label = np.array(data_label_flat).reshape((243, 17, 3))
            motion_bert_clip_dict = {
                'data_input': data_input.tolist(),
                'data_label': data_label.tolist()
            }
            output_path = os.path.join(output_dir, "%08d.pkl" % idx)
            with open(output_path, "wb") as myprofile:  
                pickle.dump(motion_bert_clip_dict, myprofile)
            if(idx %100 == 0):
                print("write file " + output_path)
            idx = idx +1

def test_read(input_dir, idx):
    motion_file = read_pkl(os.path.join(input_dir, "%08d.pkl" % idx))
    motion_3d = motion_file["data_label"]  
    data3d = torch.FloatTensor(motion_3d)
    motion_2d = motion_file["data_input"]       

    data2d = torch.FloatTensor(motion_2d)
    print(data2d.shape)
    print(data3d.shape)     

def move_file(dst_dir, src_idx, dst_idx):
    src_path =os.path.join(input_dir,"" "%08d.pkl" % src_idx)
    dst_path =os.path.join(dst_dir, "%08d.pkl" % dst_idx)
    shutil.move(src_path, dst_path)
    print(src_path + " => " + dst_path)

def copy_test_train_set(input_dir, output_dir,test_rate):
    total_count = len([name for name in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, name))])
    print("copy count " + str(total_count))
    train_dir = os.path.join(output_dir ,"train")
    test_dir = os.path.join(output_dir ,"test")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    train_idx = 0
    test_idx =0

    import random
    for idx in range(total_count):
        if random.random() <= test_rate :
            move_file(test_dir,idx,test_idx) 
            test_idx +=1
        else :
            move_file(train_dir,idx,train_idx) 
            train_idx +=1
        

max_data_count = 0
def convert_all(root_path, output_dir):
    os.makedirs(output_dir)
    data_path = root_path
    motion_list = sorted(os.listdir(data_path))
    print("total count " + str(len(motion_list) * 1000))
    idx = 0
    for flie_name in motion_list:
        idx += 1
        if(max_data_count >0 and idx > max_data_count) :
            return
        parse_unity_data(os.path.join(data_path,flie_name),output_dir)


input_dir = "../../MotionGen/Unity/Output/MotionBERT/Anim/"
output_dir = "data/motion3d/unity/"
is_only_copy = len(sys.argv) >= 4
if len(sys.argv) >= 3:
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
elif len(sys.argv) != 1:
    print("Usage: script.py <input_dir> <output_dir>")
    print(f"eg: script.py {input_dir} {output_dir}")
    exit()

print(f"input_dir= {input_dir}    output_dir = {output_dir}")

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

tmp_dir = os.path.join(output_dir, "tmp")

if not is_only_copy :
    print('===== convert from unity ==========')
    #convert_all(input_dir, tmp_dir)

print('===== create test train dataset ==========')
copy_test_train_set(tmp_dir, output_dir, 0.15)

#shutil.rmtree(tmp_dir)
print("done")