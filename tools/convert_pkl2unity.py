import json
import pickle
import os

dst_dir = ""
def convert_pkl_unity(file_path):
    with open(file_path, 'rb') as f:
        motion_data = pickle.load(f)

    data_input = motion_data.get("data_input", [])
    data_label = motion_data.get("data_label", [])

    data_input_flat = data_input.flatten().tolist() if data_input is not None else []
    data_label_flat = data_label.flatten().tolist() if data_label is not None else []

    motion_bert_clip_dict = {
        'data_input': data_input_flat,
        'data_label': data_label_flat
    }

    json_data = json.dumps(motion_bert_clip_dict, indent=4)

    base_filename = os.path.basename(file_path)
    filename_without_extension = os.path.splitext(base_filename)[0]

    json_file_path = os.path.join(dst_dir, filename_without_extension + '.json')
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_data)


def convert_all(root_path):
    data_path = root_path
    motion_list = sorted(os.listdir(data_path))
    idx = 0
    for flie_name in motion_list:
        idx+=1
        convert_pkl_unity(os.path.join(data_path,flie_name))
        if(idx %100 == 1):
            print("convert count = " + str(idx))
import sys
dst_dir = "./test"
convert_all("./data/motion3d/MB3D_f243s81/H36M-SH/test")