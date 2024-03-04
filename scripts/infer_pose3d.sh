#!/bin/bash

epoch='1'
if [ $# -ge 1 ]; then
    epoch=$1
fi

python infer_wild.py --vid_path ./examples/test.mp4 --json_path ./examples/test.json --out_path ./examples/output \
    --config configs/pose3d/MB_ft_h36m.yaml --evaluate checkpoint/pose3d/ft_unity/latest_epoch.bin\
    --unity

dst_dir=./examples/train_result
mkdir -p $dst_dir

dst_path=$dst_dir/$epoch.npy
mv ./examples/output/X3D.npy $dst_path
echo save to  $dst_path