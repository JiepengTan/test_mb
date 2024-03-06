#!/bin/bash

epoch='1'
if [ $# -ge 1 ]; then
    epoch=$1
fi

python infer_unity.py --vid_path ./examples/test.mp4 --json_path ./examples/test.json --out_path ./examples/output \
    --config configs/unity/ft_unity_rot.yaml --evaluate checkpoint/unity/ft_unity/latest_epoch.bin\
    --unity true

dst_dir=./examples/train_result
mkdir -p $dst_dir

dst_path=$dst_dir/$epoch.npy
mv ./examples/output/X3D.npy $dst_path
echo save to  $dst_path

dst_path=$dst_dir/$epoch.json
mv ./examples/output/X3D.json $dst_path
echo save to  $dst_path