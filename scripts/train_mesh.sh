#!/bin/bash
pip install pyyaml
pip install numpy==1.23.0

dir_name=ft_wp
if [ $# -ge 1 ]; then
    dir_name=$1
fi

dir_path=./checkpoint/mesh/$dir_name
mkdir -p $dir_path

echo "train in $dir_path "
rm nohup.out
ps -ef | grep train_mesh | awk '{print $2}' | xargs kill -9

# start train task
nohup python train_mesh.py --config configs/mesh/MB_ft_pw3d.yaml --pretrained checkpoint/pretrain/MB_release --checkpoint checkpoint/mesh/${dir_name} &


echo "start tensorboard "
# start tensorboard
cd ./checkpoint/mesh
rm nohup.out
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
nohup tensorboard --port 6007 --logdir ./${dir_name}/logs/ &
cd ../../

# view the train logs
tail -f nohup.out

