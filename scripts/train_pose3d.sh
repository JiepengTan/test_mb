#!/bin/bash
pip install pyyaml
pip install numpy==1.23.0

dir_name=ft_wp3
if [ $# -ge 1 ]; then
    dir_name=$1
fi

dir_path=./checkpoint/pose3d/$dir_name
mkdir -p $dir_path


echo hello
ps -ef | grep python | awk '{print $2}' | xargs kill -9

nohup_train_log=$dir_path/nohup_train.out

if [ -f $nohup_train_log ]; then
    echo "The file $nohup_train_log exist."
    rm -rf $nohup_train_log
fi

# start train task
echo "==============start train $dir_path =================="

nohup python train.py --config configs/pose3d/MB_ft_h36m.yaml --pretrained checkpoint/pretrain/MB_release --checkpoint checkpoint/pose3d/${dir_name} > ${nohup_train_log}  2>&1 &


echo "==============start tensorboard =================="
# start tensorboard
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
nohup tensorboard --port 6007 --logdir ${dir_nadir_pathme}/logs/ &


# view the train logs
tail -f $nohup_train_log

# ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
# nohup tensorboard --port 6007 --logdir ./checkpoint/pose3d/ft_wp3/logs/ &
