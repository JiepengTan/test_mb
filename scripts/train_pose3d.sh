#!/bin/bash
pip install pyyaml
pip install numpy==1.23.0

dir_name=ft_unity
if [ $# -ge 1 ]; then
    dir_name=$1
fi

dir_path=./checkpoint/pose3d/$dir_name
mkdir -p $dir_path


ps -ef | grep train.py | awk '{print $2}' | xargs kill -9

nohup_train_log=$dir_path/nohup_train.out

if [ -f $nohup_train_log ]; then
    echo "The file $nohup_train_log exist."
    rm -rf $nohup_train_log
fi

# start train task
echo "==============start train $dir_path =================="

nohup python train.py --config configs/pose3d/MB_ft_unity.yaml --pretrained checkpoint/pretrain/MB_release --checkpoint checkpoint/pose3d/${dir_name} > ${nohup_train_log}  2>&1 &


echo "==============start tensorboard =================="
# start tensorboard
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
nohup tensorboard --port 6007 --logdir $dir_path/logs/ &


# view the train logs
tail -f $nohup_train_log

#ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
#nohup tensorboard --port 6007 --logdir ./checkpoint/pose3d/ft_unity/logs/ &
# tail -f ./checkpoint/pose3d/ft_unity/nohup_train.out
