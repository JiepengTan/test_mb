#!/bin/bash

dir_name=../unity_anim_1.0.zip
if [ $# -ge 1 ]; then
    dir_name=$1
fi

ps -ef | grep python | awk '{print $2}' | xargs kill -9
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
rm -rf ./checkpoint/pose3d/ft_unity/logs/

rm -rf ./unity_data/
unzip  $dir_name -d ./unity_data/
rm $dir_name

# 3. preprocess the unity dataset
python tools/convert_unity2pkl.py ./unity_data/ ./data/motion3d/unity

./train_pose3d.sh