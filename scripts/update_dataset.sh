#!/bin/bash

is_need_train=1
if [ $# -ge 1 ]; then
    is_need_train=0
fi

dir_name=../unity_anim_1.0.zip

if [ !  -f $dir_name ]; then
    echo "The file $dir_name not exist."
    exit 1
fi

ps -ef | grep python | awk '{print $2}' | xargs kill -9

rm -rf ./unity_data/
unzip  $dir_name -d ./unity_data/


# 3. preprocess the unity dataset
python tools/convert_unity2pkl.py ./unity_data/ ./data/unity/unity

if [ $is_need_train -eq 1 ]; then
    ./train_pose3d.sh
fi


#python tools/convert_unity2pkl.py ./../../MotionGen/Unity/Output/MotionBERT/Anim/ ./data/motion3d/unity