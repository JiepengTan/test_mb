#!/bin/bash

dir_name=../unity_anim_1.0.zip
if [ $# -ge 1 ]; then
    dir_name=$1
fi

if [ !  -f $dir_name ]; then
    echo "The file $dir_name not exist."
    exit 1
fi

ps -ef | grep python | awk '{print $2}' | xargs kill -9
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
rm -rf ./checkpoint/pose3d/ft_unity/logs/
rm -rf ./examples/train_result

rm -rf ./unity_data/
unzip  $dir_name -d ./unity_data/


# 3. preprocess the unity dataset
python tools/convert_unity2pkl.py ./unity_data/ ./data/motion3d/unity

python tools/render_infer_result.py &
./train_pose3d.sh

echo "====start result render===="
python tools/render_infer_result.py &

#python tools/convert_unity2pkl.py ./../../MotionGen/Unity/Output/MotionBERT/Anim/ ./data/motion3d/unity