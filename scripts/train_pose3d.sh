
#!/bin/bash
pip install pyyaml
pip install numpy==1.23.0

is_debug=0
if [ $# -ge 1 ]; then
    is_debug=$1
fi

echo $is_debug

dir_name=ft_unity

dir_path=./checkpoint/pose3d/$dir_name
#rm -rf $dir_path
mkdir -p $dir_path

ps -ef | grep python | awk '{print $2}' | xargs kill -9

nohup_train_log=$dir_path/nohup_train.out

if [ -f $nohup_train_log ]; then
    echo "The file $nohup_train_log exist."
    rm -rf $nohup_train_log
fi

# start train task
echo "==============start train $dir_path =================="

nohup python train.py --resume "True"  --debug ${is_debug} --config configs/pose3d/MB_train_unity.yaml --checkpoint checkpoint/pose3d/${dir_name} > ${nohup_train_log}  2>&1 &


echo "==============start tensorboard =================="
# start tensorboard
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
nohup tensorboard --port 6007 --logdir $dir_path/logs/ &

echo "====start result render===="
rm -rf ./examples/train_result/
nohup python tools/render_infer_result.py &

# view the train logs
tail -f $nohup_train_log

# tail -f ./checkpoint/pose3d/ft_unity/nohup_train.out
