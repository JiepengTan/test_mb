#!/bin/bash
#!/bin/bash
pip install pyyaml
pip install numpy==1.23.0

is_debug=0
if [ $# -ge 1 ]; then
    is_debug=$1
fi

echo $is_debug

dir_name=ft_unity
dir_path=./checkpoint/unity/$dir_name
nohup_train_log=$dir_path/nohup_train.out

rm -rf $dir_path
mkdir -p $dir_path

ps -ef | grep python | awk '{print $2}' | xargs kill -9


if [ -f $nohup_train_log ]; then
    echo "The file $nohup_train_log exist."
    rm -rf $nohup_train_log
fi

# start train task
echo "==============start train $dir_path =================="

nohup python train_unity.py --debug ${is_debug} --config configs/unity/ft_unity_rot.yaml --pretrained checkpoint/pretrain/MB_release --checkpoint checkpoint/unity/${dir_name} > ${nohup_train_log}  2>&1 &



echo "==============start tensorboard =================="
# start tensorboard
nohup_tf_log=$dir_path/tf_nohup_.out
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
nohup tensorboard --port 6007 --logdir ${dir_path}/logs/ > ${nohup_tf_log}  2>&1 &

echo "====start result render===="
rm -rf ./examples/train_result/
mkdir -p ./examples/train_result/
nohup_out_log=$dir_path/out_result_.out
#nohup python tools/render_infer_result.py > ${nohup_out_log}  2>&1 &

# view the train logs
tail -f $nohup_train_log



#python train_unity.py --config configs/unity/ft_unity_rot.yaml --pretrained checkpoint/pretrain/MB_release --checkpoint checkpoint/unity/ft_unity
#ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
#nohup tensorboard --port 6007 --logdir ./checkpoint/unity/ft_unity/logs/ >  ./checkpoint/unity/ft_unity/tf_nohup_.out 2>&1 &