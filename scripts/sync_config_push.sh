#!/bin/bash

dst_dir=~/projects/motionbert_sync/
scirpt_dir=$dst_dir/scripts
config_dir=$dst_dir/configs
rm -rf $scirpt_dir
rm -rf $config_dir

mkdir $dst_dir/scripts
cp ./*.sh $scirpt_dir
cp ./*.py $scirpt_dir

cp -r ./configs $dst_dir


# auto sync to git 
WORKDIR=$dst_dir
cd $WORKDIR
if [ ! -d ".git" ]; then
    echo "Error: Directory $WORKDIR is not a git repository."
    exit 1
fi
git add .
git commit -m "Auto commit: $(date)"
git push origin main --force
cd -