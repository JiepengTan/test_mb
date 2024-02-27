#!/bin/bash
cd /root/autodl-tmp/MotionBERT
WORKDIR="_remote_config"
mkdir -p $WORKDIR
cd $WORKDIR
if [ ! -d ".git" ]; then
    git clone https://github.com/JiepengTan/test_mb.git ./
fi
git pull origin main
cd -

cp -rf $WORKDIR/scripts/* .

rm -rf ./config
cp -rf $WORKDIR/configs .


