#!/bin/bash
dst_dir=/root/autodl-tmp/MotionBERT
if [ ! -d $dst_dir ]; then
  echo "The directory $dst_dir does not exist."
  exit 1
fi

cd $dst_dir

WORKDIR="_remote_config"
mkdir -p $WORKDIR
cd $WORKDIR
if [ ! -d ".git" ]; then
    git clone https://github.com/JiepengTan/test_mb.git ./
fi
git pull origin main
cd -

cp -rf $WORKDIR/scripts/* .

dirs=("configs" "tools" "lib")

for dir in "${dirs[@]}"; do
  rm -rf "./$dir"
  cp -rf "$WORKDIR/$dir" .
done


cp -rf $WORKDIR/configs/test.json ./examples/test.json 
