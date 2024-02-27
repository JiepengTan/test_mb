#!/bin/bash

WORKDIR="./test"
mkdir -p $WORKDIR
cd $WORKDIR
if [ ! -d ".git" ]; then
    git clone https://github.com/JiepengTan/test_mb.git ./
fi
git pull origin main
cd -
echo done
exit 1
cp -rf $WORKDIR/scripts/* .

rm -rf ./config
cp -rf $WORKDIR/configs .


