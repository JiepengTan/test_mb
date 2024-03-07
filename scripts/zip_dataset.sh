#!/bin/bash
cur_dir=`pwd`
src_dir=../../MotionGen/Unity/Output/MotionBERT/Anim/
dst_file=unity_anim_1.0.zip
cd $src_dir

zip $dst_file *.mdata
cd $cur_dir

pwd
rm ../$dst_file
mv $src_dir/$dst_file ../
