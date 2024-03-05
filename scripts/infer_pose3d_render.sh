#!/bin/bash

python tools/render_infer_result.py

echo "start merge videos => ./examples/merged.mp4"
ffmpeg \
  -i ./examples/train_result/0.mp4 -i ./examples/train_result/5.mp4 -i ./examples/train_result/10.mp4 \
  -i ./examples/train_result/15.mp4 -i ./examples/train_result/20.mp4 -i ./examples/train_result/25.mp4 \
  -filter_complex "\
  [0:v]scale=320x320[0v]; \
  [1:v]scale=320x320[1v]; \
  [2:v]scale=320x320[2v]; \
  [3:v]scale=320x320[3v]; \
  [4:v]scale=320x320[4v]; \
  [5:v]scale=320x320[5v]; \
  [0v][1v][2v][3v][4v][5v]xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0[out]" \
  -map "[out]" \
  -c:v libx264 \
  ./examples/merged.mp4


