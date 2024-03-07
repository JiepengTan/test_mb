#!/bin/bash

# 定义帧率和分辨率
fps=60
resolution="768x432"
input_dir="../../MotionGen/Unity/Output/Recorder"
output_dir="../../MotionGen/Unity/Output/Recorder"

# 确保输出目录存在
mkdir -p $output_dir

# 遍历input目录下的每个数字命名的子目录
for dir in ${input_dir}/[0-9]*/ ; do
    # 检查目录名是否为数字
    if [[ $(basename $dir) =~ ^[0-9]+/?$ ]]; then
        # 去除目录名末尾的斜杠，得到视频文件名
        dirname=$(basename $dir)
        # 使用ffmpeg将图片合成视频并指定分辨率，输出到指定的output目录
        ffmpeg -framerate $fps -i "${dir}%04d.png" -s $resolution -c:v libx264 -pix_fmt yuv420p "${output_dir}/${dirname}.mp4" -y
    fi
done

echo "所有视频已经生成完毕。"

echo "start merge videos => ./examples/merged.mp4"
ffmpeg \
  -i ${input_dir}/0.mp4 -i ${input_dir}/5.mp4 -i ${input_dir}/4.mp4 \
  -i ${input_dir}/3.mp4 -i ${input_dir}/2.mp4 -i ${input_dir}/1.mp4 \
  -filter_complex "\
  [0:v]scale=768x432[0v]; \
  [1:v]scale=768x432[1v]; \
  [2:v]scale=768x432[2v]; \
  [3:v]scale=768x432[3v]; \
  [4:v]scale=768x432[4v]; \
  [5:v]scale=768x432[5v]; \
  [0v][1v][2v][3v][4v][5v]xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0[out]" \
  -map "[out]" \
  -c:v libx264 \
  ${output_dir}/merged.mp4 -y

  open  ${output_dir}/merged.mp4