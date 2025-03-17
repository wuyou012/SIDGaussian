#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
OUTPUT_DIR="./out_video"
mkdir -p $OUTPUT_DIR
for scene in flower horns fern
do
  echo "processing: $scene"
  OUTPUT="$OUTPUT_DIR/${scene}_comparison.mp4"

    python render_compare.py \
    --folder1 "/home/SIDGaussian/output/3dgs/$scene/video/ours_30000" \
    --folder2 "/home/SIDGaussian/output/output_A2D2C2_07_final_2071/$scene/video/ours_12000" \
    --output "$OUTPUT" \
    --fps 15 \
    --gif \
    --gif-quality 70 \

  echo "end processing: $OUTPUT"
done