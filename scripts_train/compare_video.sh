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
    --folder2 "/home/SIDGaussian/output/output_final/$scene/video/ours_10000" \
    --output "$OUTPUT" \
    --fps 15

  echo "generate: $OUTPUT"
done
