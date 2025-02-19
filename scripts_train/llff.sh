for SCENE in fern flower fortress horns leaves orchids room trex
do
  CUDA_VISIBLE_DEVICES=0 python train.py --source_path /home/data/nerf_llff_data/$SCENE --model_path output/llff/$SCENE --eval --n_views 3 --sample_pseudo_interval 1 --D 0.8 --W 0.5 --N 1
done