for SCENE in bonsai counter garden kitchen room stump bicycle
do
  CUDA_VISIBLE_DEVICES=0 python train.py --source_path  /home/data/mipnerf360/$SCENE --model_path output/mip360/$SCENE --eval --n_views 24 --D 0.1 --W 0.25 --N 0.1
done