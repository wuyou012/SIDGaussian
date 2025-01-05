for SCENE in bonsai counter garden kitchen room stump bicycle
do
  CUDA_VISIBLE_DEVICES=0 python train.py --source_path  /home/data1/mipnerf360/$SCENE --model_path output_mip/output_mip_1/$SCENE --eval --n_views 24 --D 0.1 --W 0.25 --N 0.1
done