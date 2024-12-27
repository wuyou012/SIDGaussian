CUDA_VISIBLE_DEVICES=4 python train_mip.py  --source_path /home/alfredchan/work3/dataset/pre_processed_mipnerf360_data_copy/bonsai --model_path out_mip/mip0/bonsai  --eval  --n_views 24  &
CUDA_VISIBLE_DEVICES=5 python train_mip.py  --source_path /home/alfredchan/work3/dataset/pre_processed_mipnerf360_data_copy/counter --model_path out_mip/mip0/counter  --eval  --n_views 24  &
CUDA_VISIBLE_DEVICES=6 python train_mip.py  --source_path /home/alfredchan/work3/dataset/pre_processed_mipnerf360_data_copy/garden --model_path out_mip/mip0/garden  --eval  --n_views 24  &
CUDA_VISIBLE_DEVICES=7 python train_mip.py  --source_path /home/alfredchan/work3/dataset/pre_processed_mipnerf360_data_copy/kitchen --model_path out_mip/mip0/kitchen  --eval  --n_views 24  &
wait;
CUDA_VISIBLE_DEVICES=4 python train_mip.py  --source_path /home/alfredchan/work3/dataset/pre_processed_mipnerf360_data_copy/room --model_path out_mip/mip0/room  --eval  --n_views 24  &
CUDA_VISIBLE_DEVICES=5 python train_mip.py  --source_path /home/alfredchan/work3/dataset/pre_processed_mipnerf360_data_copy/stump --model_path out_mip/mip0/stump  --eval  --n_views 24  &
CUDA_VISIBLE_DEVICES=6 python train_mip.py  --source_path /home/alfredchan/work3/dataset/pre_processed_mipnerf360_data_copy/bicycle --model_path out_mip/mip0/bicycle  --eval  --n_views 24  &
wait;