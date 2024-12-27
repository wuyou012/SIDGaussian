CUDA_VISIBLE_DEVICES=0 python train_final.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/fern --model_path output_eval/fern --eval --n_views 3 --sample_pseudo_interval 1 &
CUDA_VISIBLE_DEVICES=1 python train_final.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/flower --model_path output_eval/flower --eval --n_views 3 --sample_pseudo_interval 1 &
CUDA_VISIBLE_DEVICES=2 python train_final.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/fortress --model_path output_eval/fortress --eval --n_views 3 --sample_pseudo_interval 1 &
CUDA_VISIBLE_DEVICES=3 python train_final.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/horns --model_path output_eval/horns --eval --n_views 3 --sample_pseudo_interval 1 &
wait;
CUDA_VISIBLE_DEVICES=0 python train_final.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/leaves --model_path output_eval/leaves --eval --n_views 3 --sample_pseudo_interval 1 &
CUDA_VISIBLE_DEVICES=1 python train_final.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/orchids --model_path output_eval/orchids --eval --n_views 3 --sample_pseudo_interval 1 &
CUDA_VISIBLE_DEVICES=2 python train_final.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/room --model_path output_eval/room --eval --n_views 3 --sample_pseudo_interval 1 &
CUDA_VISIBLE_DEVICES=3 python train_final.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/trex --model_path output_eval/trex --eval --n_views 3 --sample_pseudo_interval 1 &
wait;
#CUDA_VISIBLE_DEVICES=0 python render.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/fern --model_path output_oblation/output_A2D0/fern --iteration 10000 --render_depth&
#CUDA_VISIBLE_DEVICES=1 python render.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/flower --model_path output_oblation/output_A2D0/flower --iteration 10000 --render_depth&
#CUDA_VISIBLE_DEVICES=2 python render.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/fortress --model_path output_oblation/output_A2D0/fortress --iteration 10000 --render_depth&
#CUDA_VISIBLE_DEVICES=3 python render.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/horns --model_path output_oblation/output_A2D0/horns --iteration 10000 --render_depth&
#CUDA_VISIBLE_DEVICES=4 python render.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/leaves --model_path output_oblation/output_A2D0/leaves --iteration 10000 --render_depth&
#CUDA_VISIBLE_DEVICES=5 python render.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/orchids --model_path output_oblation/output_A2D0/orchids --iteration 10000 --render_depth&
#CUDA_VISIBLE_DEVICES=6 python render.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/room --model_path output_oblation/output_A2D0/room --iteration 10000 --render_depth&
#CUDA_VISIBLE_DEVICES=7 python render.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/trex --model_path output_oblation/output_A2D0/trex --iteration 10000 --render_depth&
#wait;

# CUDA_VISIBLE_DEVICES=0 python metrics.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/fern --model_path output_oblation/output_A2D0fern --iteration 10000 &
# CUDA_VISIBLE_DEVICES=1 python metrics.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/flower --model_path output_oblation/output_A2D0flower --iteration 10000 &
# CUDA_VISIBLE_DEVICES=2 python metrics.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/fortress --model_path output_oblation/output_A2D0fortress --iteration 10000 &
# CUDA_VISIBLE_DEVICES=3 python metrics.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/horns --model_path output_oblation/output_A2D0horns --iteration 10000 &
# CUDA_VISIBLE_DEVICES=4 python metrics.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/leaves --model_path output_oblation/output_A2D0leaves --iteration 10000 &
# CUDA_VISIBLE_DEVICES=5 python metrics.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/orchids --model_path output_oblation/output_A2D0orchids --iteration 10000 &
# CUDA_VISIBLE_DEVICES=6 python metrics.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/room --model_path output_oblation/output_A2D0room --iteration 10000 &
# CUDA_VISIBLE_DEVICES=7 python metrics.py --source_path /home/alfredchan/work3/dataset/pre_processed_nerf_llff_data/trex --model_path output_oblation/output_A2D0trex --iteration 10000 &
# wait;

