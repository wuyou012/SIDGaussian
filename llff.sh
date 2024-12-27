# for SCENE in fern flower fortress horns leaves orchids room trex
# do
#     CUDA_VISIBLE_DEVICES=1 python train_1.py --source_path /home/data1/nerf_llff_data/$SCENE --model_path output_4/$SCENE --eval --n_views 3 --sample_pseudo_interval 1 --w_dino 0.8
#     CUDA_VISIBLE_DEVICES=1 python render.py --source_path /home/data1/nerf_llff_data/$SCENE  --model_path  output_4/$SCENE --iteration 10000
#     CUDA_VISIBLE_DEVICES=1 python metrics.py --source_path /home/data1/nerf_llff_data/$SCENE  --model_path  output_4/$SCENE --iteration 10000
# done

# Train
# for SCENE in fern flower fortress horns leaves orchids room trex
# do
#   python train.py --source_path /home/data1/nerf_llff_data/$SCENE --model_path output/$SCENE --eval --n_views 3 --sample_pseudo_interval 1 --w_dino 1
# done

# Test & Evaluat
# for SCENE in fern flower fortress horns leaves orchids room trex
# do
#   python render.py --source_path /home/data1/nerf_llff_data/$SCENE  --model_path  output/$SCENE --iteration 10000
#   python metrics.py --source_path /home/data1/nerf_llff_data/$SCENE  --model_path  output/$SCENE --iteration 10000
# done
# python score_average.py
# python train.py --source_path /home/data1/nerf_llff_data/fern --model_path output_v3.5/fern --eval --n_views 3 --sample_pseudo_interval 1 --w_dino 0.8

# python render.py --source_path /home/data1/nerf_llff_data/fern  --model_path  output_ADC2_copy/fern --iteration 10000

for SCENE in fern flower fortress horns leaves orchids room trex
do
    # CUDA_VISIBLE_DEVICES=1 python train_1.py --source_path /home/data1/nerf_llff_data/$SCENE --model_path output_4/$SCENE --eval --n_views 3 --sample_pseudo_interval 1 --w_dino 0.8
    # CUDA_VISIBLE_DEVICES=1 python render.py --source_path /home/data1/nerf_llff_data/$SCENE  --model_path  output_4/$SCENE --iteration 10000
    CUDA_VISIBLE_DEVICES=6 python metrics.py --source_path /home/data1/nerf_llff_data/$SCENE  --model_path  output_dngs/$SCENE --iteration 6000
done