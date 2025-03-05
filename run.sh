device=$1

if [ $device == 0 ]; then
    CUDA_VISIBLE_DEVICES=$device python main.py generate_images\
        --model_type eagle --output_dir generated_images/coco2017_val/lantern_k_10_lambda_5/slice_0\
        --prompt MSCOCO2017Val --slice 0-1666 --num_images 1666 --eagle_version 1 --lantern --lantern_k 10 --lantern_delta 5

elif [ $device == 1 ]; then
    CUDA_VISIBLE_DEVICES=$device python main.py generate_images\
        --model_type eagle --output_dir generated_images/coco2017_val/lantern_k_5_lambda_10/slice_0\
        --prompt MSCOCO2017Val --slice 0-1666 --num_images 1666 --eagle_version 1 --lantern --lantern_k 5 --lantern_delta 10
elif [ $device == 2 ]; then
    CUDA_VISIBLE_DEVICES=$device python main.py generate_images\
        --model_type eagle --output_dir generated_images/coco2017_val/lantern_k_5_lambda_20/slice_0\
        --prompt MSCOCO2017Val --slice 0-1666 --num_images 1666 --eagle_version 1 --lantern --lantern_k 5 --lantern_delta 20
fi

# for cfg_mode in parallel sequential; do
#     CUDA_VISIBLE_DEVICES=$device python main.py generate_images\
#         --model_type base --output_dir generated_images/SJDPrompts/benchmark/base_${cfg_mode}\
#         --prompt SJDPrompts --num_images 10 --set_seed --random_seed 42\
#         --cfg_mode $cfg_mode
# done

# for eagle_version in 1; do
#     for cfg_mode in parallel; do
#         for tree_choices in naive_extend_57 mc_sim_7b_63; do
#             for cfg in 1 3 5 7; do
#                 for drafter_top_k in 1000 2000 4000 8000; do
#                     CUDA_VISIBLE_DEVICES=$device USE_EXPERIMENTAL_FEATURES=1 python main.py generate_images\
#                         --model_type eagle --output_dir generated_images/SJDPrompts/new_benchmark/eagle_v${eagle_version}_${cfg_mode}_${tree_choices}_${cfg}_${drafter_top_k}\
#                         --prompt SJDPrompts --num_images 10 --set_seed --random_seed 42\
#                         --tree_choices $tree_choices\
#                         --eagle_version $eagle_version --cfg_mode $cfg_mode\
#                         --cfg $cfg --drafter_top_k $drafter_top_k
                    
#                     CUDA_VISIBLE_DEVICES=$device USE_EXPERIMENTAL_FEATURES=1 python main.py generate_images\
#                         --model_type eagle --output_dir generated_images/SJDPrompts/new_benchmark/lantern_v${eagle_version}_${cfg_mode}_${tree_choices}_${cfg}_${drafter_top_k}\
#                         --prompt SJDPrompts --num_images 10 --set_seed --random_seed 42\
#                         --tree_choices $tree_choices\
#                         --eagle_version $eagle_version --cfg_mode $cfg_mode\
#                         --cfg $cfg --drafter_top_k $drafter_top_k\
#                         --lantern --lantern_k 10 --lantern_delta 5
#                 done
#             done
#         done
#     done
# done

# if [ $device == 0 ]; then
#     CUDA_VISIBLE_DEVICES=$device python main.py generate_images\
#         --model_type eagle --output_dir generated_images/PartiPrompts/debug_image_test2\
#         --prompt PartiPrompts --num_images 10000 --set_seed --random_seed 42\
#         --tree_choices naive_extend_57 --lantern --lantern_k 10 --lantern_delta 5\
#         --eagle_version 2 --cfg_mode parallel
    
    # CUDA_VISIBLE_DEVICES=$device python main.py generate_images --model_type eagle --output_dir generated_images/PartiPrompts/EAGLEv1 --prompt PartiPrompts --num_images 10000\
    #     --set_seed --random_seed 42 --eagle_v1 --tree_choices naive_extend_57
    # CUDA_VISIBLE_DEVICES=$device python main.py generate_images --model_type eagle --output_dir generated_images/lantern_search_mc_sim_7b_63 --prompt SJDPrompts --num_images 10\
    # --set_seed --random_seed 42 --grid_search --eagle_v1 --tree_choices mc_sim_7b_63

# elif [ $device == 1 ]; then
#     CUDA_VISIBLE_DEVICES=$device python main.py generate_images\
#         --model_type base --output_dir generated_images/PartiPrompts/debug_base_parallel\
#         --prompt PartiPrompts --num_images 10000 --set_seed --random_seed 42\
#         --cfg_mode parallel
    # CUDA_VISIBLE_DEVICES=$device python main.py generate_images\
    #     --model_type eagle --output_dir generated_images/PartiPrompts/debug_image_test2_sequential\
    #     --prompt PartiPrompts --num_images 10000 --set_seed --random_seed 42\
    #     --tree_choices naive_extend_57 --lantern --lantern_k 10 --lantern_delta 5\
    #     --eagle_version 2 --cfg_mode sequential
    # CUDA_VISIBLE_DEVICES=$device python main.py generate_images --model_type eagle --output_dir generated_images/PartiPrompts/LANTERN/k_5_lambda_10 --prompt PartiPrompts --num_images 10000\
    #     --set_seed --random_seed 42 --eagle_v1 --tree_choices naive_extend_57 --lantern --lantern_k 5 --lantern_delta 10
    # CUDA_VISIBLE_DEVICES=$device python main.py generate_images --model_type eagle --output_dir generated_images/lantern_search_mc_sim_7b_63_balanced --prompt SJDPrompts --num_images 10\
    # --set_seed --random_seed 42 --grid_search --eagle_v1 --tree_choices mc_sim_7b_63_balanced

# elif [ $device == 2 ]; then
#     CUDA_VISIBLE_DEVICES=$device python main.py generate_images --model_type base --output_dir generated_images/PartiPrompts/base --prompt PartiPrompts --num_images 200\
#         --set_seed --random_seed 42
    # CUDA_VISIBLE_DEVICES=$device python main.py generate_images --model_type eagle --output_dir generated_images/PartiPrompts/LANTERN/k_10_lambda_5 --prompt PartiPrompts --num_images 10000\
    #     --set_seed --random_seed 42 --eagle_v1 --tree_choices naive_extend_57 --lantern --lantern_k 10 --lantern_delta 5
    # CUDA_VISIBLE_DEVICES=$device python main.py generate_images --model_type eagle --output_dir generated_images/lantern_search_naive_extend_57 --prompt SJDPrompts --num_images 10\
    # --set_seed --random_seed 42 --grid_search --eagle_v1 --tree_choices naive_extend_57

# fi

# python main.py generate_images --model_type eagle --output_dir generated_images/eagle1/SJDPrompts --prompt SJDPrompts --num_images 10\
#     --set_seed --random_seed 42 --grid_search --eagle_v1

# python main.py generate_images --model_type eagle --output_dir generated_images/eagle2/SJDPrompts --prompt SJDPrompts --num_images 10\
#     --set_seed --random_seed 42 --grid_search