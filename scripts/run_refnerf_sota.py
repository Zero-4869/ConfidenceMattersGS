import os
import sys

scenes = ["car", "coffee", "helmet", "teapot", "toaster"]

data_base_path='/lustre/scratch/data/hzhou_hpc-transparent/data/refnerf'
out_base_path='/lustre/scratch/data/hzhou_hpc-transparent/output/output_refnerf'

gpu_id=0
mask_type=sys.argv[1]
normal_type=sys.argv[2]
anchor_type=sys.argv[3]
depth_type=sys.argv[4]

out_name=f'pgsr_vggt_dynamic'

for id, scene in enumerate(scenes):

    cmd = f'rm -rf {out_base_path}/{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    common_args = f"-r 1 --data_device cuda --densify_abs_grad_threshold 0.0002 --eval --test_iterations 7000 30000"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} {common_args} --mask_type {mask_type} --normal_type {normal_type} --anchor_type {anchor_type} --depth_type {depth_type}'
    print(cmd)
    os.system(cmd)

    common_args = f"--num_cluster 1 --use_depth_filter --voxel_size 0.002 --max_depth 5.0"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = f"--skip_train"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py -m {out_base_path}/{scene}/{out_name}'
    print(cmd)
    os.system(cmd)
    
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/eval_translab/eval.py " + \
            f"--data {out_base_path}/{scene}/{out_name}/mesh/tsdf_fusion_post.ply " + \
          f"--scan {scene} --vis_out_dir {out_base_path}/{scene}/{out_name}/mesh " + \
          f"--dataset_dir {data_base_path} --mode mesh --downsample_density 0.002"
    print(cmd)
    os.system(cmd)

