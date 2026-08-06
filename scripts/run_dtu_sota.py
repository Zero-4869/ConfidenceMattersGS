import os
import sys

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

data_base_path='/lustre/scratch/data/hzhou_hpc-transparent/data/DTU'
out_base_path='/lustre/scratch/data/hzhou_hpc-transparent/output/output_dtu'
eval_path='/lustre/scratch/data/hzhou_hpc-transparent/data/DTU_eval'
gpu_id=0
mask_type=sys.argv[1]
normal_type=sys.argv[2]
anchor_type=sys.argv[3]
depth_type=sys.argv[4]
out_name=f'pgsr_vggt_{mask_type}'


for scene in scenes:
    cmd = f'rm -rf {out_base_path}/dtu_scan{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    cmd = f'cp -rf {data_base_path}/scan{scene}/sparse/0/* {data_base_path}/scan{scene}/sparse/'
    print(cmd)
    os.system(cmd)

    common_args = "--quiet -r2 --ncc_scale 0.5"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args} --mask_type {mask_type} --normal_type {normal_type} --anchor_type {anchor_type} --depth_type {depth_type}'
    print(cmd)
    os.system(cmd)

    common_args = "--quiet --num_cluster 1 --voxel_size 0.002 --max_depth 5.0"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_mesh {out_base_path}/dtu_scan{scene}/{out_name}/mesh/tsdf_fusion_post.ply " + \
          f"--scan_id {scene} --output_dir {out_base_path}/dtu_scan{scene}/{out_name}/mesh " + \
          f"--mask_dir {data_base_path} " + \
          f"--DTU {eval_path}"
    print(cmd)
    os.system(cmd)
