import os
import sys

scenes = ['Courthouse', 'Caterpillar', 'Barn', 'Meetingroom', 'Ignatius', 'Truck']
data_devices = ['cuda', 'cuda','cuda','cuda', 'cuda', 'cuda']

data_base_path='/lustre/scratch/data/hzhou_hpc-transparent/data/TnT'
out_base_path='/lustre/scratch/data/hzhou_hpc-transparent/output/output_tnt'
gpu_id=0
mask_type=sys.argv[1]
normal_type=sys.argv[2]
anchor_type=sys.argv[3]
depth_type=sys.argv[4]
ncc=sys.argv[5]
vfm=sys.argv[6]
out_name=f'pgsr_vggt_{mask_type}_fullmask_eval_{ncc}_{vfm}_init'


for id, scene in enumerate(scenes):

    cmd = f'rm -rf {out_base_path}/{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)
    
    common_args = f"-r2 --ncc_scale 0.5 --data_device {data_devices[id]} --densify_abs_grad_threshold 0.00015 --opacity_cull_threshold 0.05 --exposure_compensation --use_depth --eval --max_all_points 1_000_000 --ncc_weight {ncc} --vfm_weight {vfm}"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} {common_args} --mask_type {mask_type} --normal_type {normal_type} --anchor_type {anchor_type} --depth_type {depth_type}'
    print(cmd)
    os.system(cmd)

    common_args = f"--data_device {data_devices[id]} --num_cluster 1 --use_depth_filter"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/render_tnt.py -m {out_base_path}/{scene}/{out_name} --data_device {data_devices[id]} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/tnt_eval/run.py --dataset-dir {data_base_path}/{scene} --traj-path {data_base_path}/{scene}/{scene}_COLMAP_SfM.log --ply-path {out_base_path}/{scene}/{out_name}/mesh/tsdf_fusion_post.ply --out-dir {out_base_path}/{scene}/{out_name}/mesh'
    print(cmd)
    os.system(cmd)
