# [GCPR 2026] Confidence matters: Leveraging Multi-view Geometric Priors for GS-based Reconstruction
[Hongyu Zhou](https://zero-4869.github.io), [Zorah Lähner](https://geometryinml.cs.uni-bonn.de/team/zorah/)
[![Papers with Code: #3 on DTU](https://paperswithcode.co/api/v1/papers/2608.06117/leaderboard-badge.svg?eval=25687&live=1)](https://paperswithcode.co/api/v1/papers/2608.06117/leaderboard-badge-link?eval=25687)

## Installation
The repository is built on [PGSR](https://github.com/zju3dv/PGSR). To install, run
```
https://github.com/Zero-4869/ConfidenceMattersGS.git
cd ConfidenceMattersGS
conda create -n ConfidenceMattersGS python=3.10
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r requirements.txt
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn
```

## Datasets
The data folder follows
```
Datasets
|- dtu_dataset
|   |-dtu
|   |   |-scan24
|   |   |   |-images
|   |   |   |-mask
|   |   |   |-sparse
|   |   |   |-depth_vggt
|   |   |   |   |-0000_depth.pt
|   |   |   |-conf_vggt
|   |   |   |   |-0000_conf.pt
|   |   |   |-cameras_sphere.npz
|   |   |   |-cameras.npz
|   |-dtu_eval
|   |   |-Points
|   |   |   |-stl
|   |   |-ObsMask
```

## Training and Evaluation
```
# DTU 
sh run_dtu.sh
# Tanks and Temples
sh run_tnt.sh
# Shiny Blender
sh run_refnerf.sh
```
## Acknowledgements
The Gaussian Splatting is based on [PGSR](https://github.com/zju3dv/PGSR). The geometric priors are adopted from [VGGT](https://github.com/facebookresearch/vggt). We thank all the authors for their great work and repos.

## Citation
```
@inproceedings{zhou2026confidence,
  title={Confidence matters: Leveraging Multi-view Geometric Priors for GS-based Reconstruction},
  author={Zhou, Hongyu and L{\"a}hner, Zorah},
  booktitle={DAGM German Conference on Pattern Recognition},
  year={2026},
  publisher={Springer}
}
```
