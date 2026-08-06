#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, getProjectionMatrixCenterShift
import copy
from PIL import Image
from utils.general_utils import PILtoTorch
import os, cv2
import torch.nn.functional as F
import torchvision
from utils.general_utils import inverse_sigmoid

def dilate(bin_img, ksize=6):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=12):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

def process_image(image_path, resolution, ncc_scale, depth_path, conf_path, normal_path, stable_normal_path, depth_anything_path):
    image = Image.open(image_path)
    depth, conf, normal = None, None, None
    stable_normal = None
    resized_depth_anything = None
    resized_stable_normal = None
    if depth_path is not None:
        depth = torch.load(depth_path, weights_only=True).float() # (1 , H,  W)
    if conf_path is not None:
        conf = torch.load(conf_path, weights_only=True).float() # (1,  H,  W)
    if normal_path is not None:
        normal = torch.load(normal_path, weights_only=True).float() # (3,  H,  W)
    if stable_normal_path is not None:
        stable_normal = Image.open(stable_normal_path)
    if depth_anything_path is not None:
        depth_anything = Image.open(depth_anything_path) #(H, W, 3)
    if len(image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(image.split()[3], resolution)
        gt_image = resized_image_rgb
        # gt_image[loaded_mask.repeat((3, 1, 1)) == 0] = 0.0
            
        if ncc_scale != 1.0:
            ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
            resized_image_rgb = torch.cat([PILtoTorch(im, ncc_resolution) for im in image.split()[:3]], dim=0)
    else:
        resized_image_rgb = PILtoTorch(image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb
        if ncc_scale != 1.0:
            ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
            resized_image_rgb = PILtoTorch(image, ncc_resolution)
    C, H, W = gt_image.shape
    resize = torchvision.transforms.Resize((H, W), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    # resize_normal = torchvision.transforms.Resize((H, W), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    if depth is not None:
        depth = resize(depth)
        if loaded_mask is not None:
            depth[loaded_mask == 0] = -1
    if normal is not None:
        normal = resize(normal)
        normal /= normal.norm(dim=0, keepdim=True)

    if conf is not None:
        conf = resize(conf)
        conf = (conf - conf.min()) / (conf.max() - conf.min())
        if loaded_mask is not None:
            conf[loaded_mask == 0] = 0.0
    if stable_normal is not None:
        resized_stable_normal = PILtoTorch(stable_normal, resolution)
        resized_stable_normal = resized_stable_normal * 2 - 1
    if depth_anything is not None:
        resized_depth_anything = PILtoTorch(depth_anything, resolution)[[0]] # (1, H, W)
        assert len(resized_depth_anything.shape) == 3
        
    gray_image = (0.299 * resized_image_rgb[0] + 0.587 * resized_image_rgb[1] + 0.114 * resized_image_rgb[2])[None]
    return gt_image, gray_image, loaded_mask, depth, conf, normal, resized_stable_normal, resized_depth_anything

# def process_image(image_path, resolution, ncc_scale):
#     image = Image.open(image_path)
#     if len(image.split()) > 3:
#         resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in image.split()[:3]], dim=0)
#         loaded_mask = PILtoTorch(image.split()[3], resolution)
#         gt_image = resized_image_rgb
#         if ncc_scale != 1.0:
#             ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
#             resized_image_rgb = torch.cat([PILtoTorch(im, ncc_resolution) for im in image.split()[:3]], dim=0)
#     else:
#         resized_image_rgb = PILtoTorch(image, resolution)
#         loaded_mask = None
#         gt_image = resized_image_rgb
#         if ncc_scale != 1.0:
#             ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
#             resized_image_rgb = PILtoTorch(image, ncc_resolution)
#     gray_image = (0.299 * resized_image_rgb[0] + 0.587 * resized_image_rgb[1] + 0.114 * resized_image_rgb[2])[None]
#     return gt_image, gray_image, loaded_mask

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy,
                 image_width, image_height, image,
                 image_path, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 ncc_scale=1.0,
                 preload_img=True, data_device = "cuda"
                 ):
        super(Camera, self).__init__()
        self.uid = uid
        self.nearest_id = []
        self.nearest_names = []
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image = image
        self.image_name = image_name
        self.image_path = image_path
        self.image_width = image_width
        self.image_height = image_height
        self.resolution = (image_width, image_height)
        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image, self.image_gray, self.mask = None, None, None
        self.preload_img = preload_img
        self.ncc_scale = ncc_scale

        self.depth_path = None
        self.conf_path = None
        self.normal_path = None
        self.stable_normal_path = None
        self.depth_anything_path = None
        folder_name = os.path.dirname(os.path.dirname(os.path.join(self.image_path)))
        if os.path.exists(os.path.join(folder_name, "depth_vggt")):
            self.depth_path = os.path.join(folder_name, "depth_vggt", self.image_name + "_depth.pt")
            self.conf_path = os.path.join(folder_name, "conf_vggt", self.image_name + "_conf.pt")    
            self.normal_path = os.path.join(folder_name, "normal_vggt", self.image_name + "_normal.pt")    
        elif os.path.basename(os.path.dirname(image_path)) == "train" and os.path.exists(os.path.join(folder_name, "train_vggt")):
            self.depth_path = os.path.join(folder_name, "train_vggt", "depth_vggt", self.image_name + "_depth.pt")
            self.conf_path = os.path.join(folder_name, "train_vggt",  "conf_vggt", self.image_name + "_conf.pt")    
            self.normal_path = os.path.join(folder_name, "train_vggt", "normal_vggt", self.image_name + "_normal.pt")    
        elif os.path.basename(os.path.dirname(image_path)) == "test" and os.path.exists(os.path.join(folder_name, "test_vggt")):
            self.depth_path = os.path.join(folder_name, "test_vggt", "depth_vggt", self.image_name + "_depth.pt")
            self.conf_path = os.path.join(folder_name, "test_vggt",  "conf_vggt", self.image_name + "_conf.pt")    
            self.normal_path = os.path.join(folder_name, "test_vggt", "normal_vggt", self.image_name + "_normal.pt")  

        if os.path.exists(os.path.join(folder_name, "stablenormals")):
            self.stable_normal_path = os.path.join(folder_name, "stablenormals", self.image_name + ".png")
            if not os.path.exists(self.stable_normal_path):
                self.stable_normal_path = os.path.join(folder_name, "stablenormals", self.image_name + ".jpg")
                if not os.path.exists(self.stable_normal_path):
                    self.stable_normal_path = os.path.join(folder_name, "stablenormals", self.image_name + "_normal.png")
        elif os.path.basename(os.path.dirname(image_path)) == "train" and os.path.exists(os.path.join(folder_name, "train_stablenormals")):
            self.stable_normal_path = os.path.join(folder_name, "train_stablenormals", self.image_name + ".png")
            if not os.path.exists(self.stable_normal_path):
                self.stable_normal_path = os.path.join(folder_name, "train_stablenormals", self.image_name + ".jpg")
        elif os.path.basename(os.path.dirname(image_path)) == "test" and os.path.exists(os.path.join(folder_name, "test_stablenormals")):
            self.stable_normal_path = os.path.join(folder_name, "test_stablenormals", self.image_name + ".png")
            if not os.path.exists(self.stable_normal_path):
                self.stable_normal_path = os.path.join(folder_name, "test_stablenormals", self.image_name + ".jpg")
        
        if os.path.exists(os.path.join(folder_name, "depth-anything")):
            self.depth_anything_path = os.path.join(folder_name, "depth-anything", self.image_name + ".png")
            if not os.path.exists(self.depth_anything_path):
                self.depth_anything_path = os.path.join(folder_name, "depth-anything", self.image_name + ".jpg")
        elif os.path.basename(os.path.dirname(image_path)) == "train" and os.path.exists(os.path.join(folder_name, "train_depth-anything")):
            self.depth_anything_path = os.path.join(folder_name, "train_depth-anything", self.image_name + ".png")
            if not os.path.exists(self.depth_anything_path):
                self.depth_anything_path = os.path.join(folder_name, "train_depth-anything", self.image_name + ".jpg")
        elif os.path.basename(os.path.dirname(image_path)) == "test" and os.path.exists(os.path.join(folder_name, "test_depth-anything")):
            self.depth_anything_path = os.path.join(folder_name, "test_depth-anything", self.image_name + ".png")
            if not os.path.exists(self.depth_anything_path):
                self.depth_anything_path = os.path.join(folder_name, "test_depth-anything", self.image_name + ".jpg")

        self.depth, self.conf, self.normal = None, None, None
        self.stable_normal = None
        self.depth_anything = None

        if self.preload_img:
            gt_image, gray_image, loaded_mask, depth, conf, normal, stable_normal, depth_anything = process_image(self.image_path, self.resolution, ncc_scale, self.depth_path, self.conf_path, self.normal_path, self.stable_normal_path, self.depth_anything_path)
            self.original_image = gt_image.to(self.data_device)
            self.original_image_gray = gray_image.to(self.data_device)
            self.mask = loaded_mask
            if depth is not None:
                self.depth = depth.to(self.data_device)
            if conf is not None:
                self.conf = conf.to(self.data_device)
            #if normal is not None:
            #    self.normal = normal.to(self.data_device)
            #if stable_normal is not None:
            #    self.stable_normal = stable_normal.to(self.data_device)
            if self.mask is not None:
                self.mask = self.mask.to(self.data_device)
            #if depth_anything is not None:
            #    self.depth_anything = depth_anything.to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01
        
        if self.conf is None:
            self.conf_depth_vggt = nn.Parameter(inverse_sigmoid(0.99 * torch.ones_like(self.mask, dtype=torch.float32).cuda()), requires_grad=True)
        else:
            self.conf_depth_vggt = nn.Parameter(inverse_sigmoid(0.99 * self.conf.squeeze().detach().to(torch.float32).cuda()), requires_grad=True)
        self.optimizer = torch.optim.Adam([{"params": self.conf_depth_vggt, "lr":0.05}])

        self.trans = trans
        self.scale = scale
        
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.plane_mask, self.non_plane_mask = None, None

    def get_image(self):
        if self.preload_img:
            return self.original_image.cuda(), self.original_image_gray.cuda(), self.depth, self.conf, self.normal, (self.stable_normal, self.mask, self.depth_anything)
        else:
            gt_image, gray_image, loaded_mask, depth, conf, normal, stable_normal, depth_anything = process_image(self.image_path, self.resolution, self.ncc_scale, self.depth_path, self.conf_path, self.normal_path, self.stable_normal_path, self.depth_anything_path)
            if depth is not None:
                depth = depth.to(self.data_device)
            if conf is not None:
                conf = conf.to(self.data_device)
            if normal is not None:
                normal = normal.to(self.data_device)
            if stable_normal is not None:
                stable_normal = stable_normal.to(self.data_device)
            if depth_anything is not None:
                depth_anything = depth_anything.to(self.data_device)
            if loaded_mask is not None:
                loaded_mask = loaded_mask.to(self.data_device)
            return gt_image.cuda(), gray_image.cuda(), depth, conf, normal, (stable_normal, loaded_mask, depth_anything)
    def get_depth_conf_vggt(self):
        return F.sigmoid(self.conf_depth_vggt)

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.Fx/scale, 0, self.Cx/scale], [0, self.Fy/scale, self.Cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix
    
    def get_rays(self, scale=1.0):
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
                    [(ix-self.Cx/scale) / self.Fx * scale,
                    (iy-self.Cy/scale) / self.Fy * scale,
                    torch.ones_like(ix)], -1).float().cuda()
        return rays_d
    
    def get_k(self, scale=1.0):
        K = torch.tensor([[self.Fx / scale, 0, self.Cx / scale],
                        [0, self.Fy / scale, self.Cy / scale],
                        [0, 0, 1]]).cuda()
        return K
    
    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale/self.Fx, 0, -self.Cx/self.Fx],
                            [0, scale/self.Fy, -self.Cy/self.Fy],
                            [0, 0, 1]]).cuda()
        return K_T

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

def sample_cam(cam_l: Camera, cam_r: Camera):
    cam = copy.copy(cam_l)

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam_l.R.transpose()
    Rt[:3, 3] = cam_l.T
    Rt[3, 3] = 1.0

    Rt2 = np.zeros((4, 4))
    Rt2[:3, :3] = cam_r.R.transpose()
    Rt2[:3, 3] = cam_r.T
    Rt2[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    C2W2 = np.linalg.inv(Rt2)
    w = np.random.rand()
    pose_c2w_at_unseen =  w * C2W + (1 - w) * C2W2
    Rt = np.linalg.inv(pose_c2w_at_unseen)
    cam.R = Rt[:3, :3]
    cam.T = Rt[:3, 3]

    cam.world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, cam.trans, cam.scale)).transpose(0, 1).cuda()
    cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy).transpose(0,1).cuda()
    cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
    return cam
