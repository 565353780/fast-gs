import sys
sys.path.append('../../../camera-control/')

import os
import torch
import random
import numpy as np

from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.camera import Camera

from scene.dataset_readers import readColmapSceneInfo
from utils.graphics_utils import getProjectionMatrix, focal2fov, getWorld2View

from fast_gs.Config.config import ModelParams
from fast_gs.Model.gs import GaussianModel


def _cameras_extent_from_list(cam_list):
    """从 camera_control Camera 列表计算 cameras_extent (radius)。"""
    if not cam_list:
        return 1.0
    centers = []
    for c in cam_list:
        pos = c.pos
        if torch.is_tensor(pos):
            pos = pos.detach().cpu().numpy()
        centers.append(pos.reshape(3, 1))
    centers = np.hstack(centers)
    center = np.mean(centers, axis=1, keepdims=True)
    dist = np.linalg.norm(centers - center, axis=0, keepdims=True)
    diagonal = float(np.max(dist))
    return diagonal * 1.1


class ViewpointAdapter:
    """
    将 camera_control.Camera（自带 image/mask、world2camera）适配为 fast-gs 渲染与训练
    所需的 viewpoint 接口：world_view_transform, full_proj_transform, camera_center,
    FoVx, FoVy, image_width, image_height, original_image, image_name, R, T 等。
    """
    znear = 0.01
    zfar = 100.0

    def __init__(self, cam: Camera, data_device: str = "cuda", mask_folder: Optional[str] = None):
        self._cam = cam

        self.uid = id(cam)

        w2c = cam.world2cameraColmap.float()
        self.R = w2c[:3, :3].T.cpu().numpy()
        self.T = w2c[:3, 3].cpu().numpy()

        self.FoVx = float(focal2fov(cam.fx, cam.width))
        self.FoVy = float(focal2fov(cam.fy, cam.height))

        self.image_width = cam.width
        self.image_height = cam.height

        # 图像：(H, W, 3) -> (3, H, W)，并应用 mask（若有）
        img = cam.image.float().permute(2, 0, 1)

        # 可选：从 mask 文件夹加载并应用
        if mask_folder and os.path.isdir(mask_folder):
            base = cam.image_id if hasattr(cam, 'image_id') else None
            if base:
                for ext in ('', '.png', '.jpg', '.jpeg'):
                    name = base if base.endswith(('.png', '.jpg', '.jpeg')) else base + ext
                    mask_path = os.path.join(mask_folder, os.path.basename(name))
                    if os.path.isfile(mask_path):
                        cam.loadMaskFile(mask_path)
                        break

        if cam.mask is not None:
            mask = cam.mask.unsqueeze(0).float()
            img = img * mask + (1.0 - mask)

        self.original_image = img.clamp(0.0, 1.0).to(data_device)
        self.image_name = cam.image_id

        # 投影矩阵（与 utils.graphics_utils.getWorld2View 及 scene/cameras.py Camera 一致）
        # getWorld2View(R,T) 构造 Rt[:3,:3]=R.T、Rt[:3,3]=T，即 W2C；此处 self.R=w2c[:3,:3].T 故得到 w2c
        self.world_view_transform = torch.tensor(getWorld2View(self.R, self.T)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        w2c_np = getWorld2View(self.R, self.T)
        c2w_np = np.linalg.inv(w2c_np)
        self.camera_center = torch.tensor(c2w_np[:3, 3], dtype=torch.float32).cuda()
        return

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, shuffle=True):
        """
        :param path: Path to colmap scene main folder.
        优先使用 CameraConvertor.loadColmapDataFolder 得到自带 image/mask 的 List[Camera]，
        训练时通过 camera 的变换矩阵和 image 直接用于高斯训练。
        """
        self.model_path = args.model_path
        self.gaussians = gaussians

        colmap_cameras = CameraConvertor.loadColmapDataFolder(args.source_path)

        mask_folder = os.path.join(args.source_path, 'masks')
        data_device = getattr(args, 'data_device', 'cuda')
        n_cams = len(colmap_cameras)
        self.train_cameras: List[ViewpointAdapter] = [None] * n_cams
        with ThreadPoolExecutor(max_workers=min(8, n_cams)) as executor:
            futures = {
                executor.submit(
                    ViewpointAdapter, c,
                    data_device=data_device,
                    mask_folder=mask_folder,
                ): i
                for i, c in enumerate(colmap_cameras)
            }
            with tqdm(total=n_cams, desc="Loading viewpoints", unit="cam") as pbar:
                for fut in as_completed(futures):
                    idx = futures[fut]
                    self.train_cameras[idx] = fut.result()
                    pbar.update(1)

        self.cameras_extent = _cameras_extent_from_list(colmap_cameras)
        if shuffle:
            random.shuffle(self.train_cameras)

        pcd = readColmapSceneInfo(args.source_path)
        self.gaussians.create_from_pcd(pcd, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def __len__(self) -> int:
        return len(self.train_cameras)

    def __getitem__(self, idx: int):
        valid_idx = idx % len(self.train_cameras)
        return self.train_cameras[valid_idx]
