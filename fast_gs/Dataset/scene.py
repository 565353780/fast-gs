import sys
sys.path.append('../../../camera-control/')

import os
import torch
import random
import numpy as np

from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.camera import Camera

from fast_gs.Config.config import ModelParams
from fast_gs.Model.gs import GaussianModel
from fast_gs.Method.colmap_io import readColmapPcd
from fast_gs.Method.graphics_utils import getProjectionMatrix, focal2fov


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


class GSCamera:
    def __init__(self, cam: Camera, data_device: str = "cuda:0"):
        self._cam = cam
        self._cam.to(torch.float32, data_device)

        self.uid = id(cam)

        self.FoVx = float(focal2fov(cam.fx, cam.width))
        self.FoVy = float(focal2fov(cam.fy, cam.height))

        self.image_width = cam.width
        self.image_height = cam.height

        # 图像：(H, W, 3) -> (3, H, W)，并应用 mask（若有）
        self.original_image = cam.toMaskedImage().permute(2, 0, 1).clamp(0.0, 1.0).to(data_device)
        self.image_name = cam.image_id

        self.world_view_transform = cam.world2cameraColmap.transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(self.FoVx, self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        self.camera_center = cam.camera2worldColmap[:3, 3].cuda()
        return

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, shuffle=True):
        self.model_path = args.model_path
        self.gaussians = gaussians

        colmap_cameras = CameraConvertor.loadColmapDataFolder(args.source_path)

        data_device = getattr(args, 'data_device', 'cuda')
        n_cams = len(colmap_cameras)
        self.train_cameras: List[GSCamera] = [None] * n_cams
        with ThreadPoolExecutor(max_workers=min(8, n_cams)) as executor:
            futures = {
                executor.submit(
                    GSCamera, c,
                    data_device=data_device,
                ): i
                for i, c in enumerate(colmap_cameras)
            }
            with tqdm(total=n_cams, desc="Loading GSCamera") as pbar:
                for fut in as_completed(futures):
                    idx = futures[fut]
                    self.train_cameras[idx] = fut.result()
                    pbar.update(1)

        self.cameras_extent = _cameras_extent_from_list(colmap_cameras)
        if shuffle:
            random.shuffle(self.train_cameras)

        pcd = readColmapPcd(args.source_path)
        self.gaussians.create_from_pcd(pcd, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def __len__(self) -> int:
        return len(self.train_cameras)

    def __getitem__(self, idx: int):
        valid_idx = idx % len(self.train_cameras)
        return self.train_cameras[valid_idx]
