import os
import json
import random

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import readColmapSceneInfo
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

from fast_gs.Config.config import ModelParams
from fast_gs.Model.gs import GaussianModel


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        scene_info = readColmapSceneInfo(args.source_path, args.images)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            for id, cam in enumerate(scene_info.train_cameras):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)

        test_cam = self.train_cameras[0]
        print('fov:', test_cam.FoVx, test_cam.FoVy)
        print('image:', test_cam.image_name, test_cam.image_width, test_cam.image_height)
        print('pose:')
        print(test_cam.R)
        print(test_cam.T)
        print('transform:')
        print(test_cam.world_view_transform)
        print(test_cam.projection_matrix)
        print(test_cam.full_proj_transform)
        print('camera_center:', test_cam.camera_center)

        import cv2
        import numpy as np
        # original_image: (3, H, W), RGB, 0-1, CUDA tensor
        # Convert to (H, W, 3), uint8, BGR for cv2.imwrite
        image = (test_cam.original_image.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.model_path, 'scene_image.png'), image_bgr)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def __len__(self) -> int:
        return len(self.train_cameras)

    def __getitem__(self, idx: int):
        valid_idx = idx % len(self.train_cameras)
        return self.train_cameras[valid_idx]
