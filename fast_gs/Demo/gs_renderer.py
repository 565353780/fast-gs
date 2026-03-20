import sys
sys.path.append('../base-trainer')
sys.path.append('../base-gs-trainer')
sys.path.append('../camera-control')
sys.path.append('../x-flux-3d-mv')

import os
os.environ['CUDA_VISIBLE_DEVICES']='6'

import cv2
import numpy as np
import open3d as o3d

from math import tan

from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.camera_filter import CameraFilter

from flux_mv.Dataset.mesh import fovToCameraDist

from fast_gs.Module.gs_renderer import GSRenderer


def demo():
    data_id = 'haizei_1_v4'

    home = os.environ['HOME']
    colmap_data_folder_path = home + '/chLi/Dataset/GS/' + data_id + '/colmap_normalized/'
    gs_ply_file_path = home + '/chLi/Dataset/GS/' + data_id + '/fastgs_pcd.ply'
    save_render_folder_path = home + '/chLi/Dataset/GS/' + data_id + '/fastgs_render/'
    os.makedirs(save_render_folder_path, exist_ok=True)

    pcd = o3d.io.read_point_cloud(gs_ply_file_path)
    pts = np.asarray(pcd.points)  # Nx3

    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2

    print("bbox min:", bbox_min)
    print("bbox max:", bbox_max)
    print("bbox center:", bbox_center)

    camera_list = CameraConvertor.loadColmapDataFolder(colmap_data_folder_path)

    width = height = 784
    fov_degree = 60.0
    target_position = [0, 0, 0]
    target_position = bbox_center
    camera_dist = fovToCameraDist(fov_degree, margin=1.0)
    fovx = fovy = np.radians(fov_degree)

    fps_camera_idxs = CameraFilter.samplePolarFarCameraIdxs(
        camera_list=camera_list,
        target_position=target_position,
        sample_camera_num=10,
    )
    fps_camera_list = [camera_list[idx] for idx in fps_camera_idxs]

    for i, fps_camera in enumerate(fps_camera_list):
        fps_camera.setImageSize(width, height)
        fps_camera.fx = fps_camera.width / 2.0 / tan(fovx / 2.0)
        fps_camera.fy = fps_camera.height / 2.0 / tan(fovy / 2.0)

        polar = fps_camera.toPolar(target_position)
        phi, theta = polar[0], polar[1]
        direction = np.array([
            np.sin(phi) * np.sin(theta),
            np.sin(phi) * np.cos(theta),
            np.cos(phi),
        ])
        new_pos = np.array(target_position) + direction * camera_dist
        fps_camera.setWorldPose(look_at=target_position, pos=new_pos, up=[0, 1, 0])

        uv = fps_camera.project_points_to_uv(target_position)
        print(f"Camera {i}: target uv = ({uv[0, 0].item():.4f}, {uv[0, 1].item():.4f})")

    render_list = GSRenderer.renderCameras(
        gs_ply_file_path,
        fps_camera_list,
        sh_degree=3,
        bg_color=[1, 1, 1],
        mult=0.5,
        device='cuda:0',
    )

    for i in range(len(fps_camera_list)):
        image_name = fps_camera_list[i].image_id

        save_image_file_path = save_render_folder_path + image_name

        image = (render_list[i]['render'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        cv2.imwrite(save_image_file_path, image[..., ::-1])

    return True
