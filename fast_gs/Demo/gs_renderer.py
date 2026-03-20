import sys
sys.path.append('../base-trainer')
sys.path.append('../base-gs-trainer')
sys.path.append('../camera-control')

import os
os.environ['CUDA_VISIBLE_DEVICES']='6'

import cv2
import numpy as np

from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.camera_filter import CameraFilter

from fast_gs.Module.gs_renderer import GSRenderer


def demo():
    data_id = 'haizei_1_v4'

    home = os.environ['HOME']
    colmap_data_folder_path = home + '/chLi/Dataset/GS/' + data_id + '/colmap_normalized/'
    gs_ply_file_path = home + '/chLi/Dataset/GS/' + data_id + '/fastgs_pcd.ply'
    save_render_folder_path = home + '/chLi/Dataset/GS/' + data_id + '/fastgs_render/'
    os.makedirs(save_render_folder_path, exist_ok=True)

    camera_list = CameraConvertor.loadColmapDataFolder(colmap_data_folder_path)

    fps_camera_idxs = CameraFilter.sampleFarCameraIdxs(camera_list, sample_camera_num=10)
    fps_camera_list = [camera_list[idx] for idx in fps_camera_idxs]

    for fps_camera in fps_camera_list:
        fps_camera.setImageSize(784, 784)

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
