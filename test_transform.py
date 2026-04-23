import sys
sys.path.append('../base-trainer')
sys.path.append('../base-gs-trainer')
sys.path.append('../camera-control')

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import math
import cv2
import numpy as np
import torch

from camera_control.Module.camera_convertor import CameraConvertor

from fast_gs.Module.gs_renderer import GSRenderer
from fast_gs.Method.transform import (
    translateGSPlyFile,
    scaleGSPlyFile,
    rotateGSPlyFile,
    transformGSPlyFile,
)


def buildSimilarityTransform(angle_deg: float, axis: list, scale: float, translate: list) -> np.ndarray:
    '''
    构造行向量右乘约定下的 4x4 相似变换矩阵:
      T_right[:3, :3] = scale * R_right
      T_right[3, :3]  = translate
    其中 R_right = R_left.T, R_left 由 Rodrigues 公式按列向量约定构造。
    '''
    axis_arr = np.asarray(axis, dtype=np.float64)
    axis_arr = axis_arr / max(np.linalg.norm(axis_arr), 1e-12)

    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    K = np.array([
        [0.0, -axis_arr[2], axis_arr[1]],
        [axis_arr[2], 0.0, -axis_arr[0]],
        [-axis_arr[1], axis_arr[0], 0.0],
    ], dtype=np.float64)
    R_left = np.eye(3, dtype=np.float64) + s * K + (1 - c) * (K @ K)

    R_right = R_left.T

    T_right = np.eye(4, dtype=np.float64)
    T_right[:3, :3] = scale * R_right
    T_right[3, :3] = np.asarray(translate, dtype=np.float64)
    return T_right


def renderAll(gs_ply_file_path: str, camera_list, sh_degree: int, bg_color: list) -> list:
    render_list = GSRenderer.renderCameras(
        gs_ply_file_path,
        camera_list,
        sh_degree=sh_degree,
        bg_color=bg_color,
        mult=0.5,
        device='cuda:0',
    )
    return [r['render'] for r in render_list]


def toImageCV(tensor_chw: torch.Tensor) -> np.ndarray:
    '''
    将渲染输出 (3, H, W) float [0, 1] 转为 BGR uint8 ndarray。
    '''
    img = tensor_chw.detach().float().clamp(0.0, 1.0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def saveBeforeAfterGrid(
    renders_before: list,
    renders_after: list,
    save_dir: str,
    gap: int = 8,
    gap_color: tuple = (255, 255, 255),
) -> None:
    '''
    将每一对 (before, after) 图像横向拼接保存到 save_dir, 同时保存一张总览网格图。
    - per_cam_{i}.png: 左 before, 右 after, 中间白色分隔条
    - all_cams_grid.png: 所有相机上下堆叠的总览图
    '''
    assert len(renders_before) == len(renders_after), 'rendered image count mismatch'
    os.makedirs(save_dir, exist_ok=True)

    pair_list = []
    for i, (a, b) in enumerate(zip(renders_before, renders_after)):
        img_a = toImageCV(a)
        img_b = toImageCV(b)
        if img_a.shape != img_b.shape:
            h = min(img_a.shape[0], img_b.shape[0])
            w = min(img_a.shape[1], img_b.shape[1])
            img_a = cv2.resize(img_a, (w, h))
            img_b = cv2.resize(img_b, (w, h))

        h, w = img_a.shape[:2]
        sep = np.full((h, gap, 3), gap_color, dtype=np.uint8)
        pair = np.concatenate([img_a, sep, img_b], axis=1)

        label = f'cam {i}    [before | after]'
        cv2.putText(pair, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(pair, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        per_cam_path = os.path.join(save_dir, f'per_cam_{i:03d}.png')
        cv2.imwrite(per_cam_path, pair)
        pair_list.append(pair)

    if pair_list:
        max_w = max(p.shape[1] for p in pair_list)
        padded = []
        for p in pair_list:
            if p.shape[1] < max_w:
                pad = np.full((p.shape[0], max_w - p.shape[1], 3), gap_color, dtype=np.uint8)
                p = np.concatenate([p, pad], axis=1)
            padded.append(p)
        row_sep = np.full((gap, max_w, 3), gap_color, dtype=np.uint8)
        rows = []
        for i, p in enumerate(padded):
            rows.append(p)
            if i != len(padded) - 1:
                rows.append(row_sep)
        grid = np.concatenate(rows, axis=0)
        grid_path = os.path.join(save_dir, 'all_cams_grid.png')
        cv2.imwrite(grid_path, grid)
        print(f'[INFO][test_transform] saved render grid -> {grid_path}')


def compareImageLists(list_a: list, list_b: list, tol_mean: float = 1e-3, tol_max: float = 5e-2) -> bool:
    assert len(list_a) == len(list_b), 'rendered image count mismatch'
    all_ok = True
    for i, (a, b) in enumerate(zip(list_a, list_b)):
        diff = (a.float() - b.float()).abs()
        mean_err = float(diff.mean())
        max_err = float(diff.max())
        ok = mean_err <= tol_mean and max_err <= tol_max
        status = 'OK' if ok else 'DIFF'
        print(f'\t [cam {i}] mean={mean_err:.6f} max={max_err:.6f} -> {status}')
        if not ok:
            all_ok = False
    return all_ok


def verifyRenderInvariance():
    home = os.environ['HOME']
    data_id = 'haizei_1_v4'
    colmap_data_folder_path = f'{home}/chLi/Dataset/GS/{data_id}/colmap_normalized/'
    gs_ply_file_path = f'{home}/chLi/Dataset/GS/{data_id}/fastgs_pcd.ply'

    work_dir = f'{home}/chLi/Dataset/GS/{data_id}/transform_test/'
    os.makedirs(work_dir, exist_ok=True)
    transformed_gs_ply_file_path = work_dir + 'transformed_fastgs_pcd.ply'
    render_save_dir = work_dir + 'renders/'

    sh_degree = 3
    bg_color = [1, 1, 1]

    T_right = buildSimilarityTransform(
        angle_deg=37.0,
        axis=[0.3, 0.8, 0.5],
        scale=1.7,
        translate=[0.2, -0.15, 0.35],
    )

    print('[INFO][test_transform::verifyRenderInvariance] loading cameras...')
    camera_list = CameraConvertor.loadColmapDataFolder(colmap_data_folder_path)
    if len(camera_list) == 0:
        print('[ERROR][test_transform] no cameras loaded, abort.')
        return False
    sample_camera_list = camera_list[:4]

    print('[INFO][test_transform] rendering original scene...')
    renders_before = renderAll(gs_ply_file_path, sample_camera_list, sh_degree, bg_color)

    print('[INFO][test_transform] transforming gaussians...')
    ok = transformGSPlyFile(
        gs_ply_file_path=gs_ply_file_path,
        transform=T_right,
        save_gs_ply_file_path=transformed_gs_ply_file_path,
        overwrite=True,
        sh_degree=sh_degree,
    )
    if not ok:
        print('[ERROR][test_transform] transformGSPlyFile failed')
        return False

    print('[INFO][test_transform] transforming cameras with same T_right...')
    transformed_camera_list = CameraConvertor.transformCameras(sample_camera_list, T_right)

    print('[INFO][test_transform] rendering transformed scene with transformed cameras...')
    renders_after = renderAll(
        transformed_gs_ply_file_path, transformed_camera_list, sh_degree, bg_color,
    )

    print('[INFO][test_transform] saving before/after renders...')
    saveBeforeAfterGrid(renders_before, renders_after, render_save_dir)

    print('[INFO][test_transform] comparing renders:')
    all_ok = compareImageLists(renders_before, renders_after)
    print(f'[INFO][test_transform] render invariance check: {"PASS" if all_ok else "FAIL"}')
    return all_ok


def smokeTestIndividualAPIs():
    '''
    对 translate / scale / rotate 三个单独入口做冒烟测试,
    仅校验函数能正常跑完并输出 ply 文件,真正的不变性由 transformGSPlyFile 的端到端验证保证。
    '''
    home = os.environ['HOME']
    data_id = 'haizei_1_v4'
    gs_ply_file_path = f'{home}/chLi/Dataset/GS/{data_id}/fastgs_pcd.ply'
    work_dir = f'{home}/chLi/Dataset/GS/{data_id}/transform_test/'
    os.makedirs(work_dir, exist_ok=True)

    all_ok = True
    all_ok &= translateGSPlyFile(
        gs_ply_file_path, [0.1, -0.2, 0.3],
        work_dir + 'translated_fastgs_pcd.ply', overwrite=True,
    )
    all_ok &= scaleGSPlyFile(
        gs_ply_file_path, 1.5,
        work_dir + 'scaled_fastgs_pcd.ply', overwrite=True,
    )
    rotation_right = [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]
    all_ok &= rotateGSPlyFile(
        gs_ply_file_path, rotation_right,
        work_dir + 'rotated_fastgs_pcd.ply', overwrite=True,
    )
    print(f'[INFO][test_transform] smoke test APIs: {"PASS" if all_ok else "FAIL"}')
    return all_ok


if __name__ == '__main__':
    with torch.no_grad():
        smokeTestIndividualAPIs()
        verifyRenderInvariance()
