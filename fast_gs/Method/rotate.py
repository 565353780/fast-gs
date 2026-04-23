import os
import torch
import numpy as np

from typing import Union

from camera_control.Method.data import toNumpy
from base_gs_trainer.Method.path import removeFile, createFileFolder

def rotateGSPlyFile(
    gs_ply_file_path: str,
    rotation_right: Union[torch.Tensor, np.ndarray, list],
    save_gs_ply_file_path: str,
    overwrite: bool=False,
) -> bool:
    if os.path.exists(save_gs_ply_file_path):
        if not overwrite:
            return True

        removeFile(save_gs_ply_file_path)

    if not os.path.exists(gs_ply_file_path):
        print('[ERROR][rotate::rotateGSPlyFile]')
        print('\t gs ply file not exist!')
        print('\t gs_ply_file_path:', gs_ply_file_path)
        return False

    rotation_right = toNumpy(rotation_right, np.float64).reshape(3, 3)

    createFileFolder(save_gs_ply_file_path)
    return True
