import sys
sys.path.append('../camera-control')
sys.path.append('../base-gs-trainer')

import os

from fast_gs.Method.rotate import rotateGSPlyFile


if __name__ == '__main__':
    home = os.environ['HOME']
    gs_ply_file_path = f'{home}/chLi/Dataset/GS/haizei_1_v4/fastgs_pcd.ply'
    rotation_right = [
        [0, 2, 0],
        [0, 0, 2],
        [2, 0, 0],
    ]
    save_gs_ply_file_path = f'{home}/chLi/Dataset/GS/haizei_1_v4/rotated_fastgs_pcd.ply'
    overwrite = True

    rotateGSPlyFile(gs_ply_file_path, rotation_right, save_gs_ply_file_path, overwrite)
