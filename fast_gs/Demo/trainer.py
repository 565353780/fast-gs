import sys
sys.path.append('../base-trainer')
sys.path.append('../camera-control')

import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

from fast_gs.Method.time import getCurrentTime
from fast_gs.Module.trainer import Trainer


def demo():
    data_id = 'haizei_1_v4'

    home = os.environ['HOME']
    colmap_data_folder_path = home + '/chLi/Dataset/GS/' + data_id + '/colmap_normalized/'
    save_result_folder_path = home + '/chLi/Dataset/GS/' + data_id + '/fastgs/'

    trainer = Trainer(
        colmap_data_folder_path=colmap_data_folder_path,
        save_log_folder_path=save_result_folder_path + 'logs/' + getCurrentTime() + '/',
        save_result_folder_path=save_result_folder_path + 'results/' + getCurrentTime() + '/',
        test_freq=500,
        save_freq=500,
    )
    trainer.train(30000)
    return True
