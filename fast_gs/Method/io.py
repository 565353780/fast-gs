import os

from base_gs_trainer.Method.path import removeFile, createFileFolder

from fast_gs.Model.gs import GaussianModel


def loadGS(gs_ply_file_path: str, sh_degree: int=3) -> GaussianModel:
    if not os.path.exists(gs_ply_file_path):
        print('[ERROR][io::loadGS]')
        print('\t gs ply file not exist!')
        print('\t gs_ply_file_path:', gs_ply_file_path)
        return None

    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.load_ply(gs_ply_file_path)
    return gaussians


def saveGS(
    gaussians: GaussianModel,
    save_gs_ply_file_path: str,
    overwrite: bool=False,
) -> bool:
    if gaussians is None:
        print('[ERROR][io::saveGS]')
        print('\t gaussians is None!')
        return False

    if os.path.exists(save_gs_ply_file_path):
        if not overwrite:
            return True
        removeFile(save_gs_ply_file_path)

    createFileFolder(save_gs_ply_file_path)
    gaussians.save_ply(save_gs_ply_file_path)
    return True
