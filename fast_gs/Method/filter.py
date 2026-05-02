import torch
import numpy as np

from torch import nn

from camera_control.Method.filter import searchMainClusterPointMask

from fast_gs.Model.gs import GaussianModel


def _manualPrune(gs: GaussianModel, valid_mask: torch.Tensor) -> None:
    '''
    在 optimizer 尚未初始化的情况下, 直接对模型张量做切分,
    保持与 GaussianModel.prune_points 一致的语义。
    '''
    def _slice_param(param: torch.Tensor) -> nn.Parameter:
        new_data = param.data[valid_mask]
        new_param = nn.Parameter(new_data)
        new_param.requires_grad_(param.requires_grad)
        return new_param

    gs._xyz = _slice_param(gs._xyz)
    gs._features_dc = _slice_param(gs._features_dc)
    gs._features_rest = _slice_param(gs._features_rest)
    gs._opacity = _slice_param(gs._opacity)
    gs._scaling = _slice_param(gs._scaling)
    gs._rotation = _slice_param(gs._rotation)

    if gs.max_radii2D.numel() > 0:
        gs.max_radii2D = gs.max_radii2D[valid_mask]
    if gs.xyz_gradient_accum.numel() > 0:
        gs.xyz_gradient_accum = gs.xyz_gradient_accum[valid_mask]
    if gs.xyz_gradient_accum_abs.numel() > 0:
        gs.xyz_gradient_accum_abs = gs.xyz_gradient_accum_abs[valid_mask]
    if gs.denom.numel() > 0:
        gs.denom = gs.denom[valid_mask]

def removeFloatGS(gs: GaussianModel, **kwargs) -> bool:
    '''
    基于 "最近邻尺度连通分量 + bbox 有限膨胀" 的主簇提取剔除漂浮高斯。

    所有超参通过 kwargs 透传给 searchMainClusterPointMask, 具体语义见后者
    docstring。常用覆盖:
      cluster_distance_factor, bbox_expand_ratio, bbox_max_iters,
      debug, debug_folder_path

    剔除时优先调用 GaussianModel.prune_points (同步 optimizer 状态);
    optimizer 尚未初始化时 (例如刚 load_ply), 手动切分各属性张量。
    '''
    if gs is None:
        print('[ERROR][filter::removeFloatGS]')
        print('\t gs is None!')
        return False

    if gs._xyz is None or gs._xyz.numel() == 0:
        print('[ERROR][filter::removeFloatGS]')
        print('\t gs has no points!')
        return False

    n = gs.get_xyz.shape[0]
    if n < 4:
        return True

    try:
        with torch.no_grad():
            xyz = gs.get_xyz.detach()
            keep_np = searchMainClusterPointMask(xyz, **kwargs)
    except ValueError as e:
        print('[ERROR][filter::removeFloatGS]')
        print('\t invalid hyperparameter:', e)
        return False

    if keep_np.size != n or np.all(keep_np):
        return True

    device = xyz.device
    valid_mask = torch.from_numpy(keep_np).to(device=device, dtype=torch.bool)
    prune_mask = ~valid_mask

    if getattr(gs, 'optimizer', None) is None:
        _manualPrune(gs, valid_mask)
    else:
        if not hasattr(gs, 'tmp_radii'):
            gs.tmp_radii = None
        gs.prune_points(prune_mask)

    return True
