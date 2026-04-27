import torch
from torch import nn

from fast_gs.Model.gs import GaussianModel


@torch.no_grad()
def _computeKnnMeanDist(
    xyz: torch.Tensor,
    k: int,
    chunk_size: int = 4096,
) -> torch.Tensor:
    '''
    分块计算每个点到自身 k 近邻的平均距离 (排除自身)。
    返回形状 (N,) 的张量, 与 xyz 同 device/dtype。
    '''
    n = xyz.shape[0]
    k_eff = min(k, n - 1)
    mean_dist = torch.empty(n, dtype=xyz.dtype, device=xyz.device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = xyz[start:end]
        dist = torch.cdist(chunk, xyz, p=2)
        # 排除自身: 自身距离为 0, 通过 topk 取 k_eff+1 再丢掉最近的 1 个
        topk_vals, _ = torch.topk(dist, k=k_eff + 1, dim=1, largest=False)
        knn_vals = topk_vals[:, 1:]
        mean_dist[start:end] = knn_vals.mean(dim=1)

    return mean_dist


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


def removeFloatGS(
    gs: GaussianModel,
    k: int = 16,
    std_ratio: float = 2.0,
) -> bool:
    '''
    基于统计 kNN 距离剔除 GaussianModel 中的离群 (漂浮) 高斯。

    判定准则:
      - 对每个点计算其到 k 近邻的平均距离 d_i
      - 若 d_i > mean(d) + std_ratio * std(d), 则视为离群点

    剔除时优先调用 GaussianModel.prune_points (可同步 optimizer 状态);
    若 optimizer 尚未初始化 (例如刚 load_ply), 则手动同步切分各属性张量。
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
    if n < max(k + 1, 4):
        return True

    if std_ratio <= 0:
        print('[ERROR][filter::removeFloatGS]')
        print('\t std_ratio must be positive!')
        print('\t std_ratio:', std_ratio)
        return False

    with torch.no_grad():
        xyz = gs.get_xyz.detach()
        mean_knn = _computeKnnMeanDist(xyz, k=k)

        global_mean = mean_knn.mean()
        global_std = mean_knn.std()
        threshold = global_mean + std_ratio * global_std

        prune_mask = mean_knn > threshold

    if not bool(prune_mask.any()):
        return True

    valid_mask = ~prune_mask

    if getattr(gs, 'optimizer', None) is None:
        _manualPrune(gs, valid_mask)
    else:
        if not hasattr(gs, 'tmp_radii'):
            gs.tmp_radii = None
        gs.prune_points(prune_mask)

    return True
