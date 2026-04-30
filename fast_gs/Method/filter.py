import torch
import numpy as np

from torch import nn
from typing import Union

from camera_control.Method.data import toTensor

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


@torch.no_grad()
def searchFloatPointIdxs(
    points: Union[torch.Tensor, np.ndarray, list],
    k: int = 16,
    std_ratio: float = 2.0,
    bbox_scale: float = 1.1,
) -> np.ndarray:
    '''
    与 removeFloatGS 相同的几何规则, 返回判定为漂浮点 (应剔除) 的索引。
    对应坐标为 points[i] (i 为返回数组中的每个元素)。
    若无需剔除或点数过少, 返回 shape (0,) 的 int64 数组。
    '''
    if isinstance(points, torch.Tensor):
        xyz = points.detach()
    else:
        xyz = toTensor(points)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        return np.array([], dtype=np.int64)

    if not torch.is_floating_point(xyz):
        xyz = xyz.float()

    n = int(xyz.shape[0])
    if n < max(k + 1, 4):
        return np.array([], dtype=np.int64)

    if std_ratio <= 0 or bbox_scale <= 0:
        return np.array([], dtype=np.int64)

    mean_knn = _computeKnnMeanDist(xyz, k=k)

    global_mean = mean_knn.mean()
    global_std = mean_knn.std()
    threshold = global_mean + std_ratio * global_std

    knn_inlier_mask = mean_knn <= threshold

    if not bool(knn_inlier_mask.any()):
        return np.array([], dtype=np.int64)

    inlier_xyz = xyz[knn_inlier_mask]
    bbox_min = inlier_xyz.amin(dim=0)
    bbox_max = inlier_xyz.amax(dim=0)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    bbox_half = 0.5 * (bbox_max - bbox_min) * bbox_scale
    scaled_min = bbox_center - bbox_half
    scaled_max = bbox_center + bbox_half

    in_bbox_mask = ((xyz >= scaled_min) & (xyz <= scaled_max)).all(dim=1)
    prune_mask = ~in_bbox_mask

    if not bool(prune_mask.any()):
        return np.array([], dtype=np.int64)

    idx_t = torch.nonzero(prune_mask, as_tuple=False).squeeze(-1)
    return idx_t.cpu().numpy().astype(np.int64)


def removeFloatGS(
    gs: GaussianModel,
    k: int = 16,
    std_ratio: float = 2.0,
    bbox_scale: float = 1.1,
) -> bool:
    '''
    基于统计 kNN 距离与 bbox 膨胀策略剔除 GaussianModel 中的离群 (漂浮) 高斯。

    流程:
      1. 对每个点计算其到 k 近邻的平均距离 d_i;
         若 d_i > mean(d) + std_ratio * std(d), 则视为 kNN 意义下的离群点;
      2. 取 kNN 过滤后剩余点 (内点) 的索引, 计算其轴对齐包围盒 (AABB);
      3. 将该 bbox 围绕中心膨胀 bbox_scale 倍;
      4. 回到原始 GS, 仅删除位于膨胀后 bbox 之外的点
         (即保留 kNN 内点 + 落在 bbox 内的部分原 kNN 离群点)。

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

    if bbox_scale <= 0:
        print('[ERROR][filter::removeFloatGS]')
        print('\t bbox_scale must be positive!')
        print('\t bbox_scale:', bbox_scale)
        return False

    with torch.no_grad():
        xyz = gs.get_xyz.detach()
        float_idxs = searchFloatPointIdxs(
            xyz, k=k, std_ratio=std_ratio, bbox_scale=bbox_scale
        )

    if float_idxs.size == 0:
        return True

    device = xyz.device
    prune_mask = torch.zeros(n, dtype=torch.bool, device=device)
    prune_mask[torch.from_numpy(float_idxs).to(device=device, dtype=torch.long)] = True

    valid_mask = ~prune_mask

    if getattr(gs, 'optimizer', None) is None:
        _manualPrune(gs, valid_mask)
    else:
        if not hasattr(gs, 'tmp_radii'):
            gs.tmp_radii = None
        gs.prune_points(prune_mask)

    return True
