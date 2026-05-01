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
def searchMainClusterPointMask(
    points: Union[torch.Tensor, np.ndarray, list],
    k: int = 16,
    std_ratio: float = 2.0,
    bbox_scale: float = 1.1,
    max_iters: int = 256,
) -> np.ndarray:
    '''
    返回主簇掩码, 与输入 points 行顺序一一对应: True 为保留 (主簇), False 为漂浮点。

    流程:
      1. 计算每个点到 k 近邻的平均距离 d_i;
      2. 以稳健阈值 median(d) + std_ratio * 1.4826 * MAD(d) 选出主簇种子,
         比经典 mean+std 更抗少量极远点把统计量拉飞;
      3. 若种子退化 (MAD 近 0 或种子点数过少), 至少保留 kNN 距离最小的一半
         点作为种子, 避免空 bbox;
      4. 由种子构造初始 AABB, 记录初始边长 initial_extent;
      5. 反复将当前 bbox 沿三轴各向外扩张 0.5 * (bbox_scale - 1.0) * initial_extent,
         把落入扩张 bbox 的原始点全部纳入主簇, 再用主簇点重算 bbox 进入下一轮;
      6. 当没有新点被吸纳时收敛; 掩码中 True 为最终主簇。

    输入无效、点数过少或参数非法时, 返回与点数一致的 True 掩码 (不做剔除);
    无法确定点数时返回 shape (0,) 的 bool 数组。
    '''
    if isinstance(points, torch.Tensor):
        xyz = points.detach()
    else:
        xyz = toTensor(points)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        if xyz.ndim == 2 and int(xyz.shape[0]) > 0:
            return np.ones(int(xyz.shape[0]), dtype=np.bool_)
        return np.array([], dtype=np.bool_)

    if not torch.is_floating_point(xyz):
        xyz = xyz.float()

    n = int(xyz.shape[0])
    if n < max(k + 1, 4):
        return np.ones(n, dtype=np.bool_)

    if std_ratio <= 0 or bbox_scale <= 0:
        return np.ones(n, dtype=np.bool_)

    mean_knn = _computeKnnMeanDist(xyz, k=k)

    median_knn = mean_knn.median()
    mad = (mean_knn - median_knn).abs().median()
    robust_std = mad * 1.4826
    if float(robust_std) <= 0:
        # MAD 退化时回落到经典 std, 保证阈值不为 0
        robust_std = mean_knn.std()
    threshold = median_knn + std_ratio * robust_std

    seed_mask = mean_knn <= threshold

    # 兜底: 至少保留 kNN 距离最小的一半点作为种子, 避免极端情况下种子太小
    min_seed = max(k + 1, n // 2)
    if int(seed_mask.sum()) < min_seed:
        sorted_idx = torch.argsort(mean_knn)
        seed_mask = torch.zeros_like(seed_mask)
        seed_mask[sorted_idx[:min_seed]] = True

    seed_xyz = xyz[seed_mask]
    bbox_min = seed_xyz.amin(dim=0)
    bbox_max = seed_xyz.amax(dim=0)
    initial_extent = bbox_max - bbox_min

    # 每轮按初始边长的 (bbox_scale - 1.0) 比例扩张 bbox 总边长, 平均分到两侧
    half_step = 0.5 * max(bbox_scale - 1.0, 0.0) * initial_extent

    inside_mask = seed_mask.clone()
    inside_count = int(inside_mask.sum())

    for _ in range(max(int(max_iters), 1)):
        test_min = bbox_min - half_step
        test_max = bbox_max + half_step

        new_inside = ((xyz >= test_min) & (xyz <= test_max)).all(dim=1)
        new_count = int(new_inside.sum())
        if new_count <= inside_count:
            break

        inside_mask = new_inside
        inside_count = new_count
        kept_xyz = xyz[inside_mask]
        bbox_min = kept_xyz.amin(dim=0)
        bbox_max = kept_xyz.amax(dim=0)

    return inside_mask.cpu().numpy().astype(np.bool_)


def removeFloatGS(
    gs: GaussianModel,
    k: int = 16,
    std_ratio: float = 2.0,
    bbox_scale: float = 1.1,
) -> bool:
    '''
    基于稳健 kNN 种子 + 迭代 bbox 扩张策略剔除 GaussianModel 中的离群 (漂浮) 高斯。

    流程 (与 searchMainClusterPointMask 一致, 剔除 ~mask 部分):
      1. 对每个点计算其到 k 近邻的平均距离 d_i;
         以 median(d) + std_ratio * 1.4826 * MAD(d) 作为稳健阈值挑选主簇种子,
         必要时用最小 kNN 距离的一半点兜底, 抗少量极远漂浮点把均值/方差拉飞;
      2. 由种子点构造初始 AABB, 记录初始边长 initial_extent;
      3. 反复将当前 bbox 沿三轴各向外扩张 0.5 * (bbox_scale - 1.0) * initial_extent,
         把落入扩张 bbox 的点全部纳入主簇, 再用主簇点重算 bbox 进入下一轮;
      4. 当没有新点被吸纳时收敛, 仍未吸纳的点视为漂浮点, 从 GS 中剔除。

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
        keep_np = searchMainClusterPointMask(
            xyz, k=k, std_ratio=std_ratio, bbox_scale=bbox_scale
        )

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
