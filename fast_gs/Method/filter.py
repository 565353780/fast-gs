import math
import torch
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from torch import nn
from typing import Tuple, Union

from camera_control.Method.data import toTensor

from fast_gs.Model.gs import GaussianModel


@torch.no_grad()
def _computeKnnGraph(
    xyz: torch.Tensor,
    k: int,
    chunk_size: int = 4096,
    scale_mode: str = "median",
    scale_floor_ratio: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    分块计算每个点的 kNN 索引、距离以及局部尺度 local_scale (排除自身)。

    参数:
      scale_mode:
        "median" (推荐): 每点 kNN 距离的中位数, 最抗邻居中混入的远点;
        "kth":           第 k 近邻的距离 (HDBSCAN 意义下的 core distance);
        "mean":          简单算术平均, 受远邻居影响大, 仅用于对照。
      scale_floor_ratio:
        local_scale 的下限 = scale_floor_ratio * median(local_scale),
        防止重合/重复点导致 scale=0 进而让 edge_alpha * min(scale) 阈值过严。
        设为 0 以关闭该保护。

    返回:
      knn_idx: (N, k_eff) long;
      knn_dist: (N, k_eff) 与 xyz 同 dtype 的距离;
      local_scale: (N,) 每点局部尺度。
    '''
    n = xyz.shape[0]
    k_eff = min(k, max(n - 1, 0))
    knn_idx = torch.empty((n, k_eff), dtype=torch.long, device=xyz.device)
    knn_dist = torch.empty((n, k_eff), dtype=xyz.dtype, device=xyz.device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = xyz[start:end]
        dist = torch.cdist(chunk, xyz, p=2)
        # 排除自身: 自身距离为 0, 取 k_eff+1 后丢首列
        topk_vals, topk_idx = torch.topk(
            dist, k=k_eff + 1, dim=1, largest=False
        )
        knn_idx[start:end] = topk_idx[:, 1:]
        knn_dist[start:end] = topk_vals[:, 1:]

    if k_eff == 0:
        local_scale = torch.zeros(n, dtype=xyz.dtype, device=xyz.device)
    elif scale_mode == "median":
        local_scale = knn_dist.median(dim=1).values
    elif scale_mode == "kth":
        local_scale = knn_dist[:, -1].clone()
    elif scale_mode == "mean":
        local_scale = knn_dist.mean(dim=1)
    else:
        raise ValueError(f"unknown scale_mode: {scale_mode!r}")

    if k_eff > 0 and scale_floor_ratio > 0:
        scale_med = local_scale.median()
        floor = float(scale_floor_ratio) * float(scale_med)
        if floor > 0:
            local_scale = torch.clamp(local_scale, min=floor)

    return knn_idx, knn_dist, local_scale


@torch.no_grad()
def _largestKnnComponentMask(
    knn_idx: torch.Tensor,
    knn_dist: torch.Tensor,
    local_scale: torch.Tensor,
    edge_alpha: float,
    edge_scale_reduce: str = "min",
    mutual_knn: bool = True,
) -> torch.Tensor:
    '''
    基于 kNN 图构造无向连通分量, 返回最大分量的布尔 mask。

    边 (i, j) 的保留条件:
        dist(i, j) <= edge_alpha * f(local_scale[i], local_scale[j])
      其中 f:
        "min" (推荐): 抗孤立漂浮点 (自身 scale 偏大时阈值依旧由对侧收紧);
        "max":        更保守, 容易把近邻噪声团一起合并, 仅在已知主体密度极
                      不均时启用。

    mutual_knn=True 时, 进一步要求 j ∈ kNN(i) 且 i ∈ kNN(j) 才连边, 能显著
    压制 "密集噪声团 -> 主簇" 的单向伪连接 (因为主体点的 kNN 名单里通常不会
    出现远处噪声点)。
    '''
    n = knn_idx.shape[0]
    k_eff = knn_idx.shape[1] if knn_idx.ndim == 2 else 0
    device = knn_idx.device

    if n <= 1 or k_eff <= 0:
        return torch.ones(n, dtype=torch.bool, device=device)

    src = (
        torch.arange(n, device=device).unsqueeze(1).expand(n, k_eff).reshape(-1)
    )
    dst = knn_idx.reshape(-1)
    dist = knn_dist.reshape(-1)

    scale_src = local_scale[src]
    scale_dst = local_scale[dst]
    if edge_scale_reduce == "min":
        scale_pair = torch.minimum(scale_src, scale_dst)
    elif edge_scale_reduce == "max":
        scale_pair = torch.maximum(scale_src, scale_dst)
    else:
        raise ValueError(
            f"unknown edge_scale_reduce: {edge_scale_reduce!r}"
        )

    thr = float(edge_alpha) * scale_pair
    keep = dist <= thr
    src_np = src[keep].detach().cpu().numpy().astype(np.int64, copy=False)
    dst_np = dst[keep].detach().cpu().numpy().astype(np.int64, copy=False)

    if src_np.size == 0:
        return torch.zeros(n, dtype=torch.bool, device=device)

    data = np.ones(src_np.size, dtype=np.uint8)
    graph = csr_matrix((data, (src_np, dst_np)), shape=(n, n))

    if mutual_knn:
        # (i,j) AND (j,i) 同时存在 -> 0/1 矩阵的 element-wise multiply
        graph = graph.multiply(graph.T)
        graph.eliminate_zeros()
        if graph.nnz == 0:
            return torch.zeros(n, dtype=torch.bool, device=device)

    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels)
    largest = int(np.argmax(counts))
    mask_np = labels == largest
    return torch.from_numpy(mask_np).to(device=device)


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
    k_adaptive: bool = True,
    chunk_size: int = 4096,
    scale_mode: str = "median",
    scale_floor_ratio: float = 1e-3,
    edge_alpha: float = 1.5,
    edge_scale_reduce: str = "min",
    mutual_knn: bool = True,
    min_component_points: int = 32,
    min_component_ratio: float = 0.01,
    fallback_ratio: float = 0.5,
    bbox_step_mode: str = "scale",
    bbox_step_k: float = 2.0,
    bbox_scale: float = 1.1,
    bbox_max_iters: int = 64,
) -> np.ndarray:
    '''
    返回主簇掩码: True 为保留 (主簇), False 为漂浮点。算法 = kNN 图最大连通
    分量选种子 + AABB 迭代膨胀吸纳表面点。

    ------- kNN -------
      k, k_adaptive, chunk_size
      k_adaptive=True 时实际使用 min(k, max(8, round(sqrt(N)))), 小点云自动
      降 k 以避免把过大范围的邻居拉进统计。

    ------- 局部尺度 -------
      scale_mode in {"median","kth","mean"} (默认 median, 最稳健);
      scale_floor_ratio 为 local_scale 相对下限, 防重复点让边阈值塌缩。

    ------- 图边阈值 -------
      dist(i,j) <= edge_alpha * f(scale_i, scale_j),
      edge_scale_reduce in {"min","max"} (默认 min, 抗孤立漂浮点);
      mutual_knn=True 进一步要求双向 kNN 命中 (抗密集噪声团伪连接)。

    ------- 主簇 -------
      取最大连通分量。若其点数 <
          max(k+1, min_component_points, round(min_component_ratio * N))
      则判定图已退化, 退化到 local_scale 最小的前 round(fallback_ratio * N) 个点。

    ------- BBox 膨胀 -------
      bbox_step_mode:
        "scale"  (默认, 推荐): half_step = bbox_step_k * median(seed_local_scale);
                               与初始 bbox 解耦, 大小物体都能稳定扩张。
        "extent" (旧行为):     half_step = 0.5*(bbox_scale-1) * initial_extent;
                               若初始种子只覆盖物体一小角, 收敛慢但可控。
      bbox_max_iters 为迭代上限 (通常 5-15 轮即收敛)。

    输入无效或点数过少时, 返回全 True 掩码 (不做剔除);
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
    if n < max(int(k) + 1, 4):
        return np.ones(n, dtype=np.bool_)

    if edge_alpha <= 0 or bbox_scale <= 0 or bbox_step_k <= 0:
        return np.ones(n, dtype=np.bool_)

    k_used = int(k)
    if k_adaptive and n > 1:
        k_used = max(4, min(k_used, max(8, int(round(math.sqrt(n))))))

    knn_idx, knn_dist, local_scale = _computeKnnGraph(
        xyz,
        k=k_used,
        chunk_size=int(chunk_size),
        scale_mode=scale_mode,
        scale_floor_ratio=float(scale_floor_ratio),
    )

    seed_mask = _largestKnnComponentMask(
        knn_idx=knn_idx,
        knn_dist=knn_dist,
        local_scale=local_scale,
        edge_alpha=float(edge_alpha),
        edge_scale_reduce=edge_scale_reduce,
        mutual_knn=bool(mutual_knn),
    )

    # 回退: 最大连通分量过小, 视为图结构退化
    min_component = max(
        k_used + 1,
        int(min_component_points),
        int(round(float(min_component_ratio) * n)),
    )
    if int(seed_mask.sum()) < min_component:
        fallback_n = max(k_used + 1, int(round(float(fallback_ratio) * n)))
        fallback_n = min(fallback_n, n)
        sorted_idx = torch.argsort(local_scale)
        seed_mask = torch.zeros(n, dtype=torch.bool, device=xyz.device)
        seed_mask[sorted_idx[:fallback_n]] = True

    seed_xyz = xyz[seed_mask]
    bbox_min = seed_xyz.amin(dim=0)
    bbox_max = seed_xyz.amax(dim=0)

    if bbox_step_mode == "scale":
        seed_scale_med = local_scale[seed_mask].median()
        half_step = float(bbox_step_k) * seed_scale_med
    elif bbox_step_mode == "extent":
        initial_extent = bbox_max - bbox_min
        half_step = 0.5 * max(float(bbox_scale) - 1.0, 0.0) * initial_extent
    else:
        raise ValueError(f"unknown bbox_step_mode: {bbox_step_mode!r}")

    inside_mask = seed_mask.clone()
    inside_count = int(inside_mask.sum())

    for _ in range(max(int(bbox_max_iters), 1)):
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


def removeFloatGS(gs: GaussianModel, **kwargs) -> bool:
    '''
    基于 kNN 图主簇 + AABB 膨胀剔除 GaussianModel 中的漂浮高斯。

    所有超参通过 kwargs 透传给 searchMainClusterPointMask, 具体语义见后者 docstring。
    常用覆盖:
      k, edge_alpha, mutual_knn, min_component_ratio, bbox_step_k, bbox_max_iters

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
    k = int(kwargs.get('k', 16))
    if n < max(k + 1, 4):
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
