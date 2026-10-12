"""AMOD 旋转目标检测 学生模型 student_v2 —— 单模态 RGB + 两个创新模块（自包含）。

相对 student_v1 的唯一结构差异：**AGF 不再作用于 P16**。v1 里 P16 同时被 OFE 与 AGF
串联处理（OFE 输出再喂给 AGF），两个模块在同一层共享并相互耦合；v2 把两者的作用层
彻底分开，各自只在自己的层级独立生效：

    OFE -> 浅层高分辨率 P8 / P16（方向性细节最丰富）
    AGF -> 深层低分辨率 P32 / P64（逐像素网格采样开销可控）

结构沿用 model_v8.py 的底座（原生尺度主干 + 双向金字塔 Neck + 归一化角度头 +
环形角度损失），但只保留**单模态 RGB（cfg.modality='eo'）**，并把 v8 的跨模态融合
单元替换为两个面向单模态旋转目标检测的新模块：

    创新一 OFE（Oriented Frequency Enhancement，方向解耦频域增强）：
        把特征投影成单通道显著性图，用 K 个方向的 Gabor 带通核提取**方向性**高频
        响应，再由可学习的方向门控合成"旋转结构先验"，以残差方式调制回特征。
        与 v6/v8 里各向同性的低/高频线索不同，这里显式把"方向"与"频率"解耦，
        专门增强坦克 / 舰船 / 飞机等细长旋转目标的方向性轮廓。

    创新二 AGF（Angle-Guided Aggregation，角度引导条带聚合）：
        预测逐位置主轴方向与采样步长，沿该主轴用 grid_sample 可变形采样若干点，
        用可学习门控加权聚合，为细长目标补轴向上下文；角度场额外加平滑正则。
        它把"旋转框的角度"从检测头输出前移到特征聚合阶段，让特征表示自带朝向先验。

底座（非创新，直接沿用 v8）：RGBDetailPyramid 细节金字塔、自顶向下 + 自底向上
双向 Neck、NormOBBHead 角度通道 L2 归一化、环形角度损失。

本文件完全自包含：几何算子、标签分配、LoRA、细节金字塔、创新模块、Neck、检测头、
损失与总装全部在本文件内，只依赖 PyTorch 与 third_party/dinov3，不引用其他 model_vN。
"""


import copy
import math
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 本文件位于 <工程根>/model/ 下，third_party 在工程根目录
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THIRD_PARTY = os.path.join(_ROOT, "third_party")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

from dinov3.hub.backbones import dinov3_convnext_tiny  # noqa: E402


# ===========================================================================
# 1. 旋转框几何算子
# ===========================================================================

MAXV = 8


def corners_to_obb(corners, min_size=2.0):
    """四角点 -> le90 旋转框。

    Args:
        corners: (N, 4, 2) 或 (N, 8)，按顺序环绕（如 左上->右上->右下->左下）
        min_size: 边长小于该值的框会被丢弃（与官方 AMOD 设置一致）

    Returns:
        obbs: (M, 5) float32 [cx, cy, w, h, angle]
        keep: (N,) bool，哪些输入框被保留
    """
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 4, 2)
    if corners.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=bool)

    p1, p2, p3 = corners[:, 0], corners[:, 1], corners[:, 2]
    e1, e2 = p2 - p1, p3 - p2
    len1 = np.linalg.norm(e1, axis=1)
    len2 = np.linalg.norm(e2, axis=1)

    w, h = len1, len2
    angle = np.arctan2(e1[:, 1], e1[:, 0])

    # 让 w 始终是长边，对应的角度加 90 度
    swap = w < h
    w, h = np.where(swap, h, w), np.where(swap, w, h)
    angle = np.where(swap, angle + np.pi / 2, angle)
    angle = (angle + np.pi / 2) % np.pi - np.pi / 2  # 归一化到 [-pi/2, pi/2)

    center = corners.mean(axis=1)
    obbs = np.stack([center[:, 0], center[:, 1], w, h, angle], axis=1).astype(np.float32)

    keep = (w >= float(min_size)) & (h >= float(min_size))
    return obbs[keep], keep


def obb_to_corners_np(obbs):
    """(N, 5) le90 旋转框 -> (N, 4, 2) 四角点（环绕顺序）。"""
    obbs = np.asarray(obbs, dtype=np.float64).reshape(-1, 5)
    cx, cy, w, h, angle = [obbs[:, i] for i in range(5)]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    half_w, half_h = w / 2.0, h / 2.0
    # 沿长边方向的单位向量 u=(cos,sin)，短边方向 v=(-sin,cos)
    ux, uy = cos_a * half_w, sin_a * half_w
    vx, vy = -sin_a * half_h, cos_a * half_h
    signs = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=np.float64)
    corners = np.stack(
        [
            cx[:, None] + signs[None, :, 0] * ux[:, None] + signs[None, :, 1] * vx[:, None],
            cy[:, None] + signs[None, :, 0] * uy[:, None] + signs[None, :, 1] * vy[:, None],
        ],
        axis=-1,
    )
    return corners


def obb_to_corners_torch(obbs):
    """(N, 5) le90 旋转框 -> (N, 4, 2) 四角点（torch 版本，可导）。"""
    cx, cy, w, h, angle = obbs.unbind(-1)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    half_w, half_h = w / 2.0, h / 2.0
    ux, uy = cos_a * half_w, sin_a * half_w
    vx, vy = -sin_a * half_h, cos_a * half_h
    signs = obbs.new_tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
    sx, sy = signs[:, 0], signs[:, 1]
    xs = cx[:, None] + sx[None, :] * ux[:, None] + sy[None, :] * vx[:, None]
    ys = cy[:, None] + sx[None, :] * uy[:, None] + sy[None, :] * vy[:, None]
    return torch.stack([xs, ys], dim=-1)


def obb_to_hbb(obbs):
    """旋转框 -> 水平外接框 [x1, y1, x2, y2]（用于尺寸分组等粗略用途）。"""
    corners = obb_to_corners_torch(obbs) if torch.is_tensor(obbs) else obb_to_corners_np(obbs)
    if torch.is_tensor(corners):
        return torch.cat([corners.min(dim=1).values, corners.max(dim=1).values], dim=-1)
    return np.concatenate([corners.min(axis=1), corners.max(axis=1)], axis=-1)


def _signed_area_quad(quads):
    """(N, 4, 2) -> (N,) 有符号面积（用于判断环绕方向）。"""
    x, y = quads[..., 0], quads[..., 1]
    total = (
        x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]
        + x[:, 1] * y[:, 2] - x[:, 2] * y[:, 1]
        + x[:, 2] * y[:, 3] - x[:, 3] * y[:, 2]
        + x[:, 3] * y[:, 0] - x[:, 0] * y[:, 3]
    )
    return 0.5 * total


def _polygon_area(polys, counts):
    """(N, MAXV, 2) + 有效顶点数 -> (N,) 面积（Shoelace）。"""
    x, y = polys[..., 0], polys[..., 1]
    acc = (x[:, :-1] * y[:, 1:] - x[:, 1:] * y[:, :-1]).sum(dim=-1)
    last_idx = (counts - 1).clamp(min=0)
    last = polys[torch.arange(polys.shape[0], device=polys.device), last_idx]
    acc = acc + (last[:, 0] * polys[:, 0, 1] - last[:, 1] * polys[:, 0, 0])
    return 0.5 * acc.abs()


def _write(out, mask, index, value):
    """在 out[mask, index[mask]] 处写入 value[mask]，并丢弃越界位置。"""
    if mask.any():
        safe_index = index.clamp(max=MAXV - 1)
        out[mask, safe_index[mask]] = value[mask]


def intersection_area(poly_a, poly_b):
    """两个凸四边形的精确交集面积（矢量化）。

    Args:
        poly_a: (N, 4, 2) 被裁剪多边形（凸、环绕顺序）
        poly_b: (N, 4, 2) 裁剪多边形（凸、环绕顺序）

    Returns:
        (N,) 交集面积
    """
    device = poly_a.device
    n = poly_a.shape[0]
    ar = torch.arange(n, device=device)

    # 统一裁剪多边形的环绕方向：保证“每条有向边的左侧为内部”
    flip = _signed_area_quad(poly_b) < 0
    if flip.any():
        poly_b = torch.where(flip[:, None, None], poly_b[:, [0, 3, 2, 1]], poly_b)

    # 平移到裁剪框首个角点的局部坐标系：面积与位置无关，但坐标量级
    # 从整图尺寸降到框尺寸，可显著降低 float32 的舍入误差
    origin = poly_b[:, 0:1, :]
    poly_a = poly_a - origin
    poly_b = poly_b - origin

    subject = poly_a.new_zeros(n, MAXV, 2)
    subject[:, :4] = poly_a
    counts = torch.full((n,), 4, dtype=torch.long, device=device)
    eps = 1e-9

    for edge in range(4):
        a = poly_b[:, edge]
        b = poly_b[:, (edge + 1) % 4]
        e = b - a
        normal = torch.stack([-e[:, 1], e[:, 0]], dim=-1)  # 左法线
        dist = ((subject - a[:, None, :]) * normal[:, None, :]).sum(-1)  # (N, MAXV)
        inside = dist >= 0

        out = subject.new_zeros(n, MAXV, 2)
        new_counts = torch.zeros_like(counts)
        valid_poly = counts >= 2

        # (1) 顺序边 j -> j+1
        for j in range(MAXV - 1):
            valid = (j < counts - 1)
            cur_in = inside[:, j] & valid
            nxt_in = inside[:, j + 1] & valid
            d_cur, d_nxt = dist[:, j], dist[:, j + 1]
            cur, nxt = subject[:, j], subject[:, j + 1]

            denom = d_cur - d_nxt
            safe_denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)
            t = (d_cur / safe_denom).unsqueeze(-1)
            crossing = cur_in ^ nxt_in

            idx_i = new_counts
            idx_n = new_counts + (~cur_in).long()
            _write(out, crossing, idx_i, cur + t * (nxt - cur))
            _write(out, nxt_in, idx_n, nxt)
            new_counts = new_counts + nxt_in.long() + crossing.long()

        # (2) 环绕边 v_{count-1} -> v_0（顺时针遍历的最后一条边）
        last_idx = (counts - 1).clamp(min=0)
        cur_in = inside[ar, last_idx] & valid_poly
        nxt_in = inside[:, 0] & valid_poly
        d_cur, d_nxt = dist[ar, last_idx], dist[:, 0]
        cur, nxt = subject[ar, last_idx], subject[:, 0]

        denom = d_cur - d_nxt
        safe_denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)
        t = (d_cur / safe_denom).unsqueeze(-1)
        crossing = cur_in ^ nxt_in

        idx_i = new_counts
        idx_n = new_counts + (~cur_in).long()
        _write(out, crossing, idx_i, cur + t * (nxt - cur))
        _write(out, nxt_in, idx_n, nxt)
        new_counts = new_counts + nxt_in.long() + crossing.long()

        subject = out
        counts = new_counts.clamp(max=MAXV)

    return _polygon_area(subject, counts.clamp(min=0, max=MAXV))


def rotated_iou_matrix(boxes_a, boxes_b, eps=1e-7):
    """两组 le90 旋转框之间的精确 IoU 矩阵。

    Args:
        boxes_a: (Na, 5)
        boxes_b: (Nb, 5)

    Returns:
        (Na, Nb)
    """
    boxes_a = boxes_a.reshape(-1, 5)
    boxes_b = boxes_b.reshape(-1, 5)
    na, nb = boxes_a.shape[0], boxes_b.shape[0]
    if na == 0 or nb == 0:
        return boxes_a.new_zeros(na, nb)

    corners_a = obb_to_corners_torch(boxes_a)[:, None].expand(na, nb, 4, 2).reshape(-1, 4, 2)
    corners_b = obb_to_corners_torch(boxes_b)[None].expand(na, nb, 4, 2).reshape(-1, 4, 2)

    inter = intersection_area(corners_a, corners_b).reshape(na, nb)
    area_a = (boxes_a[:, 2] * boxes_a[:, 3]).reshape(na, 1)
    area_b = (boxes_b[:, 2] * boxes_b[:, 3]).reshape(1, nb)
    union = area_a + area_b - inter
    return (inter / union.clamp(min=eps)).clamp(0.0, 1.0)


def rotated_iou(boxes_a, boxes_b, eps=1e-7):
    """逐对计算旋转框 IoU（长度需一致）。"""
    assert boxes_a.shape[0] == boxes_b.shape[0]
    return rotated_iou_matrix(boxes_a, boxes_b).diagonal()


def nms_rotated(boxes, scores, iou_threshold=0.5):
    """单类别旋转框 NMS（贪心，基于精确 IoU 矩阵）。

    Args:
        boxes: (N, 5) le90
        scores: (N,)
        iou_threshold: float

    Returns:
        keep: (K,) long，按分数降序
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    iou = rotated_iou_matrix(boxes, boxes)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        current = order[0]
        keep.append(int(current))
        if order.numel() == 1:
            break
        rest = order[1:]
        order = rest[iou[current, rest] <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def batched_nms_rotated(boxes, scores, labels, iou_threshold=0.5, max_num=300):
    """多类别旋转 NMS：等价于 mmcv 的 nms_rotated + batched_nms。

    Args:
        boxes: (N, 5)
        scores: (N,)
        labels: (N,)
        iou_threshold: float
        max_num: 每张图最多保留的框数

    Returns:
        keep: (K,) long
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    kept = []
    for cls_id in labels.unique():
        cls_mask = labels == cls_id
        cls_keep = nms_rotated(
            boxes[cls_mask], scores[cls_mask], iou_threshold=iou_threshold
        )
        kept.append(cls_mask.nonzero(as_tuple=False).squeeze(1)[cls_keep])
    if not kept:
        return boxes.new_zeros((0,), dtype=torch.long)

    keep = torch.cat(kept, dim=0)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep[:max_num]


# ===========================================================================
# 2. 标签分配（FCOS 式 center sampling + 逐层尺度划分）
# ===========================================================================

def make_locations(height, width, stride, device, dtype=torch.float32):
    """生成特征图上每个位置对应的原图坐标（像素中心）。

    Returns:
        (H*W, 2)，元素为 (x, y)
    """
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def _axis_vectors(angles):
    """(G,) 角度 -> 单位向量 u (G,2) 与 v (G,2)。"""
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    u = torch.stack([cos_a, sin_a], dim=-1)
    v = torch.stack([-sin_a, cos_a], dim=-1)
    return u, v


def encode_obb_targets(locations, gt_boxes):
    """计算所有 (特征点, GT) 组合的局部坐标与到四边的距离。

    Args:
        locations: (N, 2)
        gt_boxes: (G, 5)

    Returns:
        a: (N, G) 沿 GT 长边方向的局部坐标
        b: (N, G) 沿 GT 短边方向的局部坐标
        inside: (N, G) 点是否落在 GT 矩形内部
        ltrb: (N, G, 4) 到四边（沿长短边方向）的距离
    """
    u, v = _axis_vectors(gt_boxes[:, 4])           # (G,2)
    delta = locations[:, None, :] - gt_boxes[None, :, :2]  # (N,G,2)
    a = (delta * u[None]).sum(-1)
    b = (delta * v[None]).sum(-1)

    half_w = gt_boxes[None, :, 2] / 2.0
    half_h = gt_boxes[None, :, 3] / 2.0
    inside = (a.abs() <= half_w) & (b.abs() <= half_h)

    du1, du2 = half_w + a, half_w - a
    dv1, dv2 = half_h + b, half_h - b
    ltrb = torch.stack([du1, du2, dv1, dv2], dim=-1)
    return a, b, inside, ltrb


def assign_single_image(
    locations,
    stride,
    gt_boxes,
    gt_labels,
    scale_range,
    center_radius=1.5,
):
    """单张图、单个特征层的标签分配。

    Args:
        locations: (N, 2) 原图坐标
        stride: int，该层步长
        gt_boxes: (G, 5) le90 旋转框（已过滤掉无效框）
        gt_labels: (G,)
        scale_range: (lo, hi) 该层负责的目标尺寸区间（sqrt(w*h)）
        center_radius: 中心采样半径（单位为 stride），仅中心附近位置参与回归

    Returns:
        cls_targets: (N,) long，-1 表示负样本
        box_targets: (N, 5) 正样本对应的 GT 框（负样本为 0）
        ltrb_targets: (N, 4) 正样本对应的四边距离
        pos_mask: (N,) bool
    """
    num_loc = locations.shape[0]
    device = locations.device
    cls_targets = torch.full((num_loc,), -1, dtype=torch.long, device=device)
    box_targets = torch.zeros(num_loc, 5, device=device)
    ltrb_targets = torch.zeros(num_loc, 4, device=device)
    pos_mask = torch.zeros(num_loc, dtype=torch.bool, device=device)

    if gt_boxes.numel() == 0:
        return cls_targets, box_targets, ltrb_targets, pos_mask

    a, b, inside, ltrb = encode_obb_targets(locations, gt_boxes)

    size = (gt_boxes[:, 2] * gt_boxes[:, 3]).sqrt()          # (G,)
    size_ok = (size >= scale_range[0]) & (size < scale_range[1])
    center_ok = (a.abs() <= center_radius * stride) & (b.abs() <= center_radius * stride)
    valid = inside & center_ok & size_ok[None, :]            # (N, G)

    if not valid.any():
        return cls_targets, box_targets, ltrb_targets, pos_mask

    # 命中多个 GT 时取面积最小者（FCOS 策略）
    areas = gt_boxes[:, 2] * gt_boxes[:, 3]
    masked_areas = torch.where(valid, areas[None, :], torch.full_like(a, float("inf")))
    best_gt = masked_areas.argmin(dim=1)                     # (N,)
    pos_mask = masked_areas.min(dim=1).values < float("inf")

    idx = best_gt[pos_mask]
    cls_targets[pos_mask] = gt_labels[idx]
    box_targets[pos_mask] = gt_boxes[idx]
    ltrb_targets[pos_mask] = ltrb[pos_mask][torch.arange(idx.numel(), device=device), idx]
    return cls_targets, box_targets, ltrb_targets, pos_mask


def decode_obb(locations, reg_preds, stride):
    """把回归输出解码成 le90 旋转框。

    Args:
        locations: (N, 2)
        reg_preds: (N, 6) = (raw_du1, raw_du2, raw_dv1, raw_dv2, sin2θ, cos2θ)
        stride: int 或 (N,)，允许每个样本使用不同步长（多尺度混合时使用）

    Returns:
        (N, 5) le90 旋转框
    """
    stride = torch.as_tensor(stride, device=reg_preds.device, dtype=reg_preds.dtype)
    if stride.ndim == 0:
        stride = stride.expand(reg_preds.shape[0])
    else:
        stride = stride.reshape(-1)

    du1 = F.softplus(reg_preds[:, 0]) * stride
    du2 = F.softplus(reg_preds[:, 1]) * stride
    dv1 = F.softplus(reg_preds[:, 2]) * stride
    dv2 = F.softplus(reg_preds[:, 3]) * stride

    width = du1 + du2
    height = dv1 + dv2
    off_a = (du1 - du2) / 2.0
    off_b = (dv1 - dv2) / 2.0

    angle = 0.5 * torch.atan2(reg_preds[:, 4], reg_preds[:, 5])
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    # 中心 = 特征点 - (局部偏移) 沿 GT 轴回投
    cx = locations[:, 0] - off_a * cos_a + off_b * sin_a
    cy = locations[:, 1] - off_a * sin_a - off_b * cos_a
    return torch.stack([cx, cy, width, height, angle], dim=-1)


def build_targets(
    locations_list,
    strides,
    gt_boxes,
    gt_labels,
    scale_ranges,
    center_radius=1.5,
):
    """对多尺度特征层批量做标签分配。

    Args:
        locations_list: list[(N_i, 2)]，每层的原图坐标
        strides: list[int]
        gt_boxes: (G, 5) 该图 GT（已是当前输入尺寸下的坐标）
        gt_labels: (G,)
        scale_ranges: list[(lo, hi)]，与层一一对应
        center_radius: float

    Returns:
        list[dict]，每层包含 cls_targets / box_targets / ltrb_targets / pos_mask
    """
    outputs = []
    for locations, stride, scale_range in zip(locations_list, strides, scale_ranges):
        cls_t, box_t, ltrb_t, pos = assign_single_image(
            locations, stride, gt_boxes, gt_labels, scale_range, center_radius
        )
        outputs.append(
            {
                "cls_targets": cls_t,
                "box_targets": box_t,
                "ltrb_targets": ltrb_t,
                "pos_mask": pos,
            }
        )
    return outputs


# ===========================================================================
# 3. 微调与构件：LoRA + 细节金字塔 + 空间精修 + 两个创新模块（OFE / AGF）
# ===========================================================================

def make_group_norm(channels, max_groups=32):
    """创建通道数可整除的 GroupNorm。"""
    max_groups = min(max_groups, channels)
    for groups in range(max_groups, 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class LoRALinear(nn.Module):
    """在线性层旁路加入可训练低秩增量，原始线性层保持冻结。"""

    def __init__(self, original_linear, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank 必须大于 0")

        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = float(alpha) / float(rank)

        for param in self.original_linear.parameters():
            param.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.in_features = in_features
        self.out_features = out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_output = self.original_linear(x)
        low_rank = F.linear(self.lora_dropout(x), self.lora_A)
        low_rank = F.linear(low_rank, self.lora_B)
        return original_output + low_rank * self.scaling


def inject_lora(module, rank=16, alpha=32, dropout=0.1, suffixes=("pwconv1", "pwconv2")):
    """把模块中指定名字的 Linear 子层替换为标准 LoRA 旁路，返回替换的层数。

    学生模型只有单模态 RGB，不存在"模态解耦"的需求，因此统一注入 LoRALinear。
    """
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(name.endswith(s) for s in suffixes):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            count += 1
        else:
            count += inject_lora(child, rank=rank, alpha=alpha, dropout=dropout, suffixes=suffixes)
    return count


class DetailResidualBlock(nn.Module):
    """轻量深度可分离残差块，用于构建真实空间细节层级。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1,
                      groups=in_channels, bias=False),
            make_group_norm(in_channels, max_groups=16),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            make_group_norm(out_channels, max_groups=16),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1,
                      groups=out_channels, bias=False),
            make_group_norm(out_channels, max_groups=16),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            make_group_norm(out_channels, max_groups=16),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                make_group_norm(out_channels, max_groups=16),
            )
        else:
            self.shortcut = nn.Identity()
        self.residual_gate_logit = nn.Parameter(torch.tensor(-0.5))
        self.activation = nn.GELU()

    def forward(self, feature):
        residual = self.main(feature)
        shortcut = self.shortcut(feature)
        return self.activation(shortcut + torch.sigmoid(self.residual_gate_logit) * residual)


class RGBDetailPyramid(nn.Module):
    """从原始图像构建 1/2、1/4、1/8 三个真实空间尺度（backbone 缺失的高频细节）。"""

    def __init__(self, in_channels=3, channels=(48, 72, 96)):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("detail channels 必须包含 1/2、1/4、1/8 三个通道数")
        c2, c4, c8 = [int(value) for value in channels]
        self.channels = (c2, c4, c8)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c2, 3, stride=2, padding=1, bias=False),
            make_group_norm(c2, max_groups=16),
            nn.GELU(),
            DetailResidualBlock(c2, c2, stride=1),
        )
        self.stage4 = DetailResidualBlock(c2, c4, stride=2)
        self.stage8 = DetailResidualBlock(c4, c8, stride=2)

    def forward(self, image):
        detail_2 = self.stem(image)
        detail_4 = self.stage4(detail_2)
        detail_8 = self.stage8(detail_4)
        return detail_2, detail_4, detail_8


class SeparableRefine(nn.Module):
    """深度可分离的空间精修：残差门控初值很小，训练初期几乎不扰动输入特征。"""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            make_group_norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            make_group_norm(channels),
        )
        self.gate = nn.Parameter(torch.tensor(-1.5))

    def forward(self, feature):
        return F.gelu(feature + torch.sigmoid(self.gate) * self.block(feature))


# ===========================================================================
# 3.1 创新模块一：OFE —— 方向解耦频域增强
# ===========================================================================

def _oriented_bandpass_bank(num_orientations, kernel_size, sigma=1.6, gamma=0.5, frequency=0.22):
    """构造 K 个方向的 Gabor 型带通卷积核（零均值、L1 归一化），返回 (K, 1, k, k)。"""
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    bank = []
    for index in range(int(num_orientations)):
        theta = math.pi * index / float(num_orientations)
        x_rot = xx * math.cos(theta) + yy * math.sin(theta)
        y_rot = -xx * math.sin(theta) + yy * math.cos(theta)
        envelope = torch.exp(-(x_rot ** 2 + (gamma * y_rot) ** 2) / (2.0 * sigma ** 2))
        wave = torch.cos(2.0 * math.pi * frequency * x_rot)
        kernel = envelope * wave
        kernel = kernel - kernel.mean()          # 去直流 => 真正的带通
        norm = kernel.abs().sum()
        if norm > 0:
            kernel = kernel / norm
        bank.append(kernel.reshape(1, 1, kernel_size, kernel_size))
    return torch.cat(bank, dim=0)


class OrientedFrequencyEnhancement(nn.Module):
    """OFE：方向解耦频域增强（创新模块一）。

    1. 先把特征压成单通道显著性图（1x1 conv + sigmoid），保留空间结构；
    2. 用 K 个方向的 Gabor 带通核对显著性图做卷积，得到 K 张方向性高频响应
       （取 abs 能量，与方向正负无关）；
    3. 由响应本身经一个小卷积网算出逐像素的方向门控（softmax over K），
       加权合成"旋转结构先验"，再用 3x3 卷积精修；
    4. 以残差方式调制回特征：out = feature * (1 + sigmoid(gate) * tanh(prior))。

    与 v6 / v8 各向同性低/高频线索的区别：这里显式引入**方向**维度，把"方向"与
    "频率"解耦，专门增强坦克 / 舰船 / 飞机等细长旋转目标的方向性轮廓。
    """

    def __init__(self, channels, num_orientations=4, kernel_size=7, residual_gate_init=-0.5):
        super().__init__()
        self.num_orientations = int(num_orientations)
        self.kernel_size = int(kernel_size)
        self.saliency = nn.Sequential(nn.Conv2d(channels, 1, 1, bias=False), nn.Sigmoid())
        self.register_buffer(
            "bank", _oriented_bandpass_bank(self.num_orientations, self.kernel_size)
        )

        gate_hidden = max(8, self.num_orientations * 2)
        self.gate = nn.Sequential(
            nn.Conv2d(self.num_orientations, gate_hidden, 1, bias=False),
            make_group_norm(gate_hidden, 16), nn.GELU(),
            nn.Conv2d(gate_hidden, self.num_orientations, 1),
        )
        # 零初始化 => 起点各方向等权，训练中再学出偏好
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        self.prior = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            make_group_norm(16, 16), nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))

    def forward(self, feature):
        saliency = self.saliency(feature)
        padding = self.kernel_size // 2
        responses = [
            F.conv2d(saliency, self.bank[index:index + 1], padding=padding).abs()
            for index in range(self.num_orientations)
        ]
        stack = torch.cat(responses, dim=1)                  # (B, K, H, W)
        weight = torch.softmax(self.gate(stack), dim=1)      # (B, K, H, W)
        prior = self.prior((weight * stack).sum(dim=1, keepdim=True))
        modulation = torch.sigmoid(self.residual_gate) * torch.tanh(prior)
        out = feature * (1.0 + modulation)
        stats = {
            "oriented_gate": weight.mean(dim=(0, 2, 3)).detach(),
            "oriented_energy": stack.mean().detach(),
        }
        return out, stats


# ===========================================================================
# 3.2 创新模块二：AGF —— 角度引导条带聚合
# ===========================================================================

class AngleGuidedAggregation(nn.Module):
    """AGF：角度引导条带聚合（创新模块二）。

    1. 用一个小卷积头预测逐位置的主轴方向 theta 与采样步长 step（都用 tanh 有界）；
    2. 沿主轴方向 [cosθ, sinθ] 在 ±step 处用 grid_sample 可变形采样 K 个点；
    3. 由特征再算一组逐像素的采样权重（softmax over K），加权聚合得到轴向上下文；
    4. 以残差方式叠加回特征：out = feature + sigmoid(gate) * proj(aggregated)。

    角度场额外输出平滑正则项，避免逐像素角度高频抖动（与 v7 的位移场平滑同理）。
    它把"旋转框的角度"从检测头输出**前移**到特征聚合阶段，让特征表示自带朝向先验。
    """

    def __init__(self, channels, num_samples=3, max_step=4.0, residual_gate_init=-1.0):
        super().__init__()
        self.num_samples = max(2, int(num_samples))
        self.max_step = float(max_step)
        hidden = max(32, int(channels) // 4)

        self.axis = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            make_group_norm(hidden, 16), nn.GELU(),
            nn.Conv2d(hidden, 2, 1),
        )
        # 零初始化 => 起点 theta=0、step=0，聚合退化为恒等映射，不破坏初始特征
        nn.init.zeros_(self.axis[-1].weight)
        nn.init.zeros_(self.axis[-1].bias)

        self.weight = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            make_group_norm(hidden, 16), nn.GELU(),
            nn.Conv2d(hidden, self.num_samples, 1),
        )
        nn.init.zeros_(self.weight[-1].weight)
        nn.init.zeros_(self.weight[-1].bias)

        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False), make_group_norm(channels),
        )
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))

    @staticmethod
    def _base_grid(height, width, device):
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        return x.unsqueeze(0), y.unsqueeze(0)

    def forward(self, feature):
        _, _, height, width = feature.shape
        params = self.axis(feature)
        theta = math.pi * torch.tanh(params[:, :1])           # (B, 1, H, W) 主轴角度
        step = self.max_step * torch.tanh(params[:, 1:2])     # (B, 1, H, W) 采样步长

        scale_x = 2.0 / max(width - 1, 1)
        scale_y = 2.0 / max(height - 1, 1)
        base_x, base_y = self._base_grid(height, width, feature.device)

        axis_x = torch.cos(theta)[:, 0]                       # (B, H, W)
        axis_y = torch.sin(theta)[:, 0]
        step_2d = step[:, 0]
        offsets = torch.arange(self.num_samples, device=feature.device, dtype=torch.float32)
        offsets = offsets - (self.num_samples - 1) / 2.0      # 关于中心对称

        weight = torch.softmax(self.weight(feature), dim=1)   # (B, K, H, W)
        aggregated = torch.zeros_like(feature)
        for index, offset in enumerate(offsets):
            delta_x = (offset * step_2d * axis_x) * scale_x
            delta_y = (offset * step_2d * axis_y) * scale_y
            grid = torch.stack([base_x + delta_x, base_y + delta_y], dim=-1)
            sampled = F.grid_sample(
                feature.float(), grid, mode="bilinear",
                padding_mode="border", align_corners=True,
            ).to(feature.dtype)
            aggregated = aggregated + weight[:, index:index + 1] * sampled

        out = feature + torch.sigmoid(self.residual_gate) * self.project(aggregated)

        smooth = torch.zeros((), device=feature.device, dtype=torch.float32)
        if width > 1:
            smooth = smooth + (theta[..., 1:] - theta[..., :-1]).float().abs().mean()
        if height > 1:
            smooth = smooth + (theta[..., 1:, :] - theta[..., :-1, :]).float().abs().mean()
        stats = {
            "angle_field_smoothness": smooth,
            "angle_field_scale": theta.abs().mean().detach(),
        }
        return out, stats


# ===========================================================================
# 3.3 创新模块总装：按层级作用到 Neck 输出
# ===========================================================================

class StudentInnovationRefiner(nn.Module):
    """把两个创新模块按层级作用到 Neck 输出的金字塔特征上（v2：两层不重叠）。

    - OFE 只作用在高分辨率层（默认 P8 / P16），那里方向性细节最丰富；
    - AGF 作用在较深层（默认 P32 / P64），分辨率低、逐像素网格采样开销可控。

    v2 取消了 v1 里 P16 的 AGF，两个模块作用层互不重叠：同一层不会被 OFE 与 AGF
    串联处理，因此每层特征的增强路径是单一且独立的。
    """

    def __init__(self, cfg, channels, num_levels=4):
        super().__init__()
        self.num_levels = int(num_levels)
        self.ofe_levels = tuple(int(i) for i in getattr(cfg, "ofe_levels", (0, 1)))
        self.agf_levels = tuple(int(i) for i in getattr(cfg, "agf_levels", (2, 3)))

        num_orientations = int(getattr(cfg, "ofe_num_orientations", 4))
        ofe_kernel = int(getattr(cfg, "ofe_kernel_size", 7))
        num_samples = int(getattr(cfg, "agf_num_samples", 3))
        max_step = float(getattr(cfg, "agf_max_step", 4.0))

        self.ofe = nn.ModuleList([
            OrientedFrequencyEnhancement(
                channels, num_orientations=num_orientations, kernel_size=ofe_kernel,
            )
            if index in self.ofe_levels else None
            for index in range(self.num_levels)
        ])
        self.agf = nn.ModuleList([
            AngleGuidedAggregation(channels, num_samples=num_samples, max_step=max_step)
            if index in self.agf_levels else None
            for index in range(self.num_levels)
        ])

    def forward(self, features):
        outputs = []
        smoothness, scale, energy, gate = [], [], [], []
        for index, feature in enumerate(features):
            if self.ofe[index] is not None:
                feature, ofe_stats = self.ofe[index](feature)
                energy.append(ofe_stats["oriented_energy"].float())
                gate.append(ofe_stats["oriented_gate"].float())
            if self.agf[index] is not None:
                feature, agf_stats = self.agf[index](feature)
                smoothness.append(agf_stats["angle_field_smoothness"].float())
                scale.append(agf_stats["angle_field_scale"].float())
            outputs.append(feature)

        aux = {}
        if smoothness:
            aux["angle_field_smoothness"] = torch.stack(smoothness).mean()
            aux["angle_field_scale"] = torch.stack(scale).mean()
        if energy:
            aux["oriented_energy"] = torch.stack(energy).mean()
        if gate:
            # 各层方向门控分布（K 维向量）取平均，用于观察 OFE 学到的方向偏好
            aux["oriented_gate"] = torch.stack(gate).mean(dim=0)
        return outputs, aux


# ===========================================================================
# 4. Neck：原生尺度金字塔（DINO stride 4/8/16/32 -> 检测 8/16/32/64）
# ===========================================================================

class WeightedMerge(nn.Module):
    """快速归一化的非负加权特征合并（BiFPN 的合并算子）。"""

    def __init__(self, num_inputs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))

    def forward(self, features):
        weight = F.relu(self.weight)
        weight = weight / (weight.sum() + 1e-4)
        out = torch.zeros_like(features[0])
        for coefficient, feature in zip(weight, features):
            out = out + coefficient * feature
        return out


class NativePyramidNeck(nn.Module):
    """原生尺度金字塔 Neck：DINO 原生 stride 4/8/16/32 -> 检测 stride 8/16/32/64。

    P8 = C8 + 上采样的 P16 + 1/8 细节 + 1/4 细节下采样得到的 1/8 语义，
    P32 直接来自 C32（不再由 P16 下采样得到），P64 再由 P32 下采样得到。
    自顶向下 + 自底向上双向：P16 / P32 在自顶向下之后再接收来自浅层的定位反馈，
    防止深层语义把小目标冲淡。
    """

    def __init__(self, cfg, in_channels_list):
        super().__init__()
        if list(map(int, cfg.fpn_strides)) != [8, 16, 32, 64]:
            raise ValueError("student_v2 要求 fpn_strides=[8, 16, 32, 64]")
        self.bidirectional = True
        dim = int(cfg.feature_dim)
        self.dim = dim
        self.lateral = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, dim, 1, bias=False), make_group_norm(dim), nn.GELU())
            for channels in in_channels_list
        ])
        self.fine_to_8 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, groups=dim, bias=False),
            make_group_norm(dim), nn.GELU(), nn.Conv2d(dim, dim, 1, bias=False), make_group_norm(dim),
        )

        image_channels = 6 if cfg.modality.lower() == "eoir" else 3
        self.detail = RGBDetailPyramid(image_channels, cfg.detail_channels)
        detail_8_channels = int(cfg.detail_channels[2])
        self.detail_to_8 = nn.Sequential(
            nn.Conv2d(detail_8_channels, dim, 1, bias=False), make_group_norm(dim), nn.GELU()
        )

        self.merge_8 = WeightedMerge(3)
        self.merge_16 = WeightedMerge(2)
        self.merge_32 = WeightedMerge(2)
        self.out_refine = nn.ModuleList([SeparableRefine(dim) for _ in range(4)])
        self.down_16 = nn.Conv2d(dim, dim, 3, stride=2, padding=1, groups=dim, bias=False)
        self.down_32 = nn.Conv2d(dim, dim, 3, stride=2, padding=1, groups=dim, bias=False)
        self.down_64 = nn.Conv2d(dim, dim, 3, stride=2, padding=1, groups=dim, bias=False)
        self.down_proj_16 = nn.Conv2d(dim, dim, 1, bias=False)
        self.down_proj_32 = nn.Conv2d(dim, dim, 1, bias=False)
        self.down_proj_64 = nn.Conv2d(dim, dim, 1, bias=False)

        # 自顶向下之后再自底向上，防止深层语义把极小目标冲淡
        self.pan_16 = WeightedMerge(2)
        self.pan_32 = WeightedMerge(2)

    @staticmethod
    def _resize(feature, reference):
        return F.interpolate(feature, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, features, image):
        c4, c8, c16, c32 = [layer(feature) for layer, feature in zip(self.lateral, features)]
        _, _, detail_8 = self.detail(image)
        detail_8 = self.detail_to_8(detail_8)
        fine_8 = self.fine_to_8(c4)
        if fine_8.shape[-2:] != c8.shape[-2:]:
            fine_8 = self._resize(fine_8, c8)

        # 自顶向下语义传播，同时保留原生 P8
        td32 = c32
        td16 = self.merge_16([c16, self._resize(td32, c16)])
        td8 = self.merge_8([c8, self._resize(td16, c8), fine_8 + detail_8])

        # 自底向上的定位反馈：防止深层语义把极小目标冲淡
        bu16 = self.pan_16([td16, self.down_proj_16(self.down_16(td8))])
        bu32 = self.pan_32([td32, self.down_proj_32(self.down_32(bu16))])
        p8, p16, p32 = td8, bu16, bu32
        p64 = self.down_proj_64(self.down_64(p32))

        outputs = [p8, p16, p32, p64]
        outputs = [refine(feature) for refine, feature in zip(self.out_refine, outputs)]
        # 保持旧的日志口径：每个 DINO stage 一个归一化贡献值
        layer_weights = image.new_full((image.shape[0], 4), 0.25)
        return outputs, layer_weights


# ===========================================================================
# 5. 检测头：角度通道 L2 归一化
# ===========================================================================

class ConvGN(nn.Module):
    """Conv3x3(+GN+GELU) 的轻量封装。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            make_group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class OBBHead(nn.Module):
    def __init__(self, cfg, in_channels, num_levels, prior_prob=0.01):
        super().__init__()
        self.num_classes = int(cfg.num_classes)
        self.num_levels = int(num_levels)
        self.num_convs = int(cfg.num_convs)
        # 逐层卷积塔层数（P8 分辨率最高，只放 1 层以节省算力）
        per_level = getattr(cfg, "head_convs_per_level", None)
        self.convs_per_level = (
            [int(value) for value in per_level] if per_level is not None
            else [self.num_convs] * self.num_levels
        )
        if len(self.convs_per_level) != self.num_levels:
            raise ValueError("head_convs_per_level 长度必须等于特征层数")

        self.cls_towers = nn.ModuleList()
        self.reg_towers = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.ctr_preds = nn.ModuleList()

        for level in range(self.num_levels):
            num_convs = max(1, self.convs_per_level[level])
            cls_tower, reg_tower = [], []
            for _ in range(num_convs):
                cls_tower.append(ConvGN(in_channels, in_channels))
                reg_tower.append(ConvGN(in_channels, in_channels))
            self.cls_towers.append(nn.Sequential(*cls_tower))
            self.reg_towers.append(nn.Sequential(*reg_tower))
            self.cls_preds.append(nn.Conv2d(in_channels, self.num_classes, 3, padding=1))
            self.reg_preds.append(nn.Conv2d(in_channels, 6, 3, padding=1))
            self.ctr_preds.append(nn.Conv2d(in_channels, 1, 3, padding=1))

        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob):
        for module_list in (self.cls_preds, self.reg_preds, self.ctr_preds):
            for conv in module_list:
                nn.init.normal_(conv.weight, std=0.01)
                nn.init.constant_(conv.bias, 0.0)
        # 分类先验：初始正样本概率很低，避免训练早期被负样本淹没
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for conv in self.cls_preds:
            nn.init.constant_(conv.bias, bias_value)

    def forward(self, features):
        """features: list[(B, C, H_l, W_l)] -> dict(cls/reg/ctr 三个 list)"""
        cls_outputs, reg_outputs, ctr_outputs = [], [], []
        for level, feature in enumerate(features):
            cls_feature = self.cls_towers[level](feature)
            reg_feature = self.reg_towers[level](feature)
            cls_outputs.append(self.cls_preds[level](cls_feature))
            reg_outputs.append(self.reg_preds[level](reg_feature))
            ctr_outputs.append(self.ctr_preds[level](reg_feature))
        return {"cls": cls_outputs, "reg": reg_outputs, "ctr": ctr_outputs}


class NormOBBHead(OBBHead):
    def __init__(self, cfg, in_channels, num_levels, prior_prob=0.01):
        super().__init__(cfg, in_channels, num_levels, prior_prob)
        self.angle_eps = float(getattr(cfg, "angle_norm_eps", 1e-4))

    def forward(self, features):
        outputs = super().forward(features)
        normalized = []
        for reg in outputs["reg"]:
            # 先转 fp32 再归一化：bf16 下 sqrt 的有效位太少，方向会有明显抖动
            angle = F.normalize(reg[:, 4:6].float(), dim=1, eps=self.angle_eps).to(reg.dtype)
            normalized.append(torch.cat([reg[:, :4], angle], dim=1))
        outputs["reg"] = normalized
        return outputs


# ===========================================================================
# 6. 主干：DINOv3 ConvNeXt-Tiny 冻结权重 + 标准 LoRA（单模态 RGB）
# ===========================================================================

_LORA_SUFFIXES = ("pwconv1", "pwconv2")


class _CheckpointBlock(nn.Module):
    """ConvNeXt Block 的包装：按需启用梯度检查点（单模态，无需下发 modality）。"""

    def __init__(self, block, use_checkpoint=False):
        super().__init__()
        self.block = block
        self.use_checkpoint = bool(use_checkpoint)

    def forward(self, x):
        if not self.use_checkpoint:
            return self.block(x)
        return torch.utils.checkpoint.checkpoint(self.block, x, use_reentrant=False)


def _build_stem(pretrained_stem, in_chans):
    """用预训练 stem（Conv k=4 s=4 + LayerNorm）构造输入投影。

    输入通道数与预训练一致时直接复制权重；若是多通道拼接（如 6 通道 EO+IR），
    按通道组复制并把权重除以组数，保证多通道求和后的响应量级与预训练一致。
    """
    conv, norm = pretrained_stem[0], pretrained_stem[1]
    new_conv = nn.Conv2d(
        in_chans,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        source = conv.weight
        if in_chans == source.shape[1]:
            new_conv.weight.copy_(source)
        else:
            groups = in_chans // source.shape[1]
            if groups * source.shape[1] != in_chans:
                raise ValueError(f"输入通道数 {in_chans} 不是预训练通道数 {source.shape[1]} 的整数倍")
            share = source / float(groups)
            for group in range(groups):
                start = group * source.shape[1]
                new_conv.weight[:, start:start + source.shape[1]] = share
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return nn.Sequential(new_conv, copy.deepcopy(norm))


class SingleModalBackbone(nn.Module):
    """DINOv3 ConvNeXt-Tiny 主干（冻结权重 + 标准 LoRA），单模态 RGB。

    只有一路 RGB，因此没有跨模态融合、质量描述子与门控。主干权重冻结，只训练输入
    stem 与 LoRA 旁路；输出 4 个 stage 的原生尺度特征（stride 4/8/16/32）。
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.convnext = dinov3_convnext_tiny(
            pretrained=bool(cfg.backbone_pretrained),
            weights=cfg.backbone_weights,
        )
        self.stage_dims = [int(dim) for dim in self.convnext.embed_dims]
        self.num_stages = len(self.stage_dims)
        # Neck 按这个通道数逐层建侧向投影，因此允许每层通道数不同
        self.out_channels = list(self.stage_dims)

        if str(cfg.modality).lower() not in ("eo", "rgb"):
            raise ValueError("SingleModalBackbone 只支持单模态 RGB（modality='eo'）")

        # ---- 输入投影（从预训练 stem 复制，可训练）----
        pretrained_stem = self.convnext.downsample_layers[0]
        self.stem = _build_stem(pretrained_stem, 3)
        # 原始 stem 不再使用（输入侧已改为单分支）
        self.convnext.downsample_layers[0] = nn.Identity()

        # ---- 冻结主干，只训练输入投影与 LoRA 旁路 ----
        for param in self.convnext.parameters():
            param.requires_grad = False
        trainable_input = bool(getattr(cfg, "trainable_patch_embed", True))
        for param in self.stem.parameters():
            param.requires_grad = trainable_input

        # ---- 标准 LoRA（单模态，无模态解耦旁路）----
        self.num_lora_layers = 0
        if cfg.use_lora:
            self.num_lora_layers = inject_lora(
                self.convnext,
                rank=int(cfg.lora_rank),
                alpha=int(cfg.lora_alpha),
                dropout=float(cfg.lora_dropout),
                suffixes=_LORA_SUFFIXES,
            )

        # 统一包装：按需启用梯度检查点
        for stage in self.convnext.stages:
            for index, block in enumerate(stage):
                stage[index] = _CheckpointBlock(block, use_checkpoint=bool(cfg.grad_checkpoint))

    def trainable_parameters(self):
        """返回需要优化的参数（配合 LoRA，可单独设置学习率）。"""
        return [p for p in self.parameters() if p.requires_grad]

    def _run_stage(self, stage, feature):
        """按顺序跑完一个 stage 的 ConvNeXt blocks。"""
        for block in stage:
            feature = block(feature)
        return feature

    def forward(self, images: torch.Tensor):
        """返回 (list[(B, C_i, H_i, W_i)], aux)。"""
        feature = self.stem(images)
        features = []
        for index in range(self.num_stages):
            if index > 0:
                feature = self.convnext.downsample_layers[index](feature)
            feature = self._run_stage(self.convnext.stages[index], feature)
            features.append(feature)
        return features, {}


# ===========================================================================
# 7. 损失：基础损失 + 环形角度损失 + AGF 角度场平滑正则
# ===========================================================================

def sigmoid_focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction="sum"):
    """标准 Sigmoid Focal Loss。

    Args:
        pred: (N, C) logits
        target: (N, C) {0,1}
    """
    pred_sigmoid = pred.sigmoid()
    target = target.to(dtype=pred.dtype)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def centerness_target(ltrb):
    """FCOS centerness 目标：sqrt( (min(l,r)/max(l,r)) * (min(t,b)/max(t,b)) )。

    Args:
        ltrb: (N, 4) = (du1, du2, dv1, dv2)
    """
    left_right = ltrb[:, [0, 1]]
    top_bottom = ltrb[:, [2, 3]]
    ctr = (left_right.min(dim=-1).values / left_right.max(dim=-1).values.clamp(min=1e-6)) * (
        top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values.clamp(min=1e-6)
    )
    return torch.sqrt(ctr.clamp(min=0.0))


class AMODLoss(torch.nn.Module):
    """AMOD 旋转检测任务的总损失。"""

    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.strides = list(cfg.fpn_strides)
        self.scale_ranges = [tuple(r) for r in cfg.scale_ranges]
        self.center_radius = float(cfg.center_sampling_radius)

        self.focal_alpha = float(getattr(cfg, "focal_alpha", 0.25))
        self.focal_gamma = float(getattr(cfg, "focal_gamma", 2.0))
        self.w_cls = float(getattr(cfg, "loss_cls_weight", 1.0))
        self.w_iou = float(getattr(cfg, "loss_iou_weight", 1.0))
        self.w_ltrb = float(getattr(cfg, "loss_ltrb_weight", 0.5))
        self.w_angle = float(getattr(cfg, "loss_angle_weight", 0.5))
        self.w_ctr = float(getattr(cfg, "loss_centerness_weight", 1.0))
        self.ltrb_beta = float(getattr(cfg, "loss_ltrb_beta", 0.1))
        self.angle_beta = float(getattr(cfg, "loss_angle_beta", 0.1))

    # ------------------------------------------------------------------ #
    def forward(self, preds, gt_boxes_list, gt_labels_list):
        """计算总损失。

        Args:
            preds: dict，含
                cls: list[(B, C, H, W)]，reg: list[(B, 6, H, W)]，
                ctr: list[(B, 1, H, W)]，locations/strides 与上面同序
            gt_boxes_list: list[(G_i, 5)]，与 batch 对应（当前输入尺寸下的坐标）
            gt_labels_list: list[(G_i,)]

        Returns:
            loss: 标量；parts: dict 各分项数值（用于日志）
        """
        device = preds["cls"][0].device
        batch_size = preds["cls"][0].shape[0]

        num_pos = 0
        cls_loss = torch.zeros((), device=device)
        iou_loss = torch.zeros((), device=device)
        ltrb_loss = torch.zeros((), device=device)
        angle_loss = torch.zeros((), device=device)
        ctr_loss = torch.zeros((), device=device)

        for b in range(batch_size):
            gt_boxes = gt_boxes_list[b].to(device)
            gt_labels = gt_labels_list[b].to(device)
            valid = gt_boxes[:, 2] > 1e-3
            gt_boxes, gt_labels = gt_boxes[valid], gt_labels[valid]

            targets = build_targets(
                preds["locations"],
                preds["strides"],
                gt_boxes,
                gt_labels,
                self.scale_ranges,
                self.center_radius,
            )

            # ---- 分类：所有位置 ----
            cls_logits, cls_targets = [], []
            for level, (logits, target) in enumerate(zip(preds["cls"], targets)):
                logits = logits[b].permute(1, 2, 0).reshape(-1, self.num_classes)
                cls_logits.append(logits)
                onehot = torch.zeros_like(logits)
                pos = target["pos_mask"]
                if pos.any():
                    onehot[pos, target["cls_targets"][pos]] = 1.0
                cls_targets.append(onehot)
            cls_loss = cls_loss + sigmoid_focal_loss(
                torch.cat(cls_logits), torch.cat(cls_targets),
                alpha=self.focal_alpha, gamma=self.focal_gamma,
            )

            # ---- 回归 + 质量：仅正样本 ----
            pos_regs, pos_locs, pos_box_t, pos_ltrb_t, pos_stride = [], [], [], [], []
            for level, (reg, stride) in enumerate(zip(preds["reg"], preds["strides"])):
                mask = targets[level]["pos_mask"]
                if not mask.any():
                    continue
                reg_flat = reg[b].permute(1, 2, 0).reshape(-1, 6)[mask]
                loc_flat = preds["locations"][level][mask]
                pos_regs.append(reg_flat)
                pos_locs.append(loc_flat)
                pos_box_t.append(targets[level]["box_targets"][mask])
                pos_ltrb_t.append(targets[level]["ltrb_targets"][mask])
                pos_stride.append(torch.full((reg_flat.shape[0],), stride, device=device))

            if pos_regs:
                reg_all = torch.cat(pos_regs)
                loc_all = torch.cat(pos_locs)
                box_all = torch.cat(pos_box_t)
                ltrb_all = torch.cat(pos_ltrb_t)
                stride_all = torch.cat(pos_stride)
                num_pos += reg_all.shape[0]

                # 逐样本 stride 解码（多尺度正样本混合）
                # 几何运算统一转 fp32：混合精度下 bf16 只有 ~3 位有效数字，
                # 会让旋转 IoU 的数值误差明显变大
                pred_boxes = decode_obb(loc_all.float(), reg_all.float(), stride_all.float())

                iou = rotated_iou(pred_boxes, box_all.float())
                ctr_t = centerness_target(ltrb_all)
                weight = ctr_t if self.w_ctr > 0 else torch.ones_like(ctr_t)

                iou_loss = iou_loss + ((1.0 - iou) * weight).sum()

                ltrb_pred = F.softplus(reg_all[:, :4]) * stride_all[:, None]
                ltrb_loss = ltrb_loss + (
                    F.smooth_l1_loss(
                        ltrb_pred / (stride_all[:, None] * 4.0),
                        ltrb_all / (stride_all[:, None] * 4.0),
                        beta=self.ltrb_beta, reduction="none",
                    ).sum(-1) * weight
                ).sum()

                angle_pred = reg_all[:, 4:6]
                angle_target = torch.stack(
                    [torch.sin(2 * box_all[:, 4]), torch.cos(2 * box_all[:, 4])], dim=-1
                )
                angle_loss = angle_loss + (
                    F.smooth_l1_loss(
                        angle_pred, angle_target, beta=self.angle_beta, reduction="none"
                    ).sum(-1) * weight
                ).sum()

            ctr_logits, ctr_targets = [], []
            for level in range(len(preds["strides"])):
                mask = targets[level]["pos_mask"]
                if not mask.any():
                    continue
                ctr_logits.append(preds["ctr"][level][b].permute(1, 2, 0).reshape(-1)[mask])
                ctr_targets.append(centerness_target(targets[level]["ltrb_targets"][mask]))
            if ctr_logits:
                ctr_loss = ctr_loss + F.binary_cross_entropy_with_logits(
                    torch.cat(ctr_logits), torch.cat(ctr_targets), reduction="sum"
                )

        norm = max(num_pos, 1)
        total = (
            self.w_cls * cls_loss
            + self.w_iou * iou_loss
            + self.w_ltrb * ltrb_loss
            + self.w_angle * angle_loss
            + self.w_ctr * ctr_loss
        ) / norm

        parts = {
            "loss": float(total.detach()),
            "loss_cls": float(self.w_cls * cls_loss.detach() / norm),
            "loss_iou": float(self.w_iou * iou_loss.detach() / norm),
            "loss_ltrb": float(self.w_ltrb * ltrb_loss.detach() / norm),
            "loss_angle": float(self.w_angle * angle_loss.detach() / norm),
            "loss_ctr": float(self.w_ctr * ctr_loss.detach() / norm),
            "num_pos": num_pos,
        }
        return total, parts


class AMODLossV4(AMODLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.w_circle = float(getattr(cfg, "loss_angle_circle_weight", 0.0))

    def forward(self, preds, gt_boxes_list, gt_labels_list):
        total, parts = super().forward(preds, gt_boxes_list, gt_labels_list)

        if self.w_circle > 0:
            device = preds["cls"][0].device
            circle_sum = torch.zeros((), device=device)
            num_pos = 0

            for batch_index, (gt_boxes, gt_labels) in enumerate(
                zip(gt_boxes_list, gt_labels_list)
            ):
                gt_boxes = gt_boxes.to(device)
                gt_labels = gt_labels.to(device)
                valid = gt_boxes[:, 2] > 1e-3
                gt_boxes, gt_labels = gt_boxes[valid], gt_labels[valid]

                targets = build_targets(
                    preds["locations"], preds["strides"], gt_boxes, gt_labels,
                    self.scale_ranges, self.center_radius,
                )
                for level in range(len(preds["strides"])):
                    mask = targets[level]["pos_mask"]
                    if not mask.any():
                        continue
                    reg_pos = preds["reg"][level][batch_index].permute(1, 2, 0).reshape(-1, 6)[mask]
                    box_target = targets[level]["box_targets"][mask]
                    ltrb_target = targets[level]["ltrb_targets"][mask]
                    num_pos += reg_pos.shape[0]

                    weight = centerness_target(ltrb_target)
                    angle_target = torch.stack(
                        [torch.sin(2 * box_target[:, 4]), torch.cos(2 * box_target[:, 4])], dim=-1
                    )
                    # 预测与目标都是单位向量 => 内积就是 cos(2Δθ)
                    cos_sim = (reg_pos[:, 4:6].float() * angle_target).sum(dim=-1)
                    circle_sum = circle_sum + ((1.0 - cos_sim) * weight).sum()

            if num_pos > 0:
                circle_loss = self.w_circle * circle_sum / num_pos
                total = total + circle_loss
                parts["loss"] = float(total.detach())
                parts["loss_circle"] = float(circle_loss.detach())
            else:
                parts["loss_circle"] = 0.0

        return total, parts


class AMODLossStudent(AMODLossV4):
    """学生模型损失：检测损失 + 环形角度损失（AMODLossV4）+ AGF 角度场平滑正则。

    单模态没有门控 / 融合一致性，因此只保留共享底座（环形角度损失）并新增角度场平滑。
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.w_angle_field = float(getattr(cfg, "loss_angle_field_smooth_weight", 0.005))

    def forward(self, preds, gt_boxes_list, gt_labels_list):
        total, parts = super().forward(preds, gt_boxes_list, gt_labels_list)
        aux = preds.get("aux") if isinstance(preds.get("aux"), dict) else None
        if not aux:
            return total, parts

        if self.w_angle_field > 0 and "angle_field_smoothness" in aux:
            value = self.w_angle_field * aux["angle_field_smoothness"].float()
            total = total + value
            parts["loss_angle_field"] = float(value.detach())
        # 记录创新模块的统计量（仅日志，不参与反传）
        for key in ("angle_field_scale", "oriented_energy", "oriented_gate"):
            if key in aux:
                parts[key] = float(aux[key].float().mean().detach())
        parts["loss"] = float(total.detach())
        return total, parts


# ===========================================================================
# 8. 总装：backbone + neck + head + criterion + 解码与旋转 NMS
# ===========================================================================

class AMODDetector(nn.Module):
    """学生模型 student_v2：单模态 RGB 主干 + 双向金字塔 Neck + OFE + AGF + 归一化角度头。

    与 student_v1 的唯一差异：AGF 只作用在 P32 / P64，不再作用于 P16（P16 仅由 OFE 处理）。
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(cfg.num_classes)
        self.strides = [int(s) for s in cfg.fpn_strides]
        self.num_levels = len(self.strides)

        self.backbone = SingleModalBackbone(cfg)
        self.neck = NativePyramidNeck(cfg, self.backbone.out_channels)
        # 两个创新模块：OFE（方向解耦频域增强）+ AGF（角度引导条带聚合）
        self.refiner = StudentInnovationRefiner(cfg, int(cfg.feature_dim), num_levels=self.num_levels)
        self.head = NormOBBHead(cfg, int(cfg.feature_dim), self.num_levels)
        self.criterion = AMODLossStudent(cfg)

        self._location_cache = {}


    def locations_for(self, feats, device):
        """缓存各层坐标栅格（同一特征尺寸下复用，避免每个 iteration 重建）。

        栅格尺寸以检测头实际输出的特征图为准，而不是 height // stride：Neck 的 P64 由
        stride=2 卷积从 P32 下采样得到（行数按 ceil 计），输入边长不能被 64 整除时它会比
        height // stride 多一行。若栅格按整除尺寸生成，推理时用特征图算出的候选下标会越过
        栅格上界（CUDA index out of bounds）。
        """
        shapes = tuple((int(feat.shape[-2]), int(feat.shape[-1])) for feat in feats)
        key = (shapes, str(device))
        cached = self._location_cache.get(key)
        if cached is None:
            cached = [
                make_locations(level_h, level_w, stride, device)
                for (level_h, level_w), stride in zip(shapes, self.strides)
            ]
            self._location_cache[key] = cached
        return cached


    def extract_features(self, images):
        features, _ = self.backbone(images)
        feats, layer_weights = self.neck(features, images)
        feats, innov_aux = self.refiner(feats)
        return feats, layer_weights, innov_aux


    def forward(self, images):
        features, layer_weights, aux = self.extract_features(images)
        preds = self.head(features)
        preds["locations"] = self.locations_for(preds["cls"], images.device)
        preds["strides"] = self.strides
        preds["layer_weights"] = layer_weights
        preds["aux"] = aux
        return preds


    @torch.no_grad()
    def inference(self, images) -> List[dict]:
        """对一批图像做前向 + 解码 + 多类别旋转 NMS。"""
        preds = self(images)
        cfg = self.cfg
        height, width = images.shape[-2:]
        batch_size = images.shape[0]
        results = []

        for batch_index in range(batch_size):
            boxes_per_level, scores_per_level, labels_per_level = [], [], []
            for level, stride in enumerate(self.strides):
                cls_logits = preds["cls"][level][batch_index].permute(1, 2, 0).reshape(-1, self.num_classes)
                reg_preds = preds["reg"][level][batch_index].permute(1, 2, 0).reshape(-1, 6)
                ctr_logits = preds["ctr"][level][batch_index].permute(1, 2, 0).reshape(-1)

                cls_scores = torch.sigmoid(cls_logits)
                centerness = torch.sigmoid(ctr_logits)
                # 与 FCOS 一致：用 centerness 抑制远离目标中心的低质量预测
                scores, labels = (cls_scores * centerness[:, None]).max(dim=-1)

                keep = scores > cfg.score_thr
                if not keep.any():
                    continue
                keep = keep.nonzero(as_tuple=False).squeeze(1)
                if keep.numel() > cfg.nms_pre_per_level:
                    topk = scores[keep].topk(cfg.nms_pre_per_level).indices
                    keep = keep[topk]

                locations = preds["locations"][level][keep]
                level_boxes = decode_obb(locations, reg_preds[keep], stride)
                boxes_per_level.append(level_boxes)
                scores_per_level.append(scores[keep])
                labels_per_level.append(labels[keep])

            if not boxes_per_level:
                results.append(
                    {
                        "boxes": images.new_zeros(0, 5),
                        "scores": images.new_zeros(0),
                        "labels": torch.zeros(0, dtype=torch.long, device=images.device),
                    }
                )
                continue

            # NMS 依赖精确旋转 IoU，统一用 fp32
            boxes = torch.cat(boxes_per_level).float()
            scores = torch.cat(scores_per_level).float()
            labels = torch.cat(labels_per_level)

            # 丢掉中心落在有效图像区域外的框（padding 区域）
            valid = (
                (boxes[:, 0] >= 0) & (boxes[:, 0] <= width)
                & (boxes[:, 1] >= 0) & (boxes[:, 1] <= height)
                & (boxes[:, 2] > 1.0) & (boxes[:, 3] > 1.0)
            )
            boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
            if boxes.shape[0] == 0:
                results.append(
                    {
                        "boxes": images.new_zeros(0, 5),
                        "scores": images.new_zeros(0),
                        "labels": torch.zeros(0, dtype=torch.long, device=images.device),
                    }
                )
                continue

            keep = batched_nms_rotated(
                boxes, scores, labels, iou_threshold=cfg.nms_iou_thr, max_num=cfg.max_per_img
            )
            results.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
        return results
