"""AMOD 旋转目标检测 v8 —— 频域线索 + 双向对齐 + 双向金字塔（自包含）。

v8 = v6 + v7 + **双向金字塔细化**（四个变体里最完整的组合）：
    1. 主干内部仍然没有跨模态交互，融合在各 stage 的原生尺度上进行；
    2. 融合单元同时使用局部低/高频差异线索（进门控 + 叠加到融合特征）与差异引导的
       EO<->IR 双向几何对齐（位移场 tanh 有界、带平滑正则）；
    3. Neck 在自顶向下传播之后再做一次自底向上的定位反馈（P16 / P32 接收浅层反馈），
       防止深层语义把极小目标冲淡；合并权重用快速归一化的非负加权（BiFPN 式）；
    4. 共享底座：检测头角度通道 L2 归一化 + 环形角度损失；
    5. 新增创新模块 CMCA（跨模态对比一致性对齐）：前 4 条解决的是"两模态信息怎么混合"，
       CMCA 补上"两模态是否落在同一语义流形上"——把两模态 stage 特征投到共享嵌入空间，
       在同一正样本位置做对称 InfoNCE 对齐，显式引入模态不变性约束。

本文件完全自包含：几何算子、标签分配、LoRA、质量描述子、融合单元、对比对齐模块、Neck、
检测头、损失与总装全部在本文件内，只依赖 PyTorch 与 third_party/dinov3，不引用其他 model_vN。
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

from dinov3.hub.backbones import dinov3_convnext_large  # noqa: E402


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
# 3. 微调与融合构件：LoRA + 模态质量描述子 + 细节金字塔 + 融合单元
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


class ModalLoRALinear(nn.Module):
    """模态解耦的低秩适配：共享旁路 + 每个模态各自的专属旁路。

        h = W0·x + s_shared · B_s A_s x + s_modal · B_m A_m x

    m 由 ``modality`` 属性决定（0=EO，1=IR；None 时只用共享旁路）。

    冻结主干被两个模态共享，但可见光与红外的成像特性差异很大，用同一组低秩增量去
    适配两者是次优的：共享旁路负责学模态共有的适配，专属旁路负责各自的成像特异性，
    从而在不改动主干权重的前提下让同一套主干适配两种模态。
    """

    def __init__(self, original_linear, rank=16, alpha=32, modal_rank=8, modal_alpha=16,
                 num_modalities=2, dropout=0.1):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank 必须大于 0")
        if modal_rank <= 0:
            raise ValueError("模态专属 rank 必须大于 0，需要标准 LoRA 请用 LoRALinear")

        self.original_linear = original_linear
        for param in self.original_linear.parameters():
            param.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.in_features = in_features
        self.out_features = out_features

        self.rank, self.alpha = rank, alpha
        self.scaling = float(alpha) / float(rank)
        self.modal_rank, self.modal_alpha = modal_rank, modal_alpha
        self.modal_scaling = float(modal_alpha) / float(modal_rank)
        self.num_modalities = int(num_modalities)

        # 共享旁路：所有模态都用
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        # 模态专属旁路：按 modality 取用对应的一组
        self.modal_A = nn.Parameter(torch.empty(num_modalities, modal_rank, in_features))
        self.modal_B = nn.Parameter(torch.empty(num_modalities, out_features, modal_rank))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # 当前模态，由主干在逐层前向时下发；普通属性，不进 state_dict
        self.modality = None
        self.reset_parameters()

    def reset_parameters(self):
        # B 零初始化 => 训练初始旁路输出为 0，不破坏预训练主干
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)
        nn.init.kaiming_uniform_(self.modal_A, a=5 ** 0.5)
        nn.init.zeros_(self.modal_B)

    def forward(self, x):
        dropped = self.lora_dropout(x)
        output = self.original_linear(x)
        output = output + F.linear(F.linear(dropped, self.lora_A), self.lora_B) * self.scaling
        if self.modality is not None:
            modal_a = self.modal_A[self.modality]
            modal_b = self.modal_B[self.modality]
            output = output + F.linear(F.linear(dropped, modal_a), modal_b) * self.modal_scaling
        return output


def inject_lora(module, rank=16, alpha=32, dropout=0.1, suffixes=("qkv", "proj", "fc1", "fc2"),
                modal_rank=0, modal_alpha=16, num_modalities=2):
    """把模块中的指定 Linear 子层替换为 LoRA 旁路，返回替换的层数。

    modal_rank > 0 时注入模态解耦版本 ModalLoRALinear，否则注入标准 LoRALinear。
    """
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(name.endswith(s) for s in suffixes):
            if modal_rank > 0:
                replacement = ModalLoRALinear(
                    child, rank=rank, alpha=alpha, modal_rank=modal_rank,
                    modal_alpha=modal_alpha, num_modalities=num_modalities, dropout=dropout,
                )
            else:
                replacement = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            setattr(module, name, replacement)
            count += 1
        else:
            count += inject_lora(
                child, rank=rank, alpha=alpha, dropout=dropout, suffixes=suffixes,
                modal_rank=modal_rank, modal_alpha=modal_alpha, num_modalities=num_modalities,
            )
    return count


class ModalQualityDescriptor(nn.Module):
    """模态可靠性描述子：(B, 3, H, W) -> (B, quality_dim)。

    只算 4 个标量统计量（单图开销 < 0.5 MFLOP，可忽略）：
        亮度均值   —— 整体曝光明暗
        标准差     —— 对比度
        Laplacian 方差 —— 清晰度（模糊 / 失焦时显著变小）
        Sobel 能量 —— 边缘强度（纹理丰富度）

    输入先 ``detach()``：描述子只描述"输入长什么样"，禁止通过"猜模态"走捷径。

    注意输入是 ImageNet 归一化后的张量（减均值除标准差），像素取值可低至 -2.1，
    而亮度均值同样可以为负。统计量必须在**还原到 [0,1] 的像素**上计算，否则
    ``log1p`` 遇到 < -1 的亮度会直接产生 NaN（实测约 2.4% 的模态样本命中）。
    """

    def __init__(self, quality_dim: int = 32, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        hidden = max(16, int(quality_dim))
        self.register_buffer("pixel_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer(
            "laplacian",
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3),
        )
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_x.t().reshape(1, 1, 3, 3))

        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, int(quality_dim)),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # 反归一化回 [0,1] 再裁掉插值/填充造成的越界值；fp32 保证大图统计量精度
        rgb = image.detach().float() * self.pixel_std + self.pixel_mean
        gray = rgb.clamp(0.0, 1.0).mean(dim=1, keepdim=True)     # (B, 1, H, W)

        brightness = gray.mean(dim=(2, 3))                       # (B, 1)
        contrast = gray.std(dim=(2, 3), unbiased=False)
        lap = F.conv2d(gray, self.laplacian, padding=1)
        sharpness = lap.var(dim=(2, 3), unbiased=False)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        edges = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12).mean(dim=(2, 3))

        stats = torch.cat([brightness, contrast, sharpness, edges], dim=1)   # (B, 4)
        return self.mlp(torch.log1p(stats))


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


class LocalFrequencyCue(nn.Module):
    """参数极少的局部低/高频分解线索。

    平均池化代表低频主体，它与自身重建版本的残差代表高频轮廓。与全局 FFT 不同，
    两个线索都保留了空间位置，所以能进入逐像素的门控。
    """

    def __init__(self, channels, hidden):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(4 * channels, hidden, 1, bias=False),
            make_group_norm(hidden, 16),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            make_group_norm(hidden, 16),
            nn.GELU(),
        )

    @staticmethod
    def _split(feature):
        low_small = F.avg_pool2d(feature, kernel_size=3, stride=2, padding=1)
        low = F.interpolate(low_small, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        high = feature - low
        return low, high

    def forward(self, eo, ir):
        low_eo, high_eo = self._split(eo)
        low_ir, high_ir = self._split(ir)
        cue = torch.cat(
            [torch.abs(low_eo - low_ir), torch.abs(high_eo - high_ir),
             low_eo * low_ir, high_eo * high_ir], dim=1,
        )
        return self.reduce(cue)


class BidirectionalWarp(nn.Module):
    """预测有界的 EO<->IR 位移场，并用 grid_sample 做几何对齐。"""

    def __init__(self, hidden, max_offset=2.0):
        super().__init__()
        self.max_offset = float(max_offset)
        self.offset = nn.Sequential(
            nn.Conv2d(4 * hidden + 2, hidden, 3, padding=1, bias=False),
            make_group_norm(hidden, 16),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            make_group_norm(hidden, 16),
            nn.GELU(),
            nn.Conv2d(hidden, 4, 1),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    @staticmethod
    def _warp(feature, offset):
        batch, _, height, width = feature.shape
        dtype = feature.dtype
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=torch.float32),
            indexing="ij",
        )
        grid = torch.stack([x, y], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        # offset 以特征像素为单位，这里换算成 align_corners=True 的归一化坐标
        scale_x = 2.0 / max(width - 1, 1)
        scale_y = 2.0 / max(height - 1, 1)
        delta = torch.stack([offset[:, 0] * scale_x, offset[:, 1] * scale_y], dim=-1)
        warped = F.grid_sample(
            feature.float(), grid + delta.float(), mode="bilinear",
            padding_mode="border", align_corners=True,
        )
        return warped.to(dtype)

    def forward(self, eo_hidden, ir_hidden, common, discrepancy, reliability):
        fields = self.max_offset * torch.tanh(
            self.offset(torch.cat([eo_hidden, ir_hidden, common, discrepancy, reliability], dim=1))
        )
        offset_ir_to_eo, offset_eo_to_ir = fields[:, :2], fields[:, 2:]
        aligned_ir = self._warp(ir_hidden, offset_ir_to_eo)
        aligned_eo = self._warp(eo_hidden, offset_eo_to_ir)
        magnitude = 0.5 * (
            offset_ir_to_eo.abs().mean() + offset_eo_to_ir.abs().mean()
        )
        smoothness = torch.zeros((), device=fields.device, dtype=fields.dtype)
        if fields.shape[-1] > 1:
            smoothness = smoothness + (fields[..., 1:] - fields[..., :-1]).abs().mean()
        if fields.shape[-2] > 1:
            smoothness = smoothness + (fields[..., 1:, :] - fields[..., :-1, :]).abs().mean()
        return aligned_eo, aligned_ir, magnitude, smoothness


class DiscrepancyReliabilityFusion(nn.Module):
    """共有 / 私有 + 局部频域 + 差异引导双向对齐 + 可靠性门控（原生尺度，v8）。

    分解出两路的共有分量 common（两模态一致的结构）与各自私有分量 private（成像
    特异性），门控权重由 common / private / 差异特征加上模态质量先验共同决定，
    softmax 归一化到两路。modality_dropout 屏蔽掉的一路在门控里被压到接近 0。
    频域线索按 0.25 的固定强度叠加到融合隐特征上。
    对齐位移以 tanh 有界（max_offset，浅层减半），残差由对齐后的特征承担。
    """

    def __init__(self, channels, quality_dim, hidden=128, max_offset=2.0):
        super().__init__()
        hidden = min(max(32, int(hidden)), int(channels))
        self.hidden = hidden

        self.eo_reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), make_group_norm(hidden, 16)
        )
        self.ir_reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), make_group_norm(hidden, 16)
        )

        self.common = nn.Sequential(
            nn.Conv2d(4 * hidden, hidden, 1, bias=False),
            make_group_norm(hidden, 16), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            make_group_norm(hidden, 16), nn.GELU(),
        )
        self.private_eo = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 1, bias=False), make_group_norm(hidden, 16), nn.GELU()
        )
        self.private_ir = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 1, bias=False), make_group_norm(hidden, 16), nn.GELU()
        )

        self.frequency = LocalFrequencyCue(hidden, hidden)
        # 门控输入：common / private_eo / private_ir / 模态差异 / 频域线索
        gate_inputs = 5 * hidden
        gate_hidden = max(32, hidden // 2)
        self.reliability = nn.Sequential(
            nn.Conv2d(gate_inputs, gate_hidden, 1, bias=False),
            make_group_norm(gate_hidden, 16), nn.GELU(),
            nn.Conv2d(gate_hidden, gate_hidden, 3, padding=1, groups=gate_hidden, bias=False),
            make_group_norm(gate_hidden, 16), nn.GELU(),
            nn.Conv2d(gate_hidden, 2, 1),
        )
        # 门控末层零初始化 => 起点两路等权，训练中再学出偏向
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)
        self.quality_prior = nn.Linear(3 * quality_dim, 2)
        nn.init.zeros_(self.quality_prior.weight)
        nn.init.zeros_(self.quality_prior.bias)

        self.aligner = BidirectionalWarp(hidden, max_offset=max_offset)
        self.expand = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False), make_group_norm(channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False), make_group_norm(channels),
        )
        self.refine = SeparableRefine(channels)
        self.residual_gate = nn.Parameter(torch.tensor(-0.5))

    def forward(self, eo, ir, quality_eo, quality_ir, drop_flag=None):
        eo_hidden = F.gelu(self.eo_reduce(eo))
        ir_hidden = F.gelu(self.ir_reduce(ir))
        product = eo_hidden * ir_hidden
        difference = torch.abs(eo_hidden - ir_hidden)
        common = self.common(torch.cat([eo_hidden, ir_hidden, product, difference], dim=1))
        private_eo = self.private_eo(torch.cat([eo_hidden - common, difference], dim=1))
        private_ir = self.private_ir(torch.cat([ir_hidden - common, difference], dim=1))

        gate_features = [common, private_eo, private_ir, difference]
        frequency = self.frequency(eo_hidden, ir_hidden)
        gate_features.append(frequency)
        logits = self.reliability(torch.cat(gate_features, dim=1))
        quality_pair = torch.cat([quality_eo, quality_ir, quality_eo - quality_ir], dim=1)
        logits = logits + self.quality_prior(quality_pair).unsqueeze(-1).unsqueeze(-1)

        if drop_flag is not None:
            # 被屏蔽的模态压到接近 0：用减法而不是 log(0)，AMP 下不会出现 inf
            logits = logits - 20.0 * drop_flag.to(logits.dtype).unsqueeze(-1).unsqueeze(-1)
        reliability = F.softmax(logits, dim=1)

        aligned_eo, aligned_ir, offset_magnitude, offset_smoothness = self.aligner(
            eo_hidden, ir_hidden, common, difference, reliability
        )
        weight_eo, weight_ir = reliability[:, :1], reliability[:, 1:]
        # 每个方向都保留自己的私有线索（开启对齐时是在几何校正之后）
        eo_candidate = aligned_eo + private_eo
        ir_candidate = aligned_ir + private_ir
        fused_hidden = common + weight_eo * eo_candidate + weight_ir * ir_candidate
        fused_hidden = fused_hidden + 0.25 * frequency

        residual = self.expand(fused_hidden)
        shortcut = self.shortcut(torch.cat([eo, ir], dim=1))
        fused = self.refine(shortcut + torch.sigmoid(self.residual_gate) * residual)

        # 只用于防止共有效率 / 私有率退化的保守项，默认权重很小
        consistency_per_sample = 1.0 - F.cosine_similarity(
            F.adaptive_avg_pool2d(aligned_eo, 1).flatten(1),
            F.adaptive_avg_pool2d(aligned_ir, 1).flatten(1), dim=1,
        )
        if drop_flag is not None:
            both_valid = drop_flag.sum(dim=1) == 0
            if both_valid.any():
                consistency = consistency_per_sample[both_valid].mean()
            else:
                consistency = (aligned_eo.sum() + aligned_ir.sum()) * 0.0
        else:
            consistency = consistency_per_sample.mean()
        stats = {
            "gate_mean": weight_eo.mean(dim=(1, 2, 3)),
            "offset_magnitude": offset_magnitude,
            "offset_smoothness": offset_smoothness,
            "fusion_consistency": consistency,
        }
        return fused, stats


class CrossModalContrastiveAlign(nn.Module):
    """CMCA：跨模态对比一致性对齐（v8 新增的创新模块）。

    DiscrepancyReliabilityFusion 是"加法 / 门控"式的信息混合：它让两模态互相补充，
    但从不显式约束两模态在**同一空间位置**上编码**同一目标语义**。RGB 与 T 看的是同一
    场景，同一辆坦克在两种成像下外观差异极大，语义上却应当一致。

    本模块把两模态在各 stage 的原生特征投到**同一个共享嵌入空间**（投影头 EO / IR 共用
    同一套权重——若各投各的，"同一空间"就无从谈起），再在共享空间里做跨模态对比：同一
    位置的正样本对（EO 位置 k ↔ IR 位置 k）被拉近，不同位置互为负样本被推远（对称
    InfoNCE，温度的 log 尺度可学习）。

    与融合单元的分工：融合单元解决"信息怎么混合"，本模块解决"两模态是否落在同一语义
    流形上"，是语义级的显式约束，与融合互补而非重复。

    只挂在能与检测层在空间上严格对齐的 stage 上：stride 8/16/32 对应检测层 P8/P16/P32
    （Neck 这三层正是对应 stage 特征经 1x1 侧向投影得到，尺寸一致）；stride 64 的 P64 由
    P32 下采样得到，主干侧没有对应 stage，故不参与。本模块只输出嵌入，对齐损失在
    AMODLossV8 里用标签分配出的正样本位置计算。
    """

    def __init__(self, stage_dims, aligned_stages, aligned_levels, embed_dim=64, temperature=0.07):
        super().__init__()
        self.aligned_stages = tuple(int(index) for index in aligned_stages)
        self.aligned_levels = tuple(int(level) for level in aligned_levels)
        self.embed_dim = int(embed_dim)
        # 每个 stage 一个投影头（通道数不同），但两模态**共享同一套权重**
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(int(stage_dims[index]), self.embed_dim, 1, bias=False),
                make_group_norm(self.embed_dim),
            )
            for index in self.aligned_stages
        ])
        # 可学习温度（存 log 尺度，CLIP 惯例）：训练中自适应调节对比的尖锐度
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / float(temperature))))

    def forward(self, eo_features, ir_features):
        """返回 aux['cmca']：两模态在各对齐 stage 的 L2 归一化嵌入 + 可学习温度。"""
        eo_embeddings, ir_embeddings, similarities = [], [], []
        for slot, index in enumerate(self.aligned_stages):
            eo = self.proj[slot](eo_features[index])
            ir = self.proj[slot](ir_features[index])
            # 归一化用 fp32：bf16 下范数的量化误差会污染余弦相似度
            eo_f = F.normalize(eo.float(), dim=1)
            ir_f = F.normalize(ir.float(), dim=1)
            eo_embeddings.append(eo_f.to(eo.dtype))
            ir_embeddings.append(ir_f.to(ir.dtype))
            similarities.append((eo_f * ir_f).sum(dim=1).mean().detach())
        return {
            "eo": eo_embeddings,
            "ir": ir_embeddings,
            "levels": self.aligned_levels,
            "logit_scale": self.logit_scale,
            "similarity": torch.stack(similarities).mean(),
        }


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
    自顶向下 + 自底向上双向（v8）：P16 / P32 在自顶向下之后再接收来自浅层的
    定位反馈，防止深层语义把小目标冲淡。
    """

    def __init__(self, cfg, in_channels_list):
        super().__init__()
        if list(map(int, cfg.fpn_strides)) != [8, 16, 32, 64]:
            raise ValueError("v5-v8 要求 fpn_strides=[8, 16, 32, 64]")
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
# 6. 主干：DINOv3 ConvNeXt-L 冻结权重 + 模态解耦 LoRA + 原生尺度融合
# ===========================================================================

_LORA_SUFFIXES = ("pwconv1", "pwconv2")


class _ModalConvBlock(nn.Module):
    """ConvNeXt Block 的包装：下发所属模态，并按需启用梯度检查点。

    模态解耦 LoRA 靠 ``modality`` 属性切换专属旁路，而梯度检查点在反向时会重算
    前向 —— 若重算时读到已被后续层覆盖的模态标记，梯度就错了。所以这里把 modality
    作为闭包变量捕获，保证重算与原始前向用的是同一个模态（与 ViT 版的 _ModalBlock 同理）。
    """

    def __init__(self, block, use_checkpoint=False):
        super().__init__()
        self.block = block
        self.use_checkpoint = bool(use_checkpoint)
        self.modal_layers = [m for m in block.modules() if isinstance(m, ModalLoRALinear)]

    def _set_modality(self, modality):
        for layer in self.modal_layers:
            layer.modality = modality

    def forward(self, x, modality=None):
        if not self.use_checkpoint:
            if self.modal_layers and modality is not None:
                self._set_modality(modality)
            return self.block(x)

        def run(tensor):
            if self.modal_layers and modality is not None:
                self._set_modality(modality)
            return self.block(tensor)

        return torch.utils.checkpoint.checkpoint(run, x, use_reentrant=False)


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


class NativeScaleFusionBackbone(nn.Module):
    """DINOv3 ConvNeXt-L 主干（冻结权重 + 模态解耦 LoRA）+ 原生尺度逐 stage 融合。

    两路模态各自完整跑完主干（共享同一套冻结参数，靠 modality 标记切换 LoRA 旁路），
    融合只发生在每个 stage 输出的**原生尺度**特征上，因此没有 v2/v3 的 token 级
    交叉注意力，也没有额外的尺度重采样。    浅层（stage0）的位移上限减半：低层特征分辨率高、配准误差本就更小。

    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.convnext = dinov3_convnext_large(
            pretrained=bool(cfg.backbone_pretrained),
            weights=cfg.backbone_weights,
        )
        self.stage_dims = [int(dim) for dim in self.convnext.embed_dims]
        self.num_stages = len(self.stage_dims)
        # Neck 按这个通道数逐层建侧向投影，因此允许每层通道数不同
        self.out_channels = list(self.stage_dims)

        if cfg.modality.lower() != "eoir":
            raise ValueError("NativeScaleFusionBackbone 只支持双模态（modality='eoir'）")
        self.modality_dropout = float(getattr(cfg, "modality_dropout", 0.0))

        # ---- 模态专属输入投影（从预训练 stem 复制，可训练）----
        pretrained_stem = self.convnext.downsample_layers[0]
        self.stem_eo = _build_stem(pretrained_stem, 3)
        self.stem_ir = _build_stem(pretrained_stem, 3)
        # 原始 stem 不再使用（输入侧已改为按模态分流）
        self.convnext.downsample_layers[0] = nn.Identity()

        # 模态嵌入：让共享 stages 能够区分当前处理的是哪一路
        self.modality_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(2, dim, 1, 1)) for dim in self.stage_dims]
        )

        # ---- 冻结主干，只训练输入投影与 LoRA 旁路 ----
        for param in self.convnext.parameters():
            param.requires_grad = False
        trainable_input = bool(getattr(cfg, "trainable_patch_embed", True))
        for stem in (self.stem_eo, self.stem_ir):
            for param in stem.parameters():
                param.requires_grad = trainable_input

        # ---- 模态解耦 LoRA（共享旁路 + EO/IR 各自专属旁路）----
        self.num_lora_layers = 0
        self.lora_modal_rank = int(getattr(cfg, "lora_modal_rank", 0))
        if cfg.use_lora:
            self.num_lora_layers = inject_lora(
                self.convnext,
                rank=int(cfg.lora_rank),
                alpha=int(cfg.lora_alpha),
                dropout=float(cfg.lora_dropout),
                suffixes=_LORA_SUFFIXES,
                modal_rank=self.lora_modal_rank,
                modal_alpha=int(getattr(cfg, "lora_modal_alpha", 16)),
                num_modalities=2,
            )

        # 统一包装：既支持按需梯度检查点，也负责逐层下发当前模态
        for stage in self.convnext.stages:
            for index, block in enumerate(stage):
                stage[index] = _ModalConvBlock(block, use_checkpoint=bool(cfg.grad_checkpoint))

        # ---- 模态质量描述子（只在 stage0 之前算一次，跨 stage 共享）----
        quality_dim = int(getattr(cfg, "quality_dim", 32))
        # 描述子需要 cfg 的归一化参数把像素还原到 [0,1]，否则 log1p 可能吃到负亮度
        norm_kwargs = dict(mean=tuple(cfg.mean), std=tuple(cfg.std))
        self.quality_eo = ModalQualityDescriptor(quality_dim, **norm_kwargs)
        self.quality_ir = ModalQualityDescriptor(quality_dim, **norm_kwargs)

        # ---- 逐 stage 的原生尺度融合 ----
        hidden = int(getattr(cfg, "fusion_hidden", 128))
        max_offset = float(getattr(cfg, "alignment_max_offset", 2.0))
        self.fusions = nn.ModuleList([
            DiscrepancyReliabilityFusion(
                dim, quality_dim, hidden=hidden,
                max_offset=max_offset * (0.5 if index == 0 else 1.0),
            )
            for index, dim in enumerate(self.stage_dims)
        ])

        # ---- 跨模态对比一致性对齐（CMCA，v8 创新模块）----
        # 主干 stage 的步长是 4 * 2^i；只有步长与某个检测层相等时，两者特征图尺寸才完全
        # 一致（Neck 的 P8/P16/P32 就是对应 stage 的 1x1 侧向投影），对齐损失才能直接按
        # 检测层的正样本掩码去取嵌入。
        det_strides = [int(stride) for stride in cfg.fpn_strides]
        aligned_stages, aligned_levels = [], []
        for index in range(self.num_stages):
            stage_stride = 4 * (2 ** index)
            if stage_stride in det_strides:
                aligned_stages.append(index)
                aligned_levels.append(det_strides.index(stage_stride))
        self.cmca_used = bool(getattr(cfg, "use_cmca", True)) and bool(aligned_stages)
        self.cmca = (
            CrossModalContrastiveAlign(
                self.stage_dims, aligned_stages, aligned_levels,
                embed_dim=int(getattr(cfg, "cmca_embed_dim", 64)),
                temperature=float(getattr(cfg, "cmca_temperature", 0.07)),
            )
            if self.cmca_used else None
        )

    def trainable_parameters(self):
        """返回需要优化的参数（配合 LoRA，可单独设置学习率）。"""
        return [p for p in self.parameters() if p.requires_grad]


    def _run_stage(self, stage, feature, modality):
        """按顺序跑完一个 stage 的 ConvNeXt blocks。"""
        for block in stage:
            feature = block(feature, modality)
        return feature


    def forward(self, images: torch.Tensor):
        """返回 (list[(B, C_i, H_i, W_i)], aux)。"""
        batch = images.shape[0]
        device = images.device
        eo_image, ir_image = images[:, :3], images[:, 3:6]

        # 与 v2/v3 完全相同的随机逻辑（RNG 消耗也一致），额外保留屏蔽标志，
        # 供门控平衡正则 / 门控自监督使用
        drop_eo = torch.zeros(batch, device=device)
        drop_ir = torch.zeros(batch, device=device)
        if self.training and self.modality_dropout > 0:
            drop_ir = (torch.rand(batch, device=device) < self.modality_dropout).float()
            drop_eo = (
                (torch.rand(batch, device=device) < self.modality_dropout) & (drop_ir == 0)
            ).float()
            eo_image = eo_image.masked_fill(drop_eo.view(batch, 1, 1, 1) > 0, 0.0)
            ir_image = ir_image.masked_fill(drop_ir.view(batch, 1, 1, 1) > 0, 0.0)

        # 模态质量只在 stem 之前算一次，跨 stage 共享
        quality_eo = self.quality_eo(eo_image)
        quality_ir = self.quality_ir(ir_image)

        eo_feature = self.stem_eo(eo_image)
        ir_feature = self.stem_ir(ir_image)
        drop_flag = torch.stack([drop_eo, drop_ir], dim=1)

        features, gate_probs, consistencies = [], [], []
        offset_magnitudes, offset_smoothness, consistencies = [], [], []
        eo_stage_features, ir_stage_features = [], []
        for index in range(self.num_stages):
            if index > 0:
                eo_feature = self.convnext.downsample_layers[index](eo_feature)
                ir_feature = self.convnext.downsample_layers[index](ir_feature)
            eo_feature = eo_feature + self.modality_bias[index][0]
            ir_feature = ir_feature + self.modality_bias[index][1]
            eo_feature = self._run_stage(self.convnext.stages[index], eo_feature, 0)
            ir_feature = self._run_stage(self.convnext.stages[index], ir_feature, 1)
            # 留一份融合前的单模态原生特征，供 CMCA 做跨模态对比对齐
            eo_stage_features.append(eo_feature)
            ir_stage_features.append(ir_feature)

            # 屏蔽掉的一路在融合前严格置零，门控目标（0 / 1）才严格可达
            eo_valid = eo_feature * (1.0 - drop_eo).view(batch, 1, 1, 1)
            ir_valid = ir_feature * (1.0 - drop_ir).view(batch, 1, 1, 1)
            fused, stats = self.fusions[index](
                eo_valid, ir_valid, quality_eo, quality_ir, drop_flag=drop_flag
            )
            features.append(fused)
            gate_probs.append(stats["gate_mean"])
            offset_magnitudes.append(stats["offset_magnitude"])
            offset_smoothness.append(stats["offset_smoothness"])
            consistencies.append(stats["fusion_consistency"])

        gate_prob = torch.stack(gate_probs, dim=1)                   # (B, num_stages)
        gate_mean = gate_prob.mean(dim=1)
        both_valid = drop_flag.sum(dim=1) == 0
        if both_valid.any():
            gate_balance = ((gate_mean[both_valid] - 0.5) ** 2).mean()
        else:
            gate_balance = gate_prob.sum() * 0.0
        aux = {
            "gate_prob": gate_prob,
            "gate_mean": gate_mean.detach(),
            "gate_balance": gate_balance,
            "mod_drop": drop_flag,
            "offset_magnitude": torch.stack(offset_magnitudes).mean(),
            "offset_smoothness": torch.stack(offset_smoothness).mean(),
            "fusion_consistency": torch.stack(consistencies).mean(),
        }
        # CMCA 只输出嵌入，跨模态对比损失在 AMODLossV8 里按正样本位置计算
        if self.cmca is not None:
            aux["cmca"] = self.cmca(eo_stage_features, ir_stage_features)
        return features, aux


# ===========================================================================
# 7. 损失：基础损失 + 环形角度损失 + 门控自监督 + 融合正则
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
        # 跨模态门控平衡正则（v2）：惩罚 batch 内平均门控偏离 0.5，防止塌缩到单一模态
        self.w_gate = float(getattr(cfg, "loss_gate_balance_weight", 0.0))

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

        # ---- 跨模态门控平衡正则（不参与 num_pos 归一化）----
        aux = preds.get("aux") if isinstance(preds.get("aux"), dict) else None
        if self.w_gate > 0 and aux and "gate_balance" in aux:
            gate_loss = self.w_gate * aux["gate_balance"]
            total = total + gate_loss
            parts["loss"] = float(total.detach())
            parts["loss_gate"] = float(gate_loss.detach())
            if "gate_mean" in aux:
                parts["gate_mean"] = float(aux["gate_mean"].mean().detach())
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


class AMODLossV3(AMODLossV4):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.w_gate_sup = float(getattr(cfg, "loss_gate_sup_weight", 0.0))

    def forward(self, preds, gt_boxes_list, gt_labels_list):
        total, parts = super().forward(preds, gt_boxes_list, gt_labels_list)

        aux = preds.get("aux") if isinstance(preds.get("aux"), dict) else None
        if aux is None or "gate_prob" not in aux or "mod_drop" not in aux:
            return total, parts

        if "dir_mean" in aux:
            parts["dir_mean"] = float(aux["dir_mean"])
        if self.w_gate_sup <= 0:
            return total, parts

        gate_prob = aux["gate_prob"]                                  # (B, K) 各 stage 的 EO 权重
        drop = aux["mod_drop"].to(gate_prob.dtype)                    # (B, 2)
        drop_eo, drop_ir = drop[:, 0], drop[:, 1]
        count = float((drop_eo + drop_ir).sum())
        if count <= 0:                                                # 该 batch 没屏蔽任何模态
            parts["loss_gate_sup"] = 0.0
            return total, parts

        cost = (drop_eo[:, None] * gate_prob.pow(2)).sum() + (
            drop_ir[:, None] * (gate_prob - 1.0).pow(2)
        ).sum()
        gate_sup = self.w_gate_sup * cost / max(count, 1.0)

        total = total + gate_sup
        parts["loss"] = float(total.detach())
        parts["loss_gate_sup"] = float(gate_sup.detach())
        return total, parts


class AMODLossV8(AMODLossV3):
    """检测损失 + 保守的融合一致性正则（本变体的融合单元专用）。
    对齐位移场额外加一个平滑正则（默认权重 0.005），避免学出高频抖动。
    新增 CMCA 跨模态对比一致性损失：在共享嵌入空间把同一位置的 EO/IR 特征对齐。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.w_consistency = float(getattr(cfg, "loss_fusion_consistency_weight", 0.01))
        self.w_smooth = float(getattr(cfg, "loss_offset_smooth_weight", 0.005))
        self.w_cmca = float(getattr(cfg, "loss_cmca_weight", 0.05))
        # 可学习温度的上限（CLIP 惯例）：防止训练后期温度过小把损失压成 0
        self.cmca_max_logit_scale = math.log(100.0)

    def _cross_modal_contrastive(self, preds, aux, gt_boxes_list, gt_labels_list):
        """CMCA 对齐损失：同一空间位置的 EO/IR 嵌入互为正样本，不同位置互为负样本。

        用标签分配出的正样本位置做锚点，因此只在这些"确实有目标"的位置上施加模态
        不变性约束，背景区域不参与（避免把大片背景噪声也强行对齐）。
        返回 (loss, 参与对齐的层-图组数)；无可用正样本时返回 (None, 0)。
        """
        cmca = aux.get("cmca")
        if not isinstance(cmca, dict):
            return None, 0

        eo_embeddings, ir_embeddings = cmca["eo"], cmca["ir"]
        levels = list(cmca["levels"])
        device = eo_embeddings[0].device
        scale = cmca["logit_scale"].clamp(max=self.cmca_max_logit_scale).exp()
        mod_drop = aux.get("mod_drop")

        losses = []
        for batch_index, (gt_boxes, gt_labels) in enumerate(zip(gt_boxes_list, gt_labels_list)):
            # 被屏蔽的样本里有一路特征被人为置零，对齐没有意义，直接跳过
            if mod_drop is not None and float(mod_drop[batch_index].sum()) > 0:
                continue
            gt_boxes = gt_boxes.to(device)
            gt_labels = gt_labels.to(device)
            valid = gt_boxes[:, 2] > 1e-3
            gt_boxes, gt_labels = gt_boxes[valid], gt_labels[valid]
            if gt_boxes.numel() == 0:
                continue

            # levels 对应的检测层与主干 stage 步长相同、特征图尺寸一致，正样本掩码可直接
            # 作用在 CMCA 嵌入上
            targets = build_targets(
                [preds["locations"][level] for level in levels],
                [preds["strides"][level] for level in levels],
                gt_boxes, gt_labels,
                [self.scale_ranges[level] for level in levels],
                self.center_radius,
            )
            for slot, target in enumerate(targets):
                mask = target["pos_mask"]
                # 少于两个正样本时没有可区分的负样本，InfoNCE 退化，跳过
                if int(mask.sum()) < 2:
                    continue
                eo_sample = eo_embeddings[slot][batch_index]        # (D, H, W)
                ir_sample = ir_embeddings[slot][batch_index]
                embed_dim = eo_sample.shape[0]
                eo = eo_sample.permute(1, 2, 0).reshape(-1, embed_dim)[mask].float()
                ir = ir_sample.permute(1, 2, 0).reshape(-1, embed_dim)[mask].float()
                logits = scale * (eo @ ir.t())
                index = torch.arange(logits.shape[0], device=device)
                # 对称 InfoNCE：EO->IR 与 IR->EO 各算一次再取平均
                losses.append(0.5 * (
                    F.cross_entropy(logits, index) + F.cross_entropy(logits.t(), index)
                ))

        if not losses:
            return None, 0
        return torch.stack(losses).mean(), len(losses)

    def forward(self, preds, gt_boxes_list, gt_labels_list):
        total, parts = super().forward(preds, gt_boxes_list, gt_labels_list)
        aux = preds.get("aux") if isinstance(preds.get("aux"), dict) else None
        if not aux:
            return total, parts
        if self.w_consistency > 0 and "fusion_consistency" in aux:
            value = self.w_consistency * aux["fusion_consistency"]
            total = total + value
            parts["loss_fusion_consistency"] = float(value.detach())
        if self.w_smooth > 0 and "offset_smoothness" in aux:
            value = self.w_smooth * aux["offset_smoothness"]
            total = total + value
            parts["loss_offset_smooth"] = float(value.detach())
        if "offset_magnitude" in aux:
            parts["offset_magnitude"] = float(aux["offset_magnitude"].detach())

        # ---- CMCA 跨模态对比一致性对齐 ----
        if self.w_cmca > 0 and isinstance(aux.get("cmca"), dict):
            cmca_value, num_groups = self._cross_modal_contrastive(
                preds, aux, gt_boxes_list, gt_labels_list
            )
            if cmca_value is not None:
                cmca_loss = self.w_cmca * cmca_value
                total = total + cmca_loss
                parts["loss_cmca"] = float(cmca_loss.detach())
                parts["cmca_groups"] = float(num_groups)
            else:
                parts["loss_cmca"] = 0.0
            if "similarity" in aux["cmca"]:
                parts["cmca_similarity"] = float(aux["cmca"]["similarity"].detach())

        parts["loss"] = float(total.detach())
        return total, parts


# ===========================================================================
# 8. 总装：backbone + neck + head + criterion + 解码与旋转 NMS
# ===========================================================================

class AMODDetector(nn.Module):
    """AMOD v8 检测器：频域 + 双向对齐融合主干 + 双向金字塔 Neck + 归一化角度头。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(cfg.num_classes)
        self.strides = [int(s) for s in cfg.fpn_strides]
        self.num_levels = len(self.strides)

        self.backbone = NativeScaleFusionBackbone(cfg)
        self.neck = NativePyramidNeck(cfg, self.backbone.out_channels)
        self.head = NormOBBHead(cfg, int(cfg.feature_dim), self.num_levels)
        self.criterion = AMODLossV8(cfg)

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
        features, aux = self.backbone(images)
        feats, layer_weights = self.neck(features, images)
        return feats, layer_weights, aux


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
