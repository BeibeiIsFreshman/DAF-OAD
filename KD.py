"""
知识蒸馏损失（KD Loss）：融合蒸馏 + 预测蒸馏。

教师：model_v8（双模态 EO+IR，跨模态原生尺度融合）
学生：model_student_v1（单模态 RGB）
教师全程冻结，只把它的「融合表征」与「检测预测」当作软监督信号传给学生。

两个创新分项：
    FAFD（Fusion-Attention Feature Distillation，融合蒸馏）：
        教师看过 EO+IR 两路并做过跨模态融合，其 Neck 特征里含有"融合后的可靠目标
        表征"。本项用前景加权的逐位置余弦对齐 + 空间/通道双注意力对齐，把这份融合
        知识迁移给只看 RGB 的学生。
    QAPD（Quality-Angle-aware Prediction Distillation，预测蒸馏）：
        把教师检测头的软输出当软目标。分类用教师质量图门控，回归按旋转检测的特性
        把 6 维输出拆成"距离(ltrb)"与"角度(sin2θ, cos2θ)"分别蒸馏——距离用 stride
        归一化尺度做 smooth-L1（跨层可比），角度用环形损失（处理周期性）。

本文件只依赖 PyTorch，不引用任何 model_vN，方便单独消融与复用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FusionKDLoss", "PredictionKDLoss", "KDLossV1", "build_kd_loss"]


def teacher_quality_maps(teacher_preds):
    """教师逐层质量图 q = max_c sigmoid(cls) * sigmoid(ctr)，detach 后作为门控。

    q 越大说明教师在该位置越确信"这里有一个目标"。融合 / 预测蒸馏都用它做前景加权，
    避免把教师在背景上的噪声或不确定性硬灌给学生。
    """
    maps = []
    for cls, ctr in zip(teacher_preds["cls"], teacher_preds["ctr"]):
        prob = cls.detach().float().sigmoid().max(dim=1, keepdim=True).values
        center = ctr.detach().float().sigmoid()
        maps.append(prob * center)
    return maps


class FusionKDLoss(nn.Module):
    """融合蒸馏 FAFD：前景加权融合特征蒸馏 + 空间 / 通道注意力对齐。

    学生只有 RGB 一路，教师看过 EO+IR 两路并做了跨模态融合，因此教师的 Neck 特征里
    含有"跨模态融合后的可靠目标表征"。本模块用三路约束把它迁移给学生：

      1) 逐位置余弦对齐（前景加权）：把教师融合特征当软目标，学生特征经逐层 1x1 适配器
         投到同一空间后做 1-cos 对齐；权重由教师质量图给出，背景权重压得很低，让学生
         把有限的容量集中到目标区域；
      2) 空间注意力对齐：对齐 |F| 沿通道平均归一化后的"看哪里"分布；
      3) 通道注意力对齐：对齐 |F| 沿空间平均归一化后的"用哪些通道"分布。
    """

    def __init__(
        self,
        student_dims,
        teacher_dims,
        feat_weight=1.0,
        spatial_weight=0.5,
        channel_weight=0.5,
        background_weight=0.1,
        eps=1e-6,
    ):
        super().__init__()
        if len(student_dims) != len(teacher_dims):
            raise ValueError("学生 / 教师特征层数必须一致")
        self.feat_weight = float(feat_weight)
        self.spatial_weight = float(spatial_weight)
        self.channel_weight = float(channel_weight)
        self.background_weight = float(background_weight)
        self.eps = float(eps)

        # 学生 -> 教师的逐层 1x1 适配器（可学习，挂到学生模型上随优化器一起训练）
        self.adapters = nn.ModuleList(
            nn.Conv2d(int(cs), int(ct), kernel_size=1, bias=False)
            for cs, ct in zip(student_dims, teacher_dims)
        )
        self.num_levels = len(self.adapters)

    def forward(self, student_feats, teacher_feats, quality_maps):
        feat_loss = student_feats[0].new_zeros((), dtype=torch.float32)
        spatial_loss = student_feats[0].new_zeros((), dtype=torch.float32)
        channel_loss = student_feats[0].new_zeros((), dtype=torch.float32)

        for level in range(self.num_levels):
            s = self.adapters[level](student_feats[level].float())
            t = teacher_feats[level].detach().float()
            q = quality_maps[level].to(s.dtype)
            batch, channels, height, width = s.shape
            num_loc = height * width

            s_flat = s.reshape(batch, channels, num_loc)
            t_flat = t.reshape(batch, channels, num_loc)

            # ---- 1) 前景加权余弦对齐 ----
            s_norm = F.normalize(s_flat, dim=1, eps=self.eps)
            t_norm = F.normalize(t_flat, dim=1, eps=self.eps)
            cosine = (s_norm * t_norm).sum(dim=1)                      # (B,N)
            weight = self.background_weight + (1.0 - self.background_weight) * q.reshape(batch, num_loc)
            feat_loss = feat_loss + ((1.0 - cosine) * weight).sum() / weight.sum().clamp(min=self.eps)

            # ---- 2) 空间注意力对齐 ----
            s_spatial = F.normalize(s_flat.abs().mean(dim=1), dim=1, eps=self.eps)
            t_spatial = F.normalize(t_flat.abs().mean(dim=1), dim=1, eps=self.eps)
            spatial_loss = spatial_loss + F.mse_loss(s_spatial, t_spatial)

            # ---- 3) 通道注意力对齐 ----
            s_channel = F.normalize(s.abs().mean(dim=(2, 3)), dim=1, eps=self.eps)
            t_channel = F.normalize(t.abs().mean(dim=(2, 3)), dim=1, eps=self.eps)
            channel_loss = channel_loss + F.mse_loss(s_channel, t_channel)

        scale = 1.0 / max(self.num_levels, 1)
        feat_loss = feat_loss * scale
        spatial_loss = spatial_loss * scale
        channel_loss = channel_loss * scale
        total = (
            self.feat_weight * feat_loss
            + self.spatial_weight * spatial_loss
            + self.channel_weight * channel_loss
        )
        parts = {
            "kd_fusion": float(total.detach()),
            "kd_fusion_feat": float(feat_loss.detach()),
            "kd_fusion_spatial": float(spatial_loss.detach()),
            "kd_fusion_channel": float(channel_loss.detach()),
        }
        return total, parts


class PredictionKDLoss(nn.Module):
    """预测蒸馏 QAPD：质量门控软分类 + 距离 / 角度解耦回归蒸馏。

    教师检测头输出（sigmoid 后的类别分数、距离、角度、centerness）作为软目标：

      1) 分类：用教师逐类软分数做二值交叉熵，权重 = 背景权重 + 教师质量图，
         只在教师有把握的位置施加主要监督；
      2) 距离 ltrb：教师 / 学生都取 softplus(reg[:, :4])（即"以 stride 为单位"的到边
         距离）做 smooth-L1——天然跨层尺度可比，不会让深层的粗尺度主导；
      3) 角度：把 (sin2θ, cos2θ) 当单位向量，用环形损失 1 - cos(2Δθ) 对齐，显式处理
         角度周期性与方向歧义；
      4) centerness：sigmoid 后按质量图加权做 smooth-L1。

    回归三项只在教师质量高的前景位置上生效，避免背景回归噪声污染。
    """

    def __init__(
        self,
        cls_weight=1.0,
        ltrb_weight=0.5,
        angle_weight=0.5,
        ctr_weight=1.0,
        background_weight=0.1,
        smooth_beta=0.1,
        eps=1e-6,
    ):
        super().__init__()
        self.cls_weight = float(cls_weight)
        self.ltrb_weight = float(ltrb_weight)
        self.angle_weight = float(angle_weight)
        self.ctr_weight = float(ctr_weight)
        self.background_weight = float(background_weight)
        self.smooth_beta = float(smooth_beta)
        self.eps = float(eps)

    def forward(self, student_preds, teacher_preds, quality_maps):
        device = student_preds["cls"][0].device
        cls_loss = torch.zeros((), device=device, dtype=torch.float32)
        ltrb_loss = torch.zeros((), device=device, dtype=torch.float32)
        angle_loss = torch.zeros((), device=device, dtype=torch.float32)
        ctr_loss = torch.zeros((), device=device, dtype=torch.float32)

        num_levels = len(student_preds["cls"])
        for level in range(num_levels):
            s_cls = student_preds["cls"][level].float()
            t_cls = teacher_preds["cls"][level].detach().float()
            s_reg = student_preds["reg"][level].float()
            t_reg = teacher_preds["reg"][level].detach().float()
            s_ctr = student_preds["ctr"][level].float()
            t_ctr = teacher_preds["ctr"][level].detach().float()

            q = quality_maps[level].to(s_cls.dtype)                    # (B,1,H,W)
            batch, num_classes, height, width = s_cls.shape
            num_loc = height * width
            q_flat = q.reshape(batch, num_loc)

            # ---- 1) 分类软蒸馏（质量门控）----
            weight_cls = self.background_weight + (1.0 - self.background_weight) * q
            bce = F.binary_cross_entropy_with_logits(
                s_cls, t_cls.sigmoid(), reduction="none"
            )
            cls_loss = cls_loss + (bce * weight_cls).sum() / (
                weight_cls.sum().clamp(min=self.eps) * num_classes
            )

            # ---- 2) 距离 ltrb 蒸馏（stride 归一化尺度，跨层可比）----
            denominator = q_flat.sum().clamp(min=self.eps)
            s_dist = F.softplus(s_reg[:, :4]).reshape(batch, 4, num_loc)
            t_dist = F.softplus(t_reg[:, :4]).reshape(batch, 4, num_loc)
            dist_diff = F.smooth_l1_loss(
                s_dist, t_dist, reduction="none", beta=self.smooth_beta
            ).sum(dim=1)                                               # (B,N)
            ltrb_loss = ltrb_loss + (dist_diff * q_flat).sum() / (denominator * 4.0)

            # ---- 3) 角度蒸馏（环形损失，处理周期性）----
            s_angle = F.normalize(s_reg[:, 4:6], dim=1, eps=self.eps).reshape(batch, 2, num_loc)
            t_angle = F.normalize(t_reg[:, 4:6], dim=1, eps=self.eps).reshape(batch, 2, num_loc)
            cosine = (s_angle * t_angle).sum(dim=1)                    # (B,N)
            angle_loss = angle_loss + ((1.0 - cosine) * q_flat).sum() / denominator

            # ---- 4) centerness 蒸馏 ----
            s_center = s_ctr.sigmoid().reshape(batch, num_loc)
            t_center = t_ctr.sigmoid().reshape(batch, num_loc)
            ctr_diff = F.smooth_l1_loss(
                s_center, t_center, reduction="none", beta=self.smooth_beta
            )
            ctr_weight = weight_cls.reshape(batch, num_loc)
            ctr_loss = ctr_loss + (ctr_diff * ctr_weight).sum() / ctr_weight.sum().clamp(min=self.eps)

        scale = 1.0 / max(num_levels, 1)
        cls_loss = cls_loss * scale
        ltrb_loss = ltrb_loss * scale
        angle_loss = angle_loss * scale
        ctr_loss = ctr_loss * scale
        total = (
            self.cls_weight * cls_loss
            + self.ltrb_weight * ltrb_loss
            + self.angle_weight * angle_loss
            + self.ctr_weight * ctr_loss
        )
        parts = {
            "kd_prediction": float(total.detach()),
            "kd_pred_cls": float(cls_loss.detach()),
            "kd_pred_ltrb": float(ltrb_loss.detach()),
            "kd_pred_angle": float(angle_loss.detach()),
            "kd_pred_ctr": float(ctr_loss.detach()),
        }
        return total, parts


class KDLossV1(nn.Module):
    """KD 总损失 = 融合蒸馏 + 预测蒸馏（两项均可独立置 0 做消融）。

    输入约定（与学生 AMODDetector.forward 内部完全一致，避免重复前向）：
        student_feats / teacher_feats：Neck 输出的逐层特征 list[(B,C,H,W)]
        student_preds / teacher_preds：检测头输出 dict{cls, reg, ctr 三个 list}
    """

    def __init__(self, cfg, student_dims, teacher_dims):
        super().__init__()
        self.fusion = FusionKDLoss(
            student_dims,
            teacher_dims,
            feat_weight=float(getattr(cfg, "kd_fusion_feat_weight", 1.0)),
            spatial_weight=float(getattr(cfg, "kd_fusion_spatial_weight", 0.5)),
            channel_weight=float(getattr(cfg, "kd_fusion_channel_weight", 0.5)),
            background_weight=float(getattr(cfg, "kd_fusion_bg_weight", 0.1)),
        )
        self.prediction = PredictionKDLoss(
            cls_weight=float(getattr(cfg, "kd_pred_cls_weight", 1.0)),
            ltrb_weight=float(getattr(cfg, "kd_pred_ltrb_weight", 0.5)),
            angle_weight=float(getattr(cfg, "kd_pred_angle_weight", 0.5)),
            ctr_weight=float(getattr(cfg, "kd_pred_ctr_weight", 1.0)),
            background_weight=float(getattr(cfg, "kd_pred_bg_weight", 0.1)),
        )
        self.fusion_weight = float(getattr(cfg, "kd_fusion_weight", 1.0))
        self.prediction_weight = float(getattr(cfg, "kd_pred_weight", 1.0))

    def forward(self, student_feats, teacher_feats, student_preds, teacher_preds):
        quality = teacher_quality_maps(teacher_preds)

        total = None
        parts = {}
        if self.fusion_weight > 0:
            fusion_loss, fusion_parts = self.fusion(student_feats, teacher_feats, quality)
            total = self.fusion_weight * fusion_loss
            parts.update(fusion_parts)
        if self.prediction_weight > 0:
            prediction_loss, prediction_parts = self.prediction(
                student_preds, teacher_preds, quality
            )
            value = self.prediction_weight * prediction_loss
            total = value if total is None else total + value
            parts.update(prediction_parts)

        if total is None:
            total = student_feats[0].new_zeros((), dtype=torch.float32)
        parts["kd_total"] = float(total.detach())
        return total, parts


def build_kd_loss(cfg, student_dims, teacher_dims):
    """按配置构建 KD 总损失（适配器参数会挂在学生模型上一起优化）。"""
    return KDLossV1(cfg, list(student_dims), list(teacher_dims))
