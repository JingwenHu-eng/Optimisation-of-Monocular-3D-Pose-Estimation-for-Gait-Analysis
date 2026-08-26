# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------------
# Human3.6M / COCO 的关节角度三元组
# -------------------------------
ANGLE_PAIRS = [
    (1, 2, 3),   # xigai
    (4, 5, 6),  # xigai
    (2, 1, 3), # hip
    (5, 4, 6) # hip
]

orientation_parirs = [
    (1, 2, 3),   # xigai
    (4, 5, 6),  # xigai
    (0, 1, 2), # hip
    (0, 4, 5) # hip
]

limb_pairs = [
    ((1, 2), (4, 5)),
    ((2, 3), (5, 6))
]

def procrustes_align(predicted, target):
    """
    返回对齐后的 predicted（shape 不变）
    """
    assert predicted.shape == target.shape, f"Shape mismatch: {predicted.shape} vs {target.shape}"
    assert predicted.dim() == 4, f"Expected [B,F,J,3], got {predicted.shape}"

    B, F, J, C = predicted.shape
    assert C == 3, f"Expected last dim=3 (x,y,z), got {C}"

    # [B*F, J, 3]
    predicted = predicted.reshape(B*F, J, 3)
    target    = target.reshape(B*F, J, 3)

    # 去中心化
    muX = torch.mean(target, dim=1, keepdim=True)
    muY = torch.mean(predicted, dim=1, keepdim=True)

    X0 = target - muX
    Y0 = predicted - muY

    # 归一化
    normX = torch.sqrt(torch.sum(X0 ** 2, dim=(1, 2), keepdim=True))
    normY = torch.sqrt(torch.sum(Y0 ** 2, dim=(1, 2), keepdim=True))

    X0 = X0 / (normX + 1e-8)
    Y0 = Y0 / (normY + 1e-8)

    # SVD
    H = torch.matmul(X0.transpose(1, 2), Y0)  # [B*F, 3, 3]
    U, s, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-2, -1)
    R = torch.matmul(V, U.transpose(-2, -1))

    # 确保旋转矩阵右手系
    detR = torch.det(R).unsqueeze(-1)

    V_new = V.clone()
    s_new = s.clone()

    V_new[:, :, -1] = V[:, :, -1] * torch.sign(detR)
    s_new[:, -1] = s[:, -1] * torch.sign(detR.squeeze(-1))

    R = torch.matmul(V_new, U.transpose(-2, -1))

    # 缩放 + 平移
    tr = torch.sum(s_new, dim=1, keepdim=True).unsqueeze(-1)
    a = tr * normX / (normY + 1e-8)
    t = muX - a * torch.matmul(muY, R)

    predicted_aligned = a * torch.matmul(predicted, R) + t

    # reshape back
    pred_aligned = predicted_aligned.reshape(B, F, J, 3)
    return pred_aligned


def segment_loss(predicted, target, limb_pairs, eps=1e-6):
    """
    limb_pairs: [
        ((l_sho, l_elb), (r_sho, r_elb)),
        ...
    ]
    """
    predicted = procrustes_align(predicted, target)
    loss = 0.0
    count = 0

    for (l_pair, r_pair) in limb_pairs:
        (lp, lc), (rp, rc) = l_pair, r_pair

        l_pred = predicted[:, :, lc] - predicted[:, :, lp]
        r_pred = predicted[:, :, rc] - predicted[:, :, rp]

        l_gt = target[:, :, lc] - target[:, :, lp]
        r_gt = target[:, :, rc] - target[:, :, rp]

        # L1长度
        l_pred_len = torch.sum(torch.abs(l_pred), dim=-1)
        r_pred_len = torch.sum(torch.abs(r_pred), dim=-1)

        l_gt_len = torch.sum(torch.abs(l_gt), dim=-1)
        r_gt_len = torch.sum(torch.abs(r_gt), dim=-1)

        B_pred = l_pred_len + r_pred_len
        B_gt = l_gt_len + r_gt_len

        loss += torch.mean(torch.abs(B_pred - B_gt))
        count += 1

    return loss / count
# -------------------------------
# 基础 Loss
# -------------------------------

def orientation_loss(predicted, target, triplets, eps=1e-6):
    """
    Body Part Orientation Loss (L1 on normal vectors)

    predicted: (B, T, J, 3)
    target:    (B, T, J, 3)
    triplets:  [(A, B, C), ...]  每个 body part 用三个点定义平面

    返回:
        标量 loss
    """
    predicted = procrustes_align(predicted, target)
    B, T, J, _ = predicted.shape
    loss = 0.0
    count = 0

    for (a, b, c) in triplets:
        # 构造向量 AB 和 AC
        AB_pred = predicted[:, :, b] - predicted[:, :, a]
        AC_pred = predicted[:, :, c] - predicted[:, :, a]

        AB_gt = target[:, :, b] - target[:, :, a]
        AC_gt = target[:, :, c] - target[:, :, a]

        # 叉乘得到法向量
        n_pred = torch.cross(AB_pred, AC_pred, dim=-1)
        n_gt   = torch.cross(AB_gt, AC_gt, dim=-1)

        # 归一化（避免除0）
        n_pred = F.normalize(n_pred, dim=-1, eps=eps)
        n_gt   = F.normalize(n_gt, dim=-1, eps=eps)

        # L1 loss（对应论文里的 || · ||_1）
        loss += torch.mean(torch.abs(n_pred - n_gt))
        count += 1

    return loss / count

def angle_loss(predicted, target, angle_pairs=ANGLE_PAIRS, eps=1e-6):
    """
    关节角度 loss (使用 cos 差异替代 acos，避免梯度爆炸)
    predicted: (B, T, J, 3)
    target:    (B, T, J, 3)
    """
    B, T, J, _ = predicted.shape
    loss = 0.0
    count = 0

    for (p, j, c) in angle_pairs:
        v1_pred = predicted[:, :, j] - predicted[:, :, p]
        v2_pred = predicted[:, :, c] - predicted[:, :, j]
        v1_gt = target[:, :, j] - target[:, :, p]
        v2_gt = target[:, :, c] - target[:, :, j]

        v1_pred = F.normalize(v1_pred, dim=-1, eps=eps)
        v2_pred = F.normalize(v2_pred, dim=-1, eps=eps)
        v1_gt = F.normalize(v1_gt, dim=-1, eps=eps)
        v2_gt = F.normalize(v2_gt, dim=-1, eps=eps)

        cos_pred = torch.sum(v1_pred * v2_pred, dim=-1)
        cos_gt = torch.sum(v1_gt * v2_gt, dim=-1)

        # 用 cos 差异而不是 acos 差异，更稳定
        loss += torch.mean((cos_pred - cos_gt) ** 2)
        count += 1

    return loss / count


def mpjpe(predicted, target):
    """Mean per-joint position error (Protocol #1)"""
    t = min(predicted.shape[1], target.shape[1])
    predicted = predicted[:, :t]
    target = target[:, :t]
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=-1))

def weighted_mpjpe(predicted, target, w):
    """
    Weighted MPJPE
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=-1))


def p_mpjpe(predicted, target):
    """
    MPJPE after rigid alignment (Protocol #2, Procrustes alignment)
    输入必须是 [B, F, J, 3]
    """
    assert predicted.shape == target.shape, f"Shape mismatch: {predicted.shape} vs {target.shape}"
    assert predicted.dim() == 4, f"Expected [B,F,J,3], got {predicted.shape}"

    B, F, J, C = predicted.shape
    assert C == 3, f"Expected last dim=3 (x,y,z), got {C}"

    # [B*F, J, 3]
    predicted = predicted.reshape(B*F, J, 3)
    target    = target.reshape(B*F, J, 3)

    # 去中心化
    muX = torch.mean(target, dim=1, keepdim=True)
    muY = torch.mean(predicted, dim=1, keepdim=True)

    X0 = target - muX
    Y0 = predicted - muY

    # 归一化
    normX = torch.sqrt(torch.sum(X0 ** 2, dim=(1, 2), keepdim=True))
    normY = torch.sqrt(torch.sum(Y0 ** 2, dim=(1, 2), keepdim=True))

    X0 = X0 / (normX + 1e-8)
    Y0 = Y0 / (normY + 1e-8)

    # SVD
    H = torch.matmul(X0.transpose(1, 2), Y0)  # [B*F, 3, 3]
    U, s, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-2, -1)
    R = torch.matmul(V, U.transpose(-2, -1))

    # 确保旋转矩阵右手系
    detR = torch.det(R).unsqueeze(-1)

    V_new = V.clone()
    s_new = s.clone()

    V_new[:, :, -1] = V[:, :, -1] * torch.sign(detR)
    s_new[:, -1] = s[:, -1] * torch.sign(detR.squeeze(-1))

    R = torch.matmul(V_new, U.transpose(-2, -1))

    # 缩放 + 平移
    tr = torch.sum(s_new, dim=1, keepdim=True).unsqueeze(-1)
    a = tr * normX / (normY + 1e-8)
    t = muX - a * torch.matmul(muY, R)

    predicted_aligned = a * torch.matmul(predicted, R) + t
    return torch.mean(torch.norm(predicted_aligned - target, dim=-1))


def p_mpjpeold(predicted, target):
    """
    MPJPE after rigid alignment (Protocol #2)
    """
    assert predicted.shape == target.shape
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY
    t = muX - a*np.matmul(muY, R)

    predicted_aligned = a*np.matmul(predicted, R) + t
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=-1))


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only)
    """
    t = min(predicted.shape[1], target.shape[1])
    predicted = predicted[:, :t]
    target = target[:, :t]
    assert predicted.shape == target.shape

    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)


def mean_velocity_error(predicted, target):
    """
    Mean velocity error
    """
    assert predicted.shape == target.shape
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=-1))


# -------------------------------
# 骨骼相关 Loss
# -------------------------------
def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    """骨骼长度约束"""
    return 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()

def bonelen_temporal_loss(pred_bone_len):
    # (B, T, n_bones)
    bone_len_diff = pred_bone_len[:, 1:, :] - pred_bone_len[:, :-1, :]
    return 0.1 * torch.mean(bone_len_diff ** 2)

def weighted_boneratio_loss(predict_3d_length, gt_3d_length, eps=1e-3):
    """
    骨骼比例约束 (分母加 eps 避免除 0 或过小骨骼导致爆炸)
    """
    ratio_error = (predict_3d_length - gt_3d_length) / (gt_3d_length + eps)
    return 0.1 * torch.pow(ratio_error, 2).mean()

