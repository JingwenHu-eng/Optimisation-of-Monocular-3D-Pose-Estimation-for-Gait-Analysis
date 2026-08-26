# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Modified by Qitao Zhao (qitaozhao@mail.sdu.edu.cn)

import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import math
import logging

from einops import rearrange, repeat
from copy import deepcopy

from common.camera import *
import collections

from common.model_poseformer import *

from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import *


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义网络层
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)

model = MyModel()

args = parse_args()
log =  logging.getLogger()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ''.join(args.gpu)


try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = [p / 1000.0 for p in positions_3d]
            #anim['positions_3d'] = [p for p in positions_3d]

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

kp = keypoints  # 你脚本中 name
bad = 0
for subj in kp.keys():
    for action in kp[subj].keys():
        for cam_idx, seq in enumerate(kp[subj][action]):
            if seq is None:
                print('MISSING 2D:', subj, action, cam_idx)
                bad += 1
            else:
                shape = getattr(seq, 'shape', None)
                if shape is None:
                    print('BAD TYPE 2D:', subj, action, cam_idx, type(seq))
                    bad += 1
                elif shape[0] == 0:
                    print('EMPTY 2D:', subj, action, cam_idx, shape)
                    bad += 1
print('total bad 2D sequences:', bad)

###################
for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        #assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            if args.std != 0:
                kps += np.random.normal(loc=0.0, scale=args.std, size=kps.shape)
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

if args.dataset == 'h36m':
    subjects_train = 'P049,P050,P052,P053,P054'.split(',')
    subjects_test = 'P055'.split(',')
    #subjects_train = 'S1,S5,S6,S7,S8'.split(',')
    #subjects_test = 'S9,S11'.split(',')
else:
    subjects_train = args.subjects_train.split(',')
    subjects_test = args.subjects_test.split(',')

def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True, load_gt=False):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints_gt[subject][action] if load_gt else keypoints[subject][action]

            # iterate across camera views for this (subject,action)
            for cam_idx in range(len(poses_2d)):
                p2d = poses_2d[cam_idx]

                # 2D 必须存在并且非空
                if p2d is None:
                    print(f"[FETCH] skip missing 2D: {subject} {action} cam{cam_idx}")
                    continue
                if getattr(p2d, 'shape', None) is None or p2d.shape[0] == 0:
                    print(f"[FETCH] skip empty 2D: {subject} {action} cam{cam_idx} shape={getattr(p2d,'shape',None)}")
                    continue

                # 如果需要 3D，3D 也必须存在且非空（确保配对）
                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d_curr = dataset[subject][action]['positions_3d']
                    if cam_idx >= len(poses_3d_curr):
                        print(f"[FETCH] skip missing 3D cam index: {subject} {action} cam{cam_idx}")
                        continue
                    p3d = poses_3d_curr[cam_idx]
                    if p3d is None or getattr(p3d, 'shape', None) is None or p3d.shape[0] == 0:
                        print(f"[FETCH] skip empty 3D: {subject} {action} cam{cam_idx}")
                        continue
                    # 通过检查，追加 3D
                    out_poses_3d.append(p3d)

                # append 2D and camera intrinsics if present
                out_poses_2d.append(p2d)
                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    if cam_idx < len(cams) and 'intrinsic' in cams[cam_idx]:
                        out_camera_params.append(cams[cam_idx]['intrinsic'])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    # 下采样 / subset 操作（保留原逻辑）
    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

receptive_field = args.number_of_frames
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field -1) // 2 # Padding on each side
min_loss = 100000
width = cam['res_w']
height = cam['res_h']
num_joints = keypoints_metadata['num_joints']

#########################################PoseTransformer
# if args.resume or args.from_scratch:
model_pos_train = PoseTransformerV2(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1, args=args)

model_pos = PoseTransformerV2(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0, args=args)

#################
causal_shift = 0
model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()
    model_pos_train = nn.DataParallel(model_pos_train)
    model_pos_train = model_pos_train.cuda()


if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
    model_pos.load_state_dict(checkpoint['model_pos'], strict=False)


test_generator = UnchunkedGenerator(None, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = inputs_3d.permute(1,0,2,3)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d, inputs_3d_p

###################
def freeze_module(module):
    module.requires_grad_(False)

def unfreeze_module(module):
    module.requires_grad_(True)

def get_core_model(model):
    # 兼容 DataParallel / 非 DataParallel
    return model.module if isinstance(model, nn.DataParallel) else model

def freeze_module(module):
    module.requires_grad_(False)

def unfreeze_module(module):
    module.requires_grad_(True)

def get_core_model(model):
    # 兼容 DataParallel / 非 DataParallel
    return model.module if isinstance(model, nn.DataParallel) else model

def configure_finetune(
    model,
    mode='last2',
    frozen_eval=True,
    unfreeze_last_spatial=True,
    unfreeze_norms=True,
    unfreeze_pos_embed=True,
    unfreeze_freq_embedding=False,
):
    m = get_core_model(model)

    # 1) 先全部冻结
    for p in m.parameters():
        p.requires_grad = False

    # 2) 永远打开输出部分
    unfreeze_module(m.weighted_mean)
    unfreeze_module(m.weighted_mean_)
    unfreeze_module(m.head)

    # 3) temporal mixed blocks
    if mode == 'head':
        pass
    elif mode == 'last1':
        for blk in m.blocks[-1:]:
            unfreeze_module(blk)
    elif mode == 'last2':
        for blk in m.blocks[-2:]:
            unfreeze_module(blk)
    elif mode == 'last3':
        for blk in m.blocks[-3:]:
            unfreeze_module(blk)
    else:
        raise ValueError(f'Unknown finetune mode: {mode}')

    # 4) optional: last spatial block
    if unfreeze_last_spatial:
        for blk in m.Spatial_blocks[-1:]:
            unfreeze_module(blk)

    # 5) optional: norm layers
    if unfreeze_norms:
        unfreeze_module(m.Temporal_norm)

    # 6) optional: pos embeddings
    if unfreeze_pos_embed:
        m.Spatial_pos_embed.requires_grad = True
        m.Temporal_pos_embed.requires_grad = True
        m.Temporal_pos_embed_.requires_grad = True

    # 7) optional: freq embedding
    if unfreeze_freq_embedding:
        unfreeze_module(m.Freq_embedding)

    # 8) frozen modules -> eval
    if frozen_eval:
        m.Joint_embedding.eval()
        if not unfreeze_freq_embedding:
            m.Freq_embedding.eval()

        # LayerNorm 不靠 train/eval 切换统计，但这里保持整体风格一致
        if not unfreeze_norms:
            m.Temporal_norm.eval()

        frozen_spatial_blocks = m.Spatial_blocks[:-1] if unfreeze_last_spatial else m.Spatial_blocks
        for blk in frozen_spatial_blocks:
            blk.eval()

        if mode == 'head':
            frozen_blocks = m.blocks
        elif mode == 'last1':
            frozen_blocks = m.blocks[:-1]
        elif mode == 'last2':
            frozen_blocks = m.blocks[:-2]
        elif mode == 'last3':
            frozen_blocks = m.blocks[:-3]

        for blk in frozen_blocks:
            blk.eval()

def print_trainable_parameters(model):
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            print('[TRAIN]', name)
    print(f'trainable params: {trainable}/{total} ({100.0 * trainable / total:.2f}%)')

def print_trainable_parameters(model):
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            print('[TRAIN]', name)
    print(f'trainable params: {trainable}/{total} ({100.0 * trainable / total:.2f}%)')



if not args.evaluate:
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

    lr = args.learning_rate
    # 选择一种微调策略：'head' / 'last1' / 'last2'
    FINETUNE_MODE = 'last2'

    configure_finetune(
        model_pos_train,
        mode='last2',
        frozen_eval=True,
        unfreeze_last_spatial=False,
        unfreeze_norms=True,
        unfreeze_pos_embed=True,
        unfreeze_freq_embedding=False,
    )

    print_trainable_parameters(model_pos_train)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model_pos_train.parameters()),
        lr=lr,
        weight_decay=0.1
    )

    lr_decay = args.lr_decay
    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    train_generator = ChunkedGenerator(args.batch_size//args.stride, None, poses_train, poses_train_2d, args.stride,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

    if args.resume:
        # 如果你是“改变冻结策略后继续微调”，不要加载旧 optimizer
        print('Skipping optimizer state because trainable parameter groups changed.')
        epoch = 0  # 建议当作一次新的 finetune 开始
        # 如果你特别想沿用数据顺序，可以保留这句；不需要的话也可以去掉
        if 'random_state' in checkpoint:
            train_generator.set_random_state(checkpoint['random_state'])

        # 这里建议用你当前设定的 lr，而不是 checkpoint 里的 lr
        lr = args.learning_rate

    print('** Note: reported losses are averaged over all frames.')
    print('** The final evaluation will be carried out after the last training epoch.')

    while epoch < args.epochs:
        model_pos.train()
        epoch_loss_mpjpe = 0
        epoch_loss_angle = 0
        epoch_loss_orientation = 0
        epoch_loss_bonelen_temporal = 0
        #epoch_loss_boneratio = 0
        epoch_batches = 0
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        N = 0
        N_semi = 0
        # 打印本 epoch 使用的训练数据
        model_pos_train.train()

        for _, batch_3d, batch_2d in train_generator.next_epoch():
            if batch_3d is None or batch_3d.shape[0] == 0:
                continue  # 跳过空 batch
            if batch_3d.shape[1] != train_generator.chunk_length:
                continue  # 确保每个 batch 长度和 chunk_length 一致
            B = batch_3d.shape[0]
            wanted = args.batch_size // args.stride
            if B != wanted:
                continue  # 只保留完整批次
            if batch_3d is not None and batch_3d.shape[0] != args.batch_size:
                continue
            if batch_2d is not None and batch_2d.shape[0] != args.batch_size:
                continue
            T = batch_3d.shape[1]  # 时间长度
            wanted_B = args.batch_size // args.stride
            wanted_T = train_generator.batch_3d.shape[1] if batch_3d is not None else gen.batch_2d.shape[1]

            if B != wanted_B or T != wanted_T:
                continue  # 跳过不完整 batch
            inputs_3d = torch.from_numpy(batch_3d.astype('float32')) # [512, 1, 17, 3]
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')) # [512, 3, 17, 2]

            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
            inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()

            # Predict 3D poses
            predicted_3d_pos = model_pos_train(inputs_2d)

            if predicted_3d_pos.shape != inputs_3d.shape:
                continue  # 跳过这个批次

            from common.loss import (
                angle_loss, p_mpjpe, weighted_bonelen_loss,orientation_loss,segment_loss,limb_pairs,
                weighted_boneratio_loss, ANGLE_PAIRS, orientation_parirs
            )

            # ============= 各个子损失 =============
            # 角度约束
            loss_angle = angle_loss(predicted_3d_pos, inputs_3d, ANGLE_PAIRS)
            loss_orientation = orientation_loss(predicted_3d_pos, inputs_3d,orientation_parirs)

            # MPJPE（关节欧氏距离）
            loss_mpjpe = p_mpjpe(predicted_3d_pos, inputs_3d)
            loss_bonelen_temporal = segment_loss(predicted_3d_pos, inputs_3d, limb_pairs)

            # ===== 正确的骨长计算 (H36M骨架) =====
            # ===== 正确的骨长计算 (H36M骨架) =====
            # H36M_BONES = [
            #     (0, 1), (1, 2), (2, 3),  # 右腿
            #     (0, 4), (4, 5), (5, 6)  # 左腿
            #     #(0, 7), (7, 8), (8, 9), (9, 10),  # 躯干 + 头
            #     #(8, 11), (11, 12), (12, 13),  # 左臂
            #     #(8, 14), (14, 15), (15, 16)  # 右臂
            # ]


            # def compute_skeleton_length_per_bone(poses, bones=H36M_BONES):
            #     """
            #     计算每帧每条骨骼长度
            #     poses: (B, T, J, 3)
            #     返回: (B, num_bones)
            #     """
            #     B, T, J, _ = poses.shape
            #     num_bones = len(bones)
            #     bone_lengths = torch.empty(B, num_bones, device=poses.device)
            #
            #     for i, (p, c) in enumerate(bones):
            #         vec = poses[:, 0, c, :] - poses[:, 0, p, :]  # 只取 T=1 的帧
            #         bone_lengths[:, i] = torch.norm(vec, dim=-1)
            #
            #     return bone_lengths  # [B, num_bones]
            #
            # # 计算归一化躯干长度损失
            # # 计算预测与GT骨骼长度
            # pred_skel_len = compute_skeleton_length_per_bone(predicted_3d_pos)  # [1024, 16]
            # gt_skel_len = compute_skeleton_length_per_bone(inputs_3d)  # [1024, 16]
            # #print("predicted_3d_pos shape:", pred_skel_len.shape)
            #
            # #print("inputs_3d shape:", gt_skel_len.shape)
            # # 归一化每个骨骼长度
            # pred_norm = pred_skel_len / (pred_skel_len.mean(dim=1, keepdim=True) + 1e-8)
            # gt_norm = gt_skel_len / (gt_skel_len.mean(dim=1, keepdim=True) + 1e-8)
            #
            # loss_bonelen_temporal = torch.mean(torch.abs(pred_norm - gt_norm))

            #loss_boneratio = weighted_boneratio_loss(pred_bone_len, gt_bone_len)

            # (可选) 方向约束：和角度类似，用关节向量夹角来约束，不想复杂化先用 loss_angle 替代

            # ============= 总 loss (加权求和) =============
            loss_total = (
                    1.0 * loss_mpjpe +  # 主要项
                    15 * loss_angle  + # 骨骼角度
                    0.2 * loss_orientation  # 骨骼比例
            )

            epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_total.item()
            epoch_loss_mpjpe += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_mpjpe.item()
            epoch_loss_angle += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_angle.item()
            epoch_loss_bonelen_temporal += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_bonelen_temporal.item()
            epoch_loss_orientation += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_orientation.item()
            #epoch_loss_boneratio += loss_boneratio.item()
            epoch_batches += 1
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            #from common.loss import angle_loss, ANGLE_PAIRS

            #loss_3d_pos = angle_loss(predicted_3d_pos, inputs_3d, ANGLE_PAIRS)
            #loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            #epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()

            #N += inputs_3d.shape[0] * inputs_3d.shape[1]

            #loss_total = loss_3d_pos
            loss_3d_pos = loss_total

            loss_total.backward()
            # 梯度裁剪，避免梯度爆炸
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            del inputs_2d, inputs_3d, loss_3d_pos, predicted_3d_pos
            torch.cuda.empty_cache()

        if N == 0:
            print(f"Epoch {epoch + 1}: No valid batches found, skipping loss append.")
        else:
            losses_3d_train.append(epoch_loss_3d_train / N)

        torch.cuda.empty_cache()

        # End-of-epoch evaluation
        with (torch.no_grad()):
            model_pos.load_state_dict(model_pos_train.state_dict(), strict=False)
            model_pos.eval()

            epoch_loss_3d_valid = 0
            NN = 0
            if not args.no_eval:
                # 验证集子损失累加器
                epoch_val_loss_mpjpe = 0
                epoch_val_loss_angle = 0
                epoch_val_loss_bonelen = 0
                epoch_val_loss_orientation = 0
                epoch_loss_3d_val = 0
                NN = 0

                # Evaluate on test set
                for _, batch, batch_2d in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32'))  # [1, 2356, 17, 3]
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))  # [1, 2358, 17, 2]

                    ##### apply test-time-augmentation (following Videopose3d)
                    inputs_2d_flip = inputs_2d.clone()
                    inputs_2d_flip[:, :, :, 0] *= -1
                    inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

                    ##### convert size
                    inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)  # [2356, 3, 17, 2]
                    inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)

                    if torch.cuda.is_available():
                        inputs_2d = inputs_2d.cuda()
                        inputs_2d_flip = inputs_2d_flip.cuda()
                        inputs_3d = inputs_3d.cuda()

                    inputs_3d[:, :, 0] = 0

                    predicted_3d_pos = model_pos(inputs_2d)
                    predicted_3d_pos_flip = model_pos(inputs_2d_flip)
                    predicted_3d_pos_flip[:, :, :, 0] *= -1
                    predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                              joints_right + joints_left]

                    predicted_3d_pos = torch.mean(
                        torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1, keepdim=True
                    )

                    # ---------------- 对齐 predicted 和 target ----------------
                    min_batch = min(predicted_3d_pos.shape[0], inputs_3d.shape[0])
                    min_frames = min(predicted_3d_pos.shape[1], inputs_3d.shape[1])
                    predicted_3d_pos = predicted_3d_pos[:min_batch, :min_frames]
                    inputs_3d = inputs_3d[:min_batch, :min_frames]
                    # ----------------------------------------------------------

                    # ========= 子损失 =========
                    loss_mpjpe = p_mpjpe(predicted_3d_pos, inputs_3d)
                    loss_angle = angle_loss(predicted_3d_pos, inputs_3d)
                    loss_bonelen = segment_loss(predicted_3d_pos, inputs_3d, limb_pairs)
                    loss_orientation = orientation_loss(predicted_3d_pos, inputs_3d,orientation_parirs)

                    # # 前后帧骨骼长度一致性（时序约束）
                    # pred_skel_len1 = compute_skeleton_length_per_bone(predicted_3d_pos)  # [1024, 16]
                    # gt_skel_len1 = compute_skeleton_length_per_bone(inputs_3d)  # [1024, 16]
                    # # print("predicted_3d_pos shape:", pred_skel_len.shape)
                    #
                    # # print("inputs_3d shape:", gt_skel_len.shape)
                    # # 归一化每个骨骼长度
                    # pred_norm1 = pred_skel_len1 / (pred_skel_len1.mean(dim=1, keepdim=True) + 1e-8)
                    # gt_norm1 = gt_skel_len1 / (gt_skel_len1.mean(dim=1, keepdim=True) + 1e-8)
                    #
                    # loss_bonelen = torch.mean(torch.abs(pred_norm1 - gt_norm1))
                    # 计算归一化躯干长度损失

                    # 骨骼比例约束
                    #loss_boneratio = weighted_boneratio_loss(predicted_3d_pos, inputs_3d)
                    loss_total_val = (
                            1.0 * loss_mpjpe +  # 主要项
                            15 * loss_angle + # 骨骼角度
                           0.2 * loss_orientation  # 骨骼比例
                    )
                    # ========= 累加 =========
                    num_samples = inputs_3d.shape[0] * inputs_3d.shape[1]
                    epoch_loss_3d_val += num_samples *loss_total_val
                    epoch_val_loss_mpjpe += num_samples * loss_mpjpe.item()
                    epoch_val_loss_angle += num_samples * loss_angle.item()
                    epoch_val_loss_bonelen += num_samples * loss_bonelen.item()
                    epoch_val_loss_orientation += num_samples * loss_orientation.item()
                    NN += num_samples

                    del inputs_2d, inputs_2d_flip, inputs_3d
                    del loss_mpjpe, loss_angle, loss_bonelen, loss_orientation
                    #, loss_boneratio
                    del predicted_3d_pos, predicted_3d_pos_flip
                    torch.cuda.empty_cache()

                # ========= 保存总 loss（和原来一样） =========
                losses_3d_valid.append(epoch_loss_3d_val / NN)

                # ========= 计算平均子损失 =========
                val_mpjpe = (epoch_val_loss_mpjpe / NN)
                val_angle = (epoch_val_loss_angle / NN)
                val_bonelen = (epoch_val_loss_bonelen / NN)
                val_orientation = (epoch_val_loss_orientation / NN)

                # 这里不用立即 print，等到 epoch 完成时再和 train 一起打印

        elapsed = (time() - start_time) / 60

        if args.no_eval:
            print('[%d] time %.2f lr %.6f '
                  '3d_train %.3f p_mpjpe %.3f angle %.6f bonelen %.6f' % (
                      epoch + 1,
                      elapsed,
                      lr,
                      losses_3d_train[-1] * 1000,  # 总损失还是保持和原来一致
                      (epoch_loss_mpjpe / N) * 1000,  # MPJPE 转换成 mm
                      (epoch_loss_angle / N),  # 角度 loss 原始值
                      (epoch_loss_bonelen_temporal / N) * 1000 # 骨长 loss 原始值
                      ))  # 骨比 loss 原始值
        else:
            print('[%d] time %.2f lr %.6f '
                  '3d_train %.3f p_mpjpe %.3f angle %.6f bonelen %.6f orientation %.6f 3d_valid %.3f mpjpe %.3f angle %.6f bonelen %.6f orientation %.6f' % (
                      epoch + 1,
                      elapsed,
                      lr,
                      losses_3d_train[-1] * 1000,
                      (epoch_loss_mpjpe / N) * 1000,
                      (epoch_loss_angle / N),
                      (epoch_loss_bonelen_temporal / N)* 1000,
                      (epoch_loss_orientation / N),
                      losses_3d_valid[-1] * 1000,
                      (epoch_val_loss_mpjpe / NN) * 1000,
                      (epoch_val_loss_angle / NN),
                      (epoch_val_loss_bonelen / NN)* 1000,
                      (epoch_val_loss_orientation / NN)
                  ))
        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        if not os.path.exists(args.checkpoint):
            os.mkdir(args.checkpoint)

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
            }, chk_path)

        #### save best checkpoint
        best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin'.format(epoch))
        if losses_3d_valid[-1] * 1000 < min_loss:
            min_loss = losses_3d_valid[-1] * 1000
            print("save best checkpoint")
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
            }, best_chk_path)

        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 3:
            if 'matplotlib' not in sys.modules:
                import matplotlib

                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))

            plt.close('all')


# Evaluate
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    if args.dataset.startswith('mpi'):
        epoch_loss_pck = 0
        epoch_loss_auc = 0
    with torch.no_grad():
        if not use_trajectory_model:
            model_pos.eval()
        # else:
            # model_traj.eval()
        N = 0
        for cam, batch, batch_2d in test_generator.next_epoch():
            cam = torch.from_numpy(cam.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')) # [b, f, p, 2]
            inputs_3d = torch.from_numpy(batch.astype('float32'))

            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip [:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]

            ##### convert size
            inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)
            inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)
            cam = cam.repeat(inputs_2d.size(0), 1)

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()
                inputs_3d = inputs_3d.cuda()
                cam = cam.cuda()
            inputs_3d[:, :, 0] = 0

            predicted_3d_pos = model_pos(inputs_2d)
            predicted_3d_pos_flip = model_pos(inputs_2d_flip)

            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                      joints_right + joints_left]

            predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                          keepdim=True)

            del inputs_2d, inputs_2d_flip
            torch.cuda.empty_cache()

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()


            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * mpjpe(predicted_3d_pos, inputs)
            if args.dataset.startswith('mpi'):
                epoch_loss_pck += inputs_3d.shape[0]*inputs_3d.shape[1] * pck(predicted_3d_pos, inputs)
                epoch_loss_auc += inputs_3d.shape[0]*inputs_3d.shape[1] * auc(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    if args.dataset.startswith('mpi'):
        e4 = (epoch_loss_pck / N)
        e5 = (epoch_loss_auc / N)
        print('Protocol #4 PCK:', e4)
        print('Protocol #5 AUC:', e5)
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    if args.dataset.startswith('mpi'):
        return e1, e2, e3, e4, e5, ev
    else:
        return e1, e2, e3, ev

if args.render:
    print('Rendering...')

    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
            ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')

    gen = UnchunkedGenerator(None, [ground_truth], [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, return_predictions=True)

    if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)
        # Predictions are in camera space
        np.save(args.viz_export, prediction)

    if args.viz_output is not None:
        if ground_truth is not None:
            # Reapply trajectory
            trajectory = ground_truth[:, :1]
            ground_truth[:, 1:] += trajectory
            prediction += trajectory

        # Invert camera transformation
        cam = dataset.cameras()[args.viz_subject][args.viz_camera]
        if ground_truth is not None:
            prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
            ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
        else:
            # If the ground truth is not available, take the camera extrinsic params from a random subject.
            # They are almost the same, and anyway, we only need this for visualization purposes.
            for subject in dataset.cameras():
                if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                    rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                    break
            prediction = camera_to_world(prediction, R=rot, t=0)
            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        anim_output = {'Reconstruction': prediction}
        if ground_truth is not None and not args.viz_no_ground_truth:
            anim_output['Ground truth'] = ground_truth

        input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

        from common.visualization import render_animation

        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                         input_video_skip=args.viz_skip)

else:
    print('Evaluating...')
    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))


    def fetch_actions(actions):
        out_poses_3d = []
        out_poses_2d = []
        out_camera_params = []

        for subject, action in actions:
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            poses_3d = dataset[subject][action]['positions_3d']
            #assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_3d)):  # Iterate across cameras
                out_poses_3d.append(poses_3d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                #assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

        stride = args.downsample
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d


    def run_evaluation(actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        if args.dataset.startswith('mpi'):
            errors_p4 = []
            errors_p5 = []
        errors_vel = []

        for action_key in actions.keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            cameras_act, poses_act, poses_2d_act = fetch_actions(actions[action_key])
            gen = UnchunkedGenerator(cameras_act, poses_act, poses_2d_act,
                                     pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                     kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                     joints_right=joints_right)
            if args.dataset.startswith('mpi'):
                e1, e2, e3, e4, e5, ev = evaluate(gen, action_key)
                errors_p4.append(e4)
                errors_p5.append(e5)
            else:
                e1, e2, e3, ev = evaluate(gen, action_key)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)

        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
        if args.dataset.startswith('mpi'):
            print('PCK:', round(np.mean(errors_p4), 1))
            print('AUC:', round(np.mean(errors_p5), 1))
        print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')


    if not args.by_subject:
        run_evaluation(all_actions, action_filter)
    else:
        for subject in all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(all_actions_by_subject[subject], action_filter)
            print('')
     
