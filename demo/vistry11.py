import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os 
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy

sys.path.append(os.getcwd())
from common.model_poseformer import PoseTransformerV2 as Model
from common.camera import *

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def get_pose3D(video_path, output_dir):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.embed_dim_ratio, args.depth, args.frames = 32, 4, 243
    args.number_of_kept_frames, args.number_of_kept_coeffs = 27, 27
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/'
    args.n_joints, args.out_joints = 17, 17

    # Load the trained 3D pose model.
    model = nn.DataParallel(Model(args=args)).cuda()

    model_dict = model.state_dict()
    # Store the PoseFormerV2 checkpoint in the root checkpoint directory.
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '27_243_45.2.bin')))[0]

    pre_dict = torch.load(model_path)
    model.load_state_dict(pre_dict['model_pos'], strict=True)

    model.eval()

    all_keypoints3d = []

    # Read the 2D keypoint file from the same directory as the video.

    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)

    npz_name = os.path.splitext(video_name)[0] + ".npz"
    npz_path = os.path.join(video_dir, npz_name)

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    print("Loading 2D keypoints from:", npz_path)

    data = np.load(npz_path, allow_pickle=True)

    # Read the key used by the expected NPZ structure.
    if 'reconstruction' in data:
        keypoints = data['reconstruction']
    elif 'keypoints' in data:
        keypoints = data['keypoints']
    else:
        raise KeyError("Cannot find keypoints in npz file")

    print("2D keypoints shape:", keypoints.shape)

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate the 3D pose sequence.
    print('\nGenerating 3D pose...')
    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        if img is None:
            continue
        img_size = img.shape

        # Select the input frame window.
        start = max(0, i - args.pad)
        end =  min(i + args.pad, len(keypoints[0])-1)

        input_2D_no = keypoints[0][start:end+1]

        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != args.frames:
            if i < args.pad:
                left_pad = args.pad - i
            if i > len(keypoints[0]) - args.pad - 1:
                right_pad = i + args.pad - (len(keypoints[0]) - 1)

            input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')

        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[ :, :, 0] *= -1
        input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)

        input_2D = input_2D[np.newaxis, :, :, :, :]

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

        N = input_2D.size(0)

        # Estimate the 3D pose.
        output_3D_non_flip = model(input_2D[:, 0])
        output_3D_flip     = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()

        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])
        all_keypoints3d.append(post_out)


    print('Generating 3D pose successful!')

    all_keypoints3d = np.stack(all_keypoints3d, axis=0)  # shape: [num_frames, 17, 3]
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_kpt_path = os.path.join(video_dir, video_name + "_3D.npy")
    np.save(output_kpt_path, all_keypoints3d)
    print(f'Saved 3D keypoints to {output_kpt_path}')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Accept either a single input file or a directory of files.
    if os.path.isdir(args.video):

        video_list = glob.glob(os.path.join(args.video, "*.mp4"))

        print("Found videos:", len(video_list))

        for video_path in video_list:
            video_name = os.path.basename(video_path).split('.')[0]

            output_dir = os.path.join(os.path.dirname(video_path), "poseformer_output", video_name)

            print("\nProcessing:", video_path)

            get_pose3D(video_path, output_dir)

            print('Generating demo successful!')

    else:

        video_path = args.video

        video_name = os.path.basename(video_path).split('.')[0]

        output_dir = os.path.join(os.path.dirname(video_path), "poseformer_output", video_name)

        get_pose3D(video_path, output_dir)

        print('Generating demo successful!')

    print("All videos processed!")


