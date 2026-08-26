# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates
       
h36m_skeleton = Skeleton(
    parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
    joints_left=[4, 5, 6, 11, 12, 13],
    joints_right=[1, 2, 3, 14, 15, 16]
)

h36m_cameras_intrinsic_params = [
    {
        'id': 0,
        'center': [540, 960],
        'focal_length': [1000, 1000],
        'radial_distortion': [0, 0, 0],
        'tangential_distortion': [0, 0],
        'res_w': 1080,
        'res_h': 1920,
        'azimuth': 0,
    },
    {
        'id': 1,
        'center': [540, 960],
        'focal_length': [1000, 1000],
        'radial_distortion': [0, 0, 0],
        'tangential_distortion': [0, 0],
        'res_w': 1080,
        'res_h': 1920,
        'azimuth': 0,
    },
    {
        'id': 2,
        'center': [540, 960],
        'focal_length': [1000, 1000],
        'radial_distortion': [0, 0, 0],
        'tangential_distortion': [0, 0],
        'res_w': 1080,
        'res_h': 1920,
        'azimuth': 0,
    },
    {
        'id': 3,
        'center': [540, 960],
        'focal_length': [1000, 1000],
        'radial_distortion': [0, 0, 0],
        'tangential_distortion': [0, 0],
        'res_w': 1080,
        'res_h': 1920,
        'azimuth': 0,
    },
    {
        'id': 4,
        'center': [540, 960],
        'focal_length': [1000, 1000],
        'radial_distortion': [0, 0, 0],
        'tangential_distortion': [0, 0],
        'res_w': 1080,
        'res_h': 1920,
        'azimuth': 0,
    },
    {
        'id': 5,
        'center': [540, 960],
        'focal_length': [1000, 1000],
        'radial_distortion': [0, 0, 0],
        'tangential_distortion': [0, 0],
        'res_w': 1080,
        'res_h': 1920,
        'azimuth': 0,
    },
]

h36m_cameras_extrinsic_params = {
    'P002': [
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
    ],
    'P050': [
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
    ],
    'P049': [
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
    ],
    'P052': [
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
    ],
    'P053': [
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
    ],
    'P054': [
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
    ],
    'P055': [
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0],
        },
    ]

}


class Human36mDataset(MocapDataset):
    def __init__(self, path, remove_static_joints=True):
        super().__init__(fps=30, skeleton=h36m_skeleton)

        self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                cam.update(h36m_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')

                # Normalize camera frame
                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype(
                    'float32')
                cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2
                if 'translation' in cam:
                    cam['translation'] = cam['translation'] / 1000  # mm to meters

                # Add intrinsic parameters vector
                cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                                   cam['center'],
                                                   cam['radial_distortion'],
                                                   cam['tangential_distortion'],
                                                   [1 / cam['focal_length'][0], 0,
                                                    -cam['center'][0] / cam['focal_length'][0],
                                                    0, 1 / cam['focal_length'][1],
                                                    -cam['center'][1] / cam['focal_length'][1],
                                                    0, 0, 1]))

                # proj_matrix = np.array([1/cam['focal_length'][0], 0, -cam['center'][0]/cam['focal_length'][0],
                #                         0, 1/cam['focal_length'][1], -cam['center'][1]/cam['focal_length'][1],
                #                         0, 0, 1])
                # cam['intrinsic'] = np.concatenate(camera_intrinsics, proj_matrix)

        # Load serialized dataset
        data = np.load(path, allow_pickle=True)['positions_3d'].item()

        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    'positions': positions,
                    'cameras': self._cameras[subject],
                }

        #if remove_static_joints:
            # Bring the skeleton to 17 joints instead of the original 32
       #     self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])

            # Rewire shoulders to the correct parents
        #    self._skeleton._parents[11] = 8
        #    self._skeleton._parents[14] = 8

    def supports_semi_supervised(self):
        return True