# Monocular-3D-Pose-Estimation-for-Gait-Kinematics
Clinically Oriented Optimisation of Monocular 3D Pose Estimation for Gait Analysis Using Viewpoint Selection and Gait-Informed Fine Tuning

## Fine-tuning Code and Model

This repository provides the fine-tuning code used for gait-informed optimisation of PoseFormerV2 for monocular 3D gait analysis.

### Files

- `fine-tuning code.py`: main fine-tuning script.
- `loss.py`: implementation of the loss functions used during fine-tuning.

Please refer to the original [PoseFormerV2](https://github.com/QitaoZhao/PoseFormerV2) repository for the full model architecture, environment setup, data format, and base file structure.

### Fine-tuned model

The fine-tuned model checkpoint is available here:

[Download fine-tuned model](https://drive.google.com/file/d/1LjG6i0CGVocs1orCPRDsPzqppSRHIhwd/view?usp=drive_link)

### Acknowledgements

This work builds on [PoseFormerV2](https://github.com/QitaoZhao/PoseFormerV2): *PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation*. We sincerely thank the PoseFormerV2 authors for making their code and pre-trained models publicly available.
If you use this repository, please also cite the original PoseFormerV2 paper:

```bibtex
@InProceedings{Zhao_2023_CVPR,
    author    = {Zhao, Qitao and Zheng, Ce and Liu, Mengyuan and Wang, Pichao and Chen, Chen},
    title     = {PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {8877-8886}
}
