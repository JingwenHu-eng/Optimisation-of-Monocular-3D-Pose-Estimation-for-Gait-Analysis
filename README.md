# Optimisation of Monocular 3D Pose Estimation for Gait Analysis

Optimisation of Monocular 3D Pose Estimation for Gait Analysis Using Viewpoint Selection and Gait-Informed Fine-Tuning: Towards Clinical Usage

## About this repository

This repository is for the manuscript listed above.

This work uses PoseFormerV2 for monocular 3D gait analysis. The method is based on the original PoseFormerV2 code.

At this stage, this public repository only provides the project information and setup guide.

The fine-tuning code, loss functions, and fine-tuned model are stored in a private repository during peer review. They will be made publicly available upon publication. Access can be provided to editors and reviewers upon reasonable request.

No participant-level clinical data are included in this repository.

## What this work provides

This work provides gait-informed optimisation of PoseFormerV2 for monocular 3D gait analysis.

The full private repository includes:

- `fine-tuning code.py`: main fine-tuning script.
- `loss.py`: loss functions used during fine-tuning.
- Fine-tuned model checkpoint.

The original PoseFormerV2 repository should be used for the full model architecture, base file structure, environment setup, and data format:

https://github.com/QitaoZhao/PoseFormerV2

## For medical and clinical users

This code is written for research use.

You do not need to understand all of the code to run the model, but you need to prepare the software environment first.

The main steps are:

1. Install Anaconda.
2. Create a Python environment.
3. Install PyTorch.
4. Download PoseFormerV2.
5. Add the fine-tuning files.
6. Prepare the data in the correct format.
7. Run training or testing.

## Step 1: Install Anaconda

Download and install Anaconda:

https://www.anaconda.com/download

After installation, open:

- **Anaconda Prompt** on Windows
- **Terminal** on macOS or Linux

All commands below should be typed in Anaconda Prompt or Terminal.

## Step 2: Create a new Python environment

Create a new environment called `poseformerv2`:

```bash
conda create -n poseformerv2 python=3.9
```

Activate the environment:

```bash
conda activate poseformerv2
```

You should now see something like this at the start of the command line:

```bash
(poseformerv2)
```

## Step 3: Install PyTorch

If your computer has an NVIDIA GPU and CUDA 11.7, install PyTorch with:

```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

If you do not have an NVIDIA GPU, install the CPU version:

```bash
pip install torch torchvision torchaudio
```

The CPU version can run, but it will be much slower.

## Step 4: Download PoseFormerV2

Download the original PoseFormerV2 code from GitHub:

```bash
git clone https://github.com/QitaoZhao/PoseFormerV2.git
```

Go into the folder:

```bash
cd PoseFormerV2
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Step 5: Add the fine-tuning files

After publication, the following files will be added to the public release:

```text
fine-tuning code.py
loss.py
```

These files should be placed inside the PoseFormerV2 project folder.

The fine-tuned model checkpoint will also be made available after publication.

During peer review, access can be provided to editors and reviewers upon reasonable request.

## Step 6: Prepare the data

PoseFormerV2 needs 2D pose data as input.

In simple terms:

- A video is first processed by a 2D pose detector.
- The 2D joint coordinates are saved.
- PoseFormerV2 lifts the 2D joint coordinates into 3D joint coordinates.

Please follow the original PoseFormerV2 data format.

For Human3.6M-style data, the folder usually looks like this:

```text
PoseFormerV2/
└── data/
    ├── data_2d_h36m_gt.npz
    ├── data_2d_h36m_cpn_ft_h36m_dbb.npz
    └── data_3d_h36m.npz
```

Clinical gait data must be anonymised before use.

Do not upload patient names, hospital numbers, dates of birth, faces, or other identifiable information.

## Step 7: Test that the environment works

You can test the PoseFormerV2 environment with the original PoseFormerV2 commands.

Example command for evaluation:

```bash
python run_poseformer.py -g 0 -k cpn_ft_h36m_dbb -frame 27 -frame-kept 3 -coeff-kept 3 -c checkpoint/ --evaluate NAME_ckpt.bin
```

Replace `NAME_ckpt.bin` with the real checkpoint file name.

If you do not use a GPU, the command may need to be changed depending on your local setup.

## Step 8: Fine-tuning

The fine-tuning script will be provided after publication.

The private version includes:

```text
fine-tuning code.py
loss.py
```

The fine-tuned model checkpoint is stored in a private repository during peer review.

It will be made publicly available upon publication.

## Step 9: Application

The model can be used to reconstruct 3D human pose from a monocular 2D video.

Put the input video in the following folder:

```text
PoseFormerV2/demo/video
```

For example, the input video can be named:

```text
sample_video.mp4
```

Run the following command:

```bash
python demo/vistry11.py --video sample_video.mp4
```

This command takes a monocular 2D video as input and reconstructs the 3D human pose for gait analysis.

## Notes for clinical use

This model is for research use only.

It is not a medical device.

It should not be used as the only basis for diagnosis, treatment, or clinical decision-making.

All clinical data should be anonymised before analysis.

## Acknowledgements

This work builds on PoseFormerV2:

PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation

Original PoseFormerV2 repository:

https://github.com/QitaoZhao/PoseFormerV2

We sincerely thank the PoseFormerV2 authors for making their code and pre-trained models publicly available.

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
```

## Contact

For questions about this work, please contact the corresponding author of the manuscript.
