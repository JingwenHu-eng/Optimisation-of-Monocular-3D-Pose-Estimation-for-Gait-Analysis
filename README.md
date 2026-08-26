# Optimisation of Monocular 3D Pose Estimation for Gait Analysis

Optimisation of Monocular 3D Pose Estimation for Gait Analysis Using Viewpoint Selection and Gait-Informed Fine-Tuning: Towards Clinical Usage

## About this repository

This repository accompanies the manuscript listed above.

The work uses PoseFormerV2 for monocular (single-camera) 3D gait analysis and adds gait-informed fine-tuning, including a layer-freezing strategy and additional loss functions. The full project code is provided here, so users should download this repository directly. 

No participant-level clinical data are included in this repository.

## What this work provides

This work provides gait-informed optimisation of PoseFormerV2 for monocular 3D gait analysis. The repository includes:

- `demo/`: the application used to process a walking video and create a 3D pose visualisation.
- `checkpoint/`: the location for saved model files (`.bin`).
- `run_poseformergait.py`: the main fine-tuning script, including the layer-freezing strategy and the combined gait-informed losses.
- `common/loss.py`: definitions of MPJPE, angle, bone-related, orientation, and other loss functions.
- `common/model_poseformer.py`: the PoseFormerV2 model architecture used by this project.
- `data/`: the location for 2D keypoints and 3D reference data used for training or evaluation.

## For medical and clinical users

This code is written for research use. You do not need to understand the mathematical details to run the example, but the software environment must first be prepared.

In simple terms, the application completes three operations:

1. It identifies the person's joints in each video frame (2D pose detection).
2. It estimates where those joints are in three-dimensional space (3D pose estimation).
3. It creates a video showing the original 2D view and the reconstructed 3D skeleton.

The main workflow is divided into five sections below.

## 1. Preparation

### Step 1 of 4: Install Anaconda

Download and install Anaconda from:

https://www.anaconda.com/download

After installation, open **Anaconda Prompt** on Windows. All commands below should be typed into that window, one line at a time. Press **Enter** after each line and wait for it to finish before entering the next command.

### Step 2 of 4: Create a Python environment

Create a separate software environment called `poseformerv2`:

```bash
conda create -n poseformerv2 python=3.9
```

When asked to continue, type `y` and press **Enter**. Then activate the environment:

```bash
conda activate poseformerv2
```

You should now see `(poseformerv2)` at the beginning of the command line. This indicates that the correct environment is active.

### Step 3 of 4: Install PyTorch

The supplied video application uses an NVIDIA GPU. For an NVIDIA GPU with CUDA 11.7, install the matching PyTorch version:

```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

If the computer does not have a compatible NVIDIA GPU, a CPU version of PyTorch can be installed with:

```bash
pip install torch torchvision torchaudio
```

However, CPU processing is much slower, and the supplied `demo/vis.py` application currently expects CUDA. It will require code changes before it can run without an NVIDIA GPU.

### Step 4 of 4: Download this repository and install its packages

This repository already contains the required PoseFormerV2 code and this study's changes.

This step uses Git. If the message `'git' is not recognized` appears, install Git from https://git-scm.com/downloads, close and reopen Anaconda Prompt, and repeat the command.

On the GitHub page, select the green **Code** button and copy the HTTPS address. The command will have the following form (replace `YOUR-GITHUB-USERNAME` with the real repository owner):

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/Optimisation-of-Monocular-3D-Pose-Estimation-for-Gait-Analysis.git
```

Move into the downloaded project folder. This is the exact folder name to use after cloning:

```bash
cd Optimisation-of-Monocular-3D-Pose-Estimation-for-Gait-Analysis
```

Install the remaining required packages:

```bash
pip install -r requirements.txt
```

Do not close Anaconda Prompt. The next section continues in the same window.

## 2. Application: run a case video

This section is placed before the fine-tuning instructions because most clinical users will only need to apply an existing model, not train a new one.

### 2.1 Download the application checkpoint

Download the checkpoint file from:

[Download the application checkpoint from Google Drive](https://drive.google.com/file/d/1LjG6i0CGVocs1orCPRDsPzqppSRHIhwd/view?usp=drive_link)

Place the downloaded `.bin` file in the root `checkpoint` folder, so the layout is:

```text
Optimisation-of-Monocular-3D-Pose-Estimation-for-Gait-Analysis/
└── checkpoint/
    └── on2hpWall.bin
```

The supplied `demo/vis.py` script looks for the exact name `on2hpWall.bin`. If the downloaded file has a different name, either rename it to `on2hpWall.bin` or change the following line in `demo/vis.py` so that it contains the downloaded file's real name:

```python
model_path = sorted(glob.glob(os.path.join(args.previous_dir, 'on2hpWall.bin')))[0]
```

The 2D detector also requires `pose_hrnet_w48_384x288.pth` and `yolov3.weights` in `demo/lib/checkpoint/`. These files are included when all application-folder contents are uploaded. Before running the example, check that the folder looks like this:

```text
demo/
└── lib/
    └── checkpoint/
        ├── pose_hrnet_w48_384x288.pth
        └── yolov3.weights
```

### 2.2 Replace the sample video with your own video

Open the following project folder:

```text
demo/video/
```

Remove or rename the existing `sample_video.mp4`, copy your own video into this folder, and name it exactly:

```text
sample_video.mp4
```

For the most reliable result, use a video in which the whole body is visible, only one person is walking, the camera remains still, and the joints are not hidden by furniture or other people.

Clinical videos must be anonymised before use. Do not upload patient names, hospital numbers, dates of birth, faces, or other identifiable information to a public repository.

### 2.3 Open the command window in the correct folder

If Anaconda Prompt is not already in the project folder, use the following command after replacing the first part with the actual location on your computer:

```bash
cd C:\path\to\Optimisation-of-Monocular-3D-Pose-Estimation-for-Gait-Analysis
```

For example, if the repository was downloaded into the current user's home folder on Windows, the command may be:

```bash
cd %USERPROFILE%\Optimisation-of-Monocular-3D-Pose-Estimation-for-Gait-Analysis
```

Then activate the environment if necessary:

```bash
conda activate poseformerv2
```

### 2.4 Run the application

From the root project folder, run:

```bash
python demo/vis.py --video sample_video.mp4
```

The application first detects the person's 2D joints, reconstructs the 3D pose, and then creates the final visualisation. Processing may take several minutes, depending on the video length and GPU.

The results are saved in:

```text
demo/output/sample_video/
```

The final video is normally:

```text
demo/output/sample_video/sample_video.mp4
```

## 3. What the main files and models do

### Application folders

- `demo/video/` contains the video to be analysed.
- `demo/lib/checkpoint/` contains the YOLOv3 and HRNet files used to detect the person and their 2D joints.
- `checkpoint/` contains PoseFormerV2 model checkpoints. A checkpoint is a saved model that has already learned its parameters. It can be used to analyse a video without training the model again.
- `demo/output/` contains generated 2D joint images, 3D skeleton images, and the final combined video.

### Available model checkpoints

`on2hpWall.bin` produced the best result reported in the manuscript. It combines MPJPE, bone loss, and angle loss. The other checkpoints allow the loss combinations to be compared:

The original pretrained PoseFormerV2 checkpoint can be applied directly without fine-tuning. The fine-tuned checkpoints below are optional alternatives that adapt the model to the gait objective used in this study.

| Checkpoint | Losses used during fine-tuning | Download |
|---|---|---|
| `on2hpWall.bin` | MPJPE + bone loss + angle loss; best result in the manuscript | [Google Drive](https://drive.google.com/file/d/1LjG6i0CGVocs1orCPRDsPzqppSRHIhwd/view?usp=drive_link) |
| `on2hpWAngle.bin` | MPJPE + angle loss | [Google Drive](https://drive.google.com/file/d/1Oq8kfo43Rjx83vdksjP5mzDNbvZ6Etyn/view?usp=drive_link) |
| `on2hpWBone.bin` | MPJPE + bone loss | [Google Drive](https://drive.google.com/file/d/1EiTid2GaGvELvrlyLtITq6yxzcoma5Cx/view?usp=drive_link) |
| `on2hpWMPJPE.bin` | MPJPE only | [Google Drive](https://drive.google.com/file/d/1I6Ek92XFlvS0NxC_Nx3TmlOYB5a4NMxY/view?usp=drive_link) |

In plain language:

- **MPJPE** measures the average distance between each predicted 3D joint and its reference position. A lower value means that the predicted joints are closer to the reference joints.
- **Angle loss** encourages clinically relevant joint angles, such as knee and hip angles, to match the reference motion.
- **Bone loss** encourages the predicted body-segment lengths and proportions to remain anatomically consistent.
- **Fine-tuning** means starting from a model that has already been trained and adapting part of it to gait data. This generally needs less data and computing time than training a model from the beginning.

To use a different checkpoint in the video application, place it in `checkpoint/` and change `on2hpWall.bin` in `demo/vis.py` to the selected filename.

## 4. Fine-tuning with your own data

Most users only need Section 2. Fine-tuning is intended for research teams that have paired 2D and 3D pose data and want to adapt the model to a new gait dataset.

### 4.1 Add and understand the fine-tuning files

The necessary files are already included in this repository:

```text
run_poseformergait.py
common/loss.py
```

`run_poseformergait.py` contains the prepared layer-freezing algorithm and the combined loss configuration. Layer freezing keeps most of the pretrained model unchanged while selected final layers learn from the new gait data. This reduces the risk of losing useful information learned by the original model.

The individual loss functions can be viewed or modified in `common/loss.py`. Change loss functions or weights only if you understand how this will affect training and intend to validate the resulting model.

### 4.2 Convert the data to Human3.6M format

PoseFormerV2 does not train directly from ordinary video files. The training data must first be converted to the Human3.6M-style structure expected by the code:

```text
data/
├── data_3d_DATASET_NAME.npz
└── data_2d_DATASET_NAME_KEYPOINT_NAME.npz
```

For example:

```text
data/
├── data_3d_clinic_gait.npz
└── data_2d_clinic_gait_cpn_ft_h36m_dbb.npz
```

The 2D and 3D files must use matching participant names, action names, camera views, frame counts, and the same 17-joint order. This is essential: a 2D frame and its 3D reference frame must describe the same person at the same moment.

The corresponding camera information must also be present in the Human3.6M-style data structure. If new participant identifiers or a different camera arrangement are used, the camera metadata read by `common/h36m_dataset.py` must be updated to match. Preparing this metadata normally requires support from someone familiar with the data-collection setup.

In simple terms:

- A video is first processed by a 2D pose detector.
- The 2D joint coordinates are saved in the 2D `.npz` file.
- Matching reference 3D joint coordinates are saved in the 3D `.npz` file.
- PoseFormerV2 learns to lift the 2D joint coordinates into 3D coordinates.

All clinical data must be anonymised before preparation or transfer.

### 4.3 Add the dataset name to `run_poseformergait.py`

Open `run_poseformergait.py` in a text editor. Find this section near the beginning:

```python
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
```

If the new dataset is called `clinic_gait` and has already been converted to the same H3.6M format, change the first line to:

```python
if args.dataset in ('h36m', 'clinic_gait'):
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
```

Next, either use the command-line participant lists described below or update the training and test participant names so that they exactly match the keys in the `.npz` files. Never use the same participant for both training and testing, because this would make the reported performance unreliable.

### 4.4 Run fine-tuning

Place the starting pretrained checkpoint in `checkpoint/`. To see its exact filename on Windows, run:

```bash
dir checkpoint
```

Then run the following command from the root project folder. Replace the participant lists and `PRETRAINED_CHECKPOINT.bin` with the real values in your data and checkpoint folder:

```bash
python run_poseformergait.py -g 0 -d clinic_gait -k cpn_ft_h36m_dbb -str P001,P002,P003,P004 -ste P005 -frame 243 -frame-kept 27 -coeff-kept 27 -c checkpoint --resume PRETRAINED_CHECKPOINT.bin
```

Meaning of the items that may need changing:

- `-g 0` uses the first NVIDIA GPU.
- `-d clinic_gait` must match `DATASET_NAME` in both data filenames.
- `-k cpn_ft_h36m_dbb` must match `KEYPOINT_NAME` in the 2D filename.
- `-str P001,P002,P003,P004` lists training participants, separated by commas and without spaces.
- `-ste P005` lists test participants.
- `--resume PRETRAINED_CHECKPOINT.bin` gives the exact starting checkpoint filename located in `checkpoint/`.

Training checkpoints are saved in `checkpoint/`. The final result should be evaluated on held-out participants who were not used for training.

## Notes for clinical use

This model is for research use only. It is not a medical device and should not be used as the only basis for diagnosis, treatment, or clinical decision-making.

Model output can be affected by camera position, clothing, lighting, walking aids, joint occlusion, video resolution, and populations that differ from the training data. Any clinical interpretation therefore requires appropriate validation for the intended setting and patient group.

## 5. Acknowledgements

This work builds on PoseFormerV2:

**PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation**

Original PoseFormerV2 repository:

https://github.com/QitaoZhao/PoseFormerV2

We sincerely thank the PoseFormerV2 authors for making their code and pretrained models publicly available.

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
