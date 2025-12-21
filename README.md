<div align="center">

<img src="assets/Nankai_University_logo.svg" height="80px" alt="Nankai University" >
<img src="assets/JittorLogo_Final1220.svg" height="80px" alt="Jittor" >

# Jittor-DEA-Net

**DEA-Net: Single image dehazing based on detail-enhanced convolution and content-guided attention (IEEE TIP 2024)**

[![Jittor](https://img.shields.io/badge/Framework-Jittor-EA3323.svg)](https://cg.cs.tsinghua.edu.cn/jittor/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/ZR-1N/Jittor-DEA-Net?style=social)](https://github.com/ZR-1N/Jittor-DEA-Net)

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2301.04805)
[![Original Repo](https://img.shields.io/badge/Official-PyTorch_Repo-EE4C2C.svg)](https://github.com/cecret3350/DEA-Net)

[English](#-introduction) | [简体中文](#-项目简介)

</div>

---

## 📖 Introduction

This repository is an official implementation of **DEA-Net** based on the [Jittor (计图)](https://cg.cs.tsinghua.edu.cn/jittor/) deep learning framework. This project is part of the **"Sprouts Program" at Nankai University**.

DEA-Net proposes a novel detail-enhanced convolution (DEConv) and content-guided attention (CGA) mechanism to effectively restore haze-free images. By leveraging Jittor's **Just-In-Time (JIT) compilation** and **operator fusion**, this implementation achieves competitive training efficiency compared to the PyTorch version while maintaining algorithmic performance.

### Overall Architecture
<div align="center">
  <img src="fig/architecture.png" alt="Overall Architecture" width="90%">
</div>

### Results

<img src="fig/results.png" alt="Results" style="zoom:20%;" />

## 📖 项目简介

本项目是 IEEE TIP 2024 论文 **DEA-Net** 的 **Jittor (计图)** 版本复现，属于 **南开大学“新芽计划”** 学习成果。

DEA-Net 提出了一种细节增强卷积（DEConv）和内容引导注意力（CGA）机制，能够有效恢复去雾图像。得益于 Jittor 框架的 **即时编译 (JIT)** 和 **算子融合** 技术，本项目在保持原论文精度的同时，实现了高效的训练与推理。

---

## 📰 News

- **[2025-12-21]** 🚀 Initial release of Jittor-DEA-Net code and pre-trained weights for HAZE4K, ITS, and OTS datasets.
- **[2025-11-17]** 🏗️ Project initialized under Nankai University "Sprouts Program".

---

## 📊 Model Zoo & Results (模型库与结果对比)

We provide a comparison between our Jittor implementation (Partial Training) and the official PyTorch implementation (Full Converged Training).

**Note:** The Jittor weights provided below are from the initial training phase (e.g., 10-30 epochs), yet they already demonstrate strong performance. The official PyTorch models were trained for 300 epochs.

**注意：** 下方提供的 Jittor 权重处于训练初期阶段（仅 10-30 Epoch），但已展现出优秀的性能。官方 PyTorch 模型为完整训练 300 Epoch 的结果。

| Dataset | Framework | Epochs Trained | PSNR (dB) | SSIM | Download Link |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **HAZE4K** | **Jittor (Ours)** | **30** (Partial) | 32.54 | 0.9848 | [Google Drive (Ours)](https://drive.google.com/drive/folders/1MN1alc4gBzk90Vc8V1AXivwx2FwrF5f3?usp=sharing) |
| | PyTorch (Official) | 300 | 34.26 | 0.9985 | [Google Drive](https://drive.google.com/drive/folders/1Rjb8dpyNnvvr0XLvIX9fg8Hdru_MhMCj?usp=sharing) / [Baidu (pwd:dcyb)](https://pan.baidu.com/s/1retfKIs_Om-D4zA45sL6Kg?pwd=dcyb) |
| **RESIDE-ITS** | **Jittor (Ours)** | **10** (Partial) | 35.87 | 0.9893 | [Google Drive (Ours)](https://drive.google.com/drive/folders/1MN1alc4gBzk90Vc8V1AXivwx2FwrF5f3?usp=sharing) |
| | PyTorch (Official) | 300 | 41.31 | 0.9945 | [Google Drive](https://drive.google.com/drive/folders/1Rjb8dpyNnvvr0XLvIX9fg8Hdru_MhMCj?usp=sharing) / [Baidu (pwd:dcyb)](https://pan.baidu.com/s/1retfKIs_Om-D4zA45sL6Kg?pwd=dcyb) |
| **RESIDE-OTS** | **Jittor (Ours)** | **10** (Partial) | 32.71 | 0.9840 | [Google Drive (Ours)](https://drive.google.com/drive/folders/1MN1alc4gBzk90Vc8V1AXivwx2FwrF5f3?usp=sharing) |
| | PyTorch (Official) | 300 | 36.59 | 0.9897 | [Google Drive](https://drive.google.com/drive/folders/1Rjb8dpyNnvvr0XLvIX9fg8Hdru_MhMCj?usp=sharing) / [Baidu (pwd:dcyb)](https://pan.baidu.com/s/1retfKIs_Om-D4zA45sL6Kg?pwd=dcyb) |

## Visual Results
> ![Outdoor Dehazing Results Comparison](assets/outdoor.jpg)
> ![Indoor Dehazing Results Comparison](assets/indoor.jpg)


**Note:**  
The image results, from top to bottom, represent the input, the inference result using the model pre-trained by the authors for 300 epochs, and the inference result using a partially trained model trained by Jittor. As shown in the figure, our trained model can definitely achieve the dehazing effect, but due to the limited number of training iterations and the use of a synthetic dataset, domain offset still causes artifacts that are visible to the naked eye.

**注意：**  
图片结果从上往下分别为输入、使用作者预训练300个 epoch 的模型推理结果，以及使用 Jittor 训练的部分模型推理结果。如图所示，我们训练的模型可以起到一定的去雾效果，但由于训练次数有限且使用的是合成数据集，域偏移仍会导致肉眼可见的伪影。

---

## ⚙️ Installation (安装指南)

### Prerequisites
- Linux (Ubuntu 20.04+ recommended)
- Python 3.8+
- NVIDIA GPU + CUDA

### Setup
1. **Clone the repository:**
    ```bash
    git clone [https://github.com/ZR-1N/Jittor-DEA-Net.git](https://github.com/ZR-1N/Jittor-DEA-Net.git)
    cd Jittor-DEA-Net
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Key dependencies: `jittor`, `numpy`, `Pillow`, `matplotlib`, `tqdm`.*

---

## 📂 Data Preparation (数据准备)

Please download the datasets and organize them strictly as follows.
请下载数据集并严格按照以下目录结构整理（代码将自动识别该结构）。

**Download Links:** [RESIDE (ITS/OTS)](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) | [HAZE4K](https://github.com/liuye123321/DMT-Net)

```text
Jittor-DEA-Net/
├── code/
├── dataset/
│   ├── HAZE4K/
│   │   ├── train/
│   │   │   ├── hazy/   (Contains .png/.jpg images)
│   │   │   └── clear/  (Contains .png/.jpg images)
│   │   └── test/
│   │       ├── hazy/
│   │       └── clear/
│   ├── ITS/
│   │   ├── train/ ... (Same structure as above)
│   │   └── test/  ...
│   └── OTS/
│   │   ├── train/ ... (Same structure as above)
│   │   └── test/  ...
└── ...
```

---

## 🔥 Training (训练)

We provide training scripts for different datasets. The code automatically handles `.png` and `.jpg` matching.
我们提供了针对不同数据集的训练脚本，代码已自动适配 `.png` 和 `.jpg` 的文件名匹配。

### 1. Train on HAZE4K
```bash
cd code
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
  --model_name DEA-Net-CR-HAZE4K \
  --dataset HAZE4K \
  --epochs 300 \
  --bs 4 \
  --w_loss_CR 0.1 \
  > training_haze4k.log 2>&1 &
```

### 2. Train on RESIDE-ITS (Indoor)
```bash
cd code
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
  --model_name DEA-Net-CR-ITS \
  --dataset ITS \
  --epochs 300 \
  --bs 4 \
  --w_loss_CR 0.1 \
  > training_its.log 2>&1 &
```

### 3. Train on RESIDE-OTS (Outdoor)
```bash
cd code
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
  --model_name DEA-Net-CR-OTS \
  --dataset OTS \
  --epochs 10 \
  --bs 4 \
  --w_loss_CR 0.1 \
  > training_ots.log 2>&1 &
```

*Training logs and checkpoints will be saved in `experiment/`.*

---

## 🖼️ Inference (推理)

Use `inference_raw.py` to dehaze your own images. The script automatically pads images to support arbitrary resolutions.
使用 `inference_raw.py` 对自定义图像进行去雾。脚本会自动对图像进行 Padding 以支持任意分辨率。

```bash
cd code
python3 inference_raw.py \
  --input_dir ../my_hazy_images \
  --output_dir ../my_results \
  --model_path ../experiment/HAZE4K/DEA-Net-CR-HAZE4K/saved_model/PSNR3254__SSIM9848.pk
```

---

## 🔗 Acknowledgements & Citation (致谢与引用)

This project is based on the official PyTorch implementation of [DEA-Net](https://github.com/cecret3350/DEA-Net). We thank the authors for their excellent work.

If you find this repository useful, please consider citing the original paper:

```bibtex
@article{chen2023dea,
  title={DEA-Net: Single image dehazing based on detail-enhanced convolution and content-guided attention},
  author={Chen, Zixuan and He, Zewei and Lu, Zhe-Ming},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  volume={33},
  pages={1002-1015}
}
```

## 📧 Contact

For any questions regarding this Jittor implementation, please contact:
**Shang Wenxuan (尚文轩)**: shangwenxuan.nku@gmail.com