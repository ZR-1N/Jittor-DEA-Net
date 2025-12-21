<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/Nankai_University_logo.svg" height="80px" alt="Nankai University" >
<img src="https://raw.githubusercontent.com/Jittor/jittor/master/docs/images/logo.png" height="80px" alt="Jittor" >

# Jittor-DEA-Net

**DEA-Net: Single image dehazing based on detail-enhanced convolution and content-guided attention (IEEE TIP 2024)**

[![Jittor](https://img.shields.io/badge/Framework-Jittor-EA3323.svg)]([https://cg.cs.tsinghua.edu.cn/jittor/](https://cg.cs.tsinghua.edu.cn/jittor/))
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/ZR-1N/Jittor-DEA-Net?style=social)](https://github.com/ZR-1N/Jittor-DEA-Net)

[English](#-introduction) | [简体中文](#-项目简介)

</div>

---

## 📖 Introduction

This repository is an official implementation of **DEA-Net** based on the [Jittor (计图)](https://cg.cs.tsinghua.edu.cn/jittor/) deep learning framework. This project is part of the **"Sprouts Program" at Nankai University**.

DEA-Net proposes a novel detail-enhanced convolution (DEConv) and content-guided attention (CGA) mechanism to effectively restore haze-free images. By leveraging Jittor's **Just-In-Time (JIT) compilation** and **operator fusion**, this implementation achieves competitive training efficiency compared to the PyTorch version while maintaining algorithmic performance.

## 📖 项目简介

本项目是 IEEE TIP 2024 论文 **DEA-Net** 的 **Jittor (计图)** 版本复现，属于 **南开大学“新芽计划”** 研究成果。

DEA-Net 提出了一种细节增强卷积（DEConv）和内容引导注意力（CGA）机制，能够有效恢复去雾图像。得益于 Jittor 框架的 **即时编译 (JIT)** 和 **算子融合** 技术，本项目在保持原论文精度的同时，实现了高效的训练与推理。

---

## 📊 Model Zoo & Results (模型库与结果)

We provide pre-trained models on three mainstream dehazing datasets.
**Note:** The current weights are from the initial training phase (partial epochs), yet they already demonstrate strong performance.
**注意：** 当前提供的权重处于训练初期阶段（部分 Epoch），但已展现出优秀的性能。

| Dataset | Training Progress | PSNR (dB) | SSIM | Download |
| :--- | :---: | :---: | :---: | :---: |
| **HAZE4K** | 30 Epochs (Partial) | **32.54** | **0.9848** | [Google Drive](#) / [Baidu Netdisk](#) |
| **RESIDE-ITS** | 10 Epochs (Partial) | **35.87** | **0.9893** | [Google Drive](#) / [Baidu Netdisk](#) |
| **RESIDE-OTS** | 10 Epochs (Partial) | **32.71** | **0.9840** | [Google Drive](#) / [Baidu Netdisk](#) |

> *Visual results placeholder*
> ![Results Placeholder](https://via.placeholder.com/800x400?text=Dehazing+Results+Comparison)

---

## ⚙️ Installation (安装指南)

### Prerequisites
- Linux (Ubuntu 20.04+ recommended)
- Python 3.8+
- NVIDIA GPU + CUDA

### Setup
1. **Clone the repository:**
    ```bash
    git clone https://github.com/ZR-1N/Jittor-DEA-Net.git
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

**Download Links:** [RESIDE (ITS/OTS)](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)) | [HAZE4K](https://github.com/liuye123321/DMT-Net))

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
│       ├── train/ ... (Same structure as above)
│       └── test/  ...
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
  --model_path ../experiment/HAZE4K/DEA-Net-CR-HAZE4K/saved_model/best.pk
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
