# EENet-Dehazing

This repository contains a PyTorch implementation of **EENet**, a dual-domain network that integrates frequency-aware and spatial multiscale features for single image dehazing. The model is trained on RESIDE-6K and fine-tuned on the RB-Dust industrial dataset.

---

## 🧠 Model Overview

EENet leverages:
- **Frequency Processing Modules (FPM)** using FFT-based convolution
- **Spatial Processing Modules (SPM)** with residual CNNs
- **Dual-Domain Interaction Modules (DIM)** to fuse frequency and spatial features
- A U-Net-style encoder-decoder structure with skip connections

---

## 📦 Pretrained Model

You can download the pretrained model directly from Kaggle using the following command:

```bash
kaggle models instances versions download moshtaghioun/eenet/pyTorch/default/1
```

Unzip the file and place `best_eenet.pth` in your working directory.

---

## 🚀 Usage

```python
import torch
from model.eenet import EENet

model = EENet()
model.load_state_dict(torch.load("best_eenet.pth", map_location="cuda"))
model.eval()

# Input: Tensor of shape [B, 3, 256, 256] normalized to [0, 1]
# Output: Tensor of shape [B, 3, 256, 256] (dehazed output)
```

For full training, evaluation, and visualization scripts, see the [notebooks](notebooks/) and [scripts](scripts/) directories.

---

## 📊 Evaluation Results

| Dataset     | PSNR ↑ | SSIM ↑  |
|-------------|--------|---------|
| RESIDE-6K   | 21.45  | 0.81    |
| RB-Dust     | 24.72  | 0.7015  |

---

## 🧪 Datasets

- [RESIDE-6K](https://github.com/nttcslab/RESIDE) — synthetic outdoor haze dataset
- **RB-Dust** — real-world industrial dust dataset (private, used for fine-tuning)

---

## 📑 Reference

This work is based on the following paper:

> **"EENet: Frequency-Aware and Spatially Multiscale Network for Single Image Dehazing"**  
> *Shuang Xu, Ruichen Zhao, Bingchen Zhao, Yinqiang Zheng, Kun Zhou*  
> Pattern Recognition, 2024.  
> [DOI: 10.1016/j.patcog.2024.111074](https://doi.org/10.1016/j.patcog.2024.111074)

---

## 🪪 License

This implementation is released under the MIT License.  
Feel free to use, modify, and distribute — with credit.

---

## 🙋 Contact

For questions or contributions, feel free to open an issue or reach out via [https://roboticswith.me](https://roboticswith.me)
