# EENet-Dehazing

![PyTorch](https://img.shields.io/badge/framework-pytorch-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

This repository contains a PyTorch implementation of **EENet**, a dual-domain network that integrates frequency-aware and spatial multiscale features for single image dehazing. The model is trained on RESIDE-6K and fine-tuned on the RB-Dust agricultural dust dataset.

This is a simplified implementation of the published design, written from the paper's description around March 2025, without the authors' code, and published here in May 2025. It is separate from the authors' official code, which is available at [github.com/c-yn/EENet](https://github.com/c-yn/EENet).

This implementation was listed on Papers with Code under Image Dehazing on RB-Dust. Papers with Code has since been discontinued, so that listing and its badge are no longer available.

---

## 🧠 Model Overview

EENet leverages:

- **Frequency Processing Modules (FPM)** using FFT-based convolution
- **Spatial Processing Modules (SPM)** with residual CNNs
- **Dual-Domain Interaction Modules (DIM)** to fuse frequency and spatial features
- A U-Net-style encoder-decoder structure with skip connections

---

## 🧬 Model Architecture

Below is a schematic diagram of the EENet model architecture, showing its dual-domain structure with frequency and spatial processing modules.

![EENet Architecture](assets/EENet_architecture.png)

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

For full training, evaluation, and visualization scripts, see the [scripts](scripts/) directory.

---

## 📊 Evaluation Results

| Dataset   | PSNR ↑ | SSIM ↑ |
|-----------|--------|--------|
| RESIDE-6K | 21.45  | 0.81   |
| RB-Dust   | 24.72  | 0.7015 |

---

## 🖼 Sample Results

Below are composite visualizations showing dusty input images, EENet outputs, and ground truth side-by-side.

![Sample Results 1](results/Sample1.png)
![Sample Results 2](results/Sample2.png)
![Sample Results 3](results/Sample3.png)
![Sample Results 4](results/Sample4.png)
![Sample Results 5](results/Sample5.png)
![Sample Results 6](results/Sample6.png)

---

## 🧪 Datasets

- [RESIDE-6K](https://github.com/nttcslab/RESIDE) — synthetic outdoor haze dataset
- **RB-Dust** — real-world agricultural dust dataset: P. Buckel, T. Oksanen, and T. Dietmueller, "RB-Dust – A reference-based dataset for vision-based dust removal," in *Proc. IEEE/CVF CVPR Workshops*, 2023, pp. 1140–1149, doi: [10.1109/CVPRW59228.2023.00121](https://doi.org/10.1109/CVPRW59228.2023.00121)

---

## 📑 Reference

This work is based on the following paper:

> **"EENet: An effective and efficient network for single image dehazing"**<br>
> *Yuning Cui, Qiang Wang, Chaopeng Li, Wenqi Ren, Alois Knoll*<br>
> Pattern Recognition, vol. 158, art. no. 111074, 2025.<br>
> [DOI: 10.1016/j.patcog.2024.111074](https://doi.org/10.1016/j.patcog.2024.111074)

---

## 👥 Contributors

- **Seyed Amirhossein Moshtaghioun**<br>
  🔗 [GitHub](https://github.com/amir1373) · 🌐 [Website](https://roboticswith.me)

- **Dr. Mehran Mehrandezh**<br>
  🏫 University of Regina · 📧 Mehran.Mehrandezh@uregina.ca

- **Dr. Vali Dehrami**

> Special thanks to the authors of the original EENet paper.

---

## 🪪 License

This implementation is released under the MIT License.<br>
Feel free to use, modify, and distribute — with credit.

---

## 🙋 Contact

For questions or contributions, feel free to open an issue or reach out via [https://roboticswith.me](https://roboticswith.me).
