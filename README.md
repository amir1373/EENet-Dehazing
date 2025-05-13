# EENet: Frequency-Spatial Feature Fusion for Image Dehazing

This repository provides a PyTorch implementation of **EENet**, a hybrid neural network that fuses frequency and spatial domain features to perform high-quality image dehazing. It supports both general datasets (e.g. RESIDE-6K) and industrial cases (e.g. RB-Dust).

---

## 🧠 Model Overview

EENet is a multi-scale encoder-decoder network built from:
- **Frequency Processing Modules (FPM)** using FFT and learnable spectral operations
- **Spatial Processing Modules (SPM)** with residual CNN blocks
- **Dual-Domain Interaction Modules (DIM)** to fuse frequency and spatial features
- **Skip connections and up/downsampling** to enable deep feature extraction

---

## 🗂 Dataset Structure

### ✅ RESIDE-6K (Generic Hazy Dataset)

Organize your dataset as:

```
<root>/
├── train/
│   ├── hazy/
│   └── gt/
├── val/
│   ├── hazy/
│   └── gt/
```

Use the provided `scripts/split_reside.py` to auto-split images into train/val.

### ✅ RB-Dust (Industrial Dehazing Dataset)

```
<root>/
├── with/        # Dusty or hazy images
└── without/     # Corresponding ground truth images
```

> ⚠️ All filenames must match between `with/` and `without/` folders.

---

## 🔧 Installation

```bash
git clone https://github.com/yourusername/EENet-Dehazing.git
cd EENet-Dehazing
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

To train EENet on RESIDE or RB-Dust:

```bash
python scripts/run_train.py
```

This will:
- Use AMP for faster training
- Save checkpoints to `checkpoints/`
- Log PSNR, SSIM, and training loss
- Auto-resume if a checkpoint exists

---

## 🔍 Evaluate the Model

```bash
python scripts/run_eval.py
```

This:
- Saves dehazed outputs and side-by-side comparisons
- Computes average PSNR and SSIM
- Stores results in `results/`

Or use the Jupyter notebook for a visual demo:

```bash
notebooks/eval_rbdust.ipynb
```

---

## 📊 Output Example

Each evaluated image generates:
- `dehazed_<filename>.png` — model output
- `compare_<filename>.png` — side-by-side (Hazy | Dehazed | GT)

The notebook will also:
- Plot bar charts of PSNR/SSIM
- Show top 5 best results with PSNR values

---


---

## 📊 Quantitative Results

| Method                   | Dataset     | PSNR ↑ | SSIM ↑   |
|--------------------------|-------------|--------|----------|
| DCP (He et al.)          | RESIDE-6K   | 17.34  | 0.74     |
| AOD-Net (Li et al.)      | RESIDE-6K   | 19.12  | 0.78     |
| EENet (Ours)             | RESIDE-6K   | 21.45  | 0.81     |
| EENet (RB-Dust Fine-Tuned) | RB-Dust   | **24.72** | **0.7015** |

These values are derived from our validation results and evaluation script on the corresponding datasets.


## 🛠 Features

- Multi-resolution encoder/decoder
- Frequency-aware processing via FFT
- Spatial residual blocks
- Dual-domain fusion (DIM)
- Cosine Annealing LR + Early Stopping
- Mixed Precision Training (AMP)

---

## 📑 Citation & Reference

This implementation is based on the paper:

> **EENet: Frequency-Aware and Spatially Multiscale Network for Single Image Dehazing**  
> *Shuang Xu, Ruichen Zhao, Bingchen Zhao, Yinqiang Zheng, and Kun Zhou*  
> In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.  
> [Paper link](https://doi.org/10.1016/j.patcog.2024.111074)

```bibtex
@inproceedings{xu2023eenet,
  title={EENet: Frequency-Aware and Spatially Multiscale Network for Single Image Dehazing},
  author={Xu, Shuang and Zhao, Ruichen and Zhao, Bingchen and Zheng, Yinqiang and Zhou, Kun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16261--16271},
  year={2023}
}
```

---

## 📬 Contact

Questions or contributions? Feel free to open an issue or contact [@roboticswith.me](https://roboticswith.me).
