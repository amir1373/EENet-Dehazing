import os
import random

from PIL import Image
from torch.utils.data import Dataset


class RBDustDataset(Dataset):
    """
    RB-Dust dataset loader.

    Expected directory layout:
        root/
          with/
          without/
    """

    def __init__(self, root_dir, mode="train", split_ratio=0.8, transform=None, seed=42):
        self.with_dir = os.path.join(root_dir, "with")
        self.without_dir = os.path.join(root_dir, "without")
        self.transform = transform
        self.mode = mode

        if not os.path.exists(self.with_dir) or not os.path.exists(self.without_dir):
            raise FileNotFoundError(
                "Both 'with' and 'without' folders must exist in the root directory."
            )

        all_files = sorted(
            f
            for f in os.listdir(self.with_dir)
            if not f.startswith(".") and os.path.exists(os.path.join(self.without_dir, f))
        )
        if not all_files:
            raise ValueError("No matching image pairs found in 'with' and 'without'.")

        random.seed(seed)
        random.shuffle(all_files)
        split_idx = int(split_ratio * len(all_files))

        if mode == "train":
            self.filenames = all_files[:split_idx]
        elif mode == "val":
            self.filenames = all_files[split_idx:]
        elif mode == "test":
            self.filenames = all_files
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        hazy = Image.open(os.path.join(self.with_dir, fname)).convert("RGB")
        gt = Image.open(os.path.join(self.without_dir, fname)).convert("RGB")

        if self.transform:
            hazy = self.transform(hazy)
            gt = self.transform(gt)

        return hazy, gt, fname
