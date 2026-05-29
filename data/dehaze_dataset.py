import os

from PIL import Image
from torch.utils.data import Dataset


class DehazeDataset(Dataset):
    def __init__(self, root_dir, mode="train", transform=None):
        self.hazy_dir = os.path.join(root_dir, mode, "hazy")
        self.gt_dir = os.path.join(root_dir, mode, "gt") if mode != "test" else None
        self.transform = transform
        self.mode = mode

        if not os.path.exists(self.hazy_dir):
            raise FileNotFoundError(f"Hazy folder not found: {self.hazy_dir}")

        self.images = sorted(
            f for f in os.listdir(self.hazy_dir) if not f.startswith(".")
        )

        if self.gt_dir:
            if not os.path.exists(self.gt_dir):
                raise FileNotFoundError(f"GT folder not found: {self.gt_dir}")

            gt_images = sorted(
                f for f in os.listdir(self.gt_dir) if not f.startswith(".")
            )
            gt_set = set(gt_images)
            self.paired_images = [(h, h) for h in self.images if h in gt_set]
        else:
            self.paired_images = [(img, None) for img in self.images]

        if not self.paired_images:
            raise ValueError(f"No image pairs found for mode '{mode}' in {root_dir}")

    def __len__(self):
        return len(self.paired_images)

    def __getitem__(self, idx):
        hazy_name, gt_name = self.paired_images[idx]
        hazy = Image.open(os.path.join(self.hazy_dir, hazy_name)).convert("RGB")

        if self.transform:
            hazy = self.transform(hazy)

        if self.mode == "test":
            return hazy, hazy_name

        gt = Image.open(os.path.join(self.gt_dir, gt_name)).convert("RGB")
        if self.transform:
            gt = self.transform(gt)
        return hazy, gt, hazy_name
