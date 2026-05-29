import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dehaze_dataset import DehazeDataset
from data.rbdust_dataset import RBDustDataset
from model.eenet import EENet
from utils.eval_utils import test_model
from utils.train_utils import get_device


def build_dataset(args, transform):
    if args.dataset == "reside":
        return DehazeDataset(args.data_root, mode=args.mode, transform=transform)
    if args.dataset == "rbdust":
        return RBDustDataset(
            args.data_root,
            mode=args.mode,
            split_ratio=args.split_ratio,
            transform=transform,
            seed=args.seed,
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate EENet for image dehazing.")
    parser.add_argument("--data-root", required=True, help="Dataset root path.")
    parser.add_argument("--checkpoint", required=True, help="Path to model state dict.")
    parser.add_argument("--save-dir", default="dehaze_results", help="Output directory.")
    parser.add_argument("--dataset", choices=["reside", "rbdust"], default="reside")
    parser.add_argument("--mode", choices=["train", "val", "test"], default="test")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    device = get_device(args.device)
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )

    dataset = build_dataset(args, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = EENet()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    test_model(
        model,
        dataloader,
        save_dir=args.save_dir,
        log_csv=not args.no_csv,
        device=device,
    )


if __name__ == "__main__":
    main()
