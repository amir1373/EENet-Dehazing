import argparse
import os
import sys

from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dehaze_dataset import DehazeDataset
from data.rbdust_dataset import RBDustDataset
from model.eenet import EENet
from utils.train_utils import train_model


def build_datasets(args, transform):
    if args.dataset == "reside":
        train_dataset = DehazeDataset(args.data_root, mode="train", transform=transform)
        val_dataset = DehazeDataset(args.data_root, mode="val", transform=transform)
    elif args.dataset == "rbdust":
        train_dataset = RBDustDataset(
            args.data_root,
            mode="train",
            split_ratio=args.split_ratio,
            transform=transform,
            seed=args.seed,
        )
        val_dataset = RBDustDataset(
            args.data_root,
            mode="val",
            split_ratio=args.split_ratio,
            transform=transform,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Train EENet for image dehazing.")
    parser.add_argument("--data-root", required=True, help="Dataset root path.")
    parser.add_argument("--save-path", default="checkpoints", help="Directory for checkpoints.")
    parser.add_argument("--dataset", choices=["reside", "rbdust"], default="reside")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    train_dataset, val_dataset = build_datasets(args, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    model = EENet()
    train_model(
        model,
        train_loader,
        val_loader,
        args.save_path,
        epochs=args.epochs,
        lr=args.lr,
        resume=not args.no_resume,
        patience=args.patience,
        plot_every=args.plot_every,
        device=args.device,
    )


if __name__ == "__main__":
    main()
