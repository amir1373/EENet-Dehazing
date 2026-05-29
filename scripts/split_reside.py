import argparse
import os
import random
import shutil


def split_reside(src_train_hazy, src_train_gt, output_root, val_ratio=0.1, seed=42):
    train_hazy = os.path.join(output_root, "train", "hazy")
    train_gt = os.path.join(output_root, "train", "gt")
    val_hazy = os.path.join(output_root, "val", "hazy")
    val_gt = os.path.join(output_root, "val", "gt")

    for path in [train_hazy, train_gt, val_hazy, val_gt]:
        os.makedirs(path, exist_ok=True)

    all_files = sorted(f for f in os.listdir(src_train_hazy) if not f.startswith("."))
    random.seed(seed)
    random.shuffle(all_files)

    split_idx = int((1.0 - val_ratio) * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    for fname in train_files:
        shutil.copy(os.path.join(src_train_hazy, fname), os.path.join(train_hazy, fname))
        shutil.copy(os.path.join(src_train_gt, fname), os.path.join(train_gt, fname))

    for fname in val_files:
        shutil.copy(os.path.join(src_train_hazy, fname), os.path.join(val_hazy, fname))
        shutil.copy(os.path.join(src_train_gt, fname), os.path.join(val_gt, fname))

    return len(train_files), len(val_files)


def main():
    parser = argparse.ArgumentParser(description="Split RESIDE-6K train data into train/val folders.")
    parser.add_argument("--src-train-hazy", required=True, help="Path to RESIDE train/hazy folder.")
    parser.add_argument("--src-train-gt", required=True, help="Path to RESIDE train/GT folder.")
    parser.add_argument("--output-root", required=True, help="Output root containing train/ and val/.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_count, val_count = split_reside(
        args.src_train_hazy,
        args.src_train_gt,
        args.output_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Dataset split complete: {train_count} train / {val_count} val images.")


if __name__ == "__main__":
    main()
