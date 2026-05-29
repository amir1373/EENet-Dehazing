import os
from math import log10

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def psnr(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    return 20 * log10(1.0 / (mse**0.5)) if mse > 0 else 100.0


def ssim(img1, img2):
    from skimage.metrics import structural_similarity as ssim_fn

    img1 = img1.detach().squeeze().cpu().permute(1, 2, 0).numpy()
    img2 = img2.detach().squeeze().cpu().permute(1, 2, 0).numpy()
    return ssim_fn(img1, img2, channel_axis=-1, data_range=1.0)


def validate_model(model, val_loader, device=None):
    device = get_device(device)
    model.eval()
    psnr_total = 0.0
    ssim_total = 0.0
    count = 0

    with torch.no_grad():
        for hazy, gt, _ in val_loader:
            hazy = hazy.to(device)
            gt = gt.to(device)
            output = model(hazy)
            psnr_total += psnr(output, gt)
            ssim_total += ssim(output, gt)
            count += 1

    if count == 0:
        raise ValueError("Validation loader is empty.")
    return psnr_total / count, ssim_total / count


def train_model(
    model,
    train_loader,
    val_loader,
    save_path,
    epochs=60,
    lr=3e-5,
    resume=True,
    patience=8,
    plot_every=5,
    device=None,
):
    from tqdm import tqdm

    device = get_device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)
    os.makedirs(save_path, exist_ok=True)

    start_epoch = 1
    best_loss = float("inf")
    best_epoch = 0
    loss_history = []
    psnr_history = []
    ssim_history = []
    no_improve_counter = 0

    checkpoint_file = os.path.join(save_path, "latest.pth")
    if resume and os.path.exists(checkpoint_file):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        best_epoch = checkpoint.get("best_epoch", 0)
        loss_history = checkpoint.get("loss_history", [])
        psnr_history = checkpoint.get("psnr_history", [])
        ssim_history = checkpoint.get("ssim_history", [])

    t_max = max(1, epochs - (start_epoch - 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=100)

        for hazy, gt, _ in pbar:
            hazy = hazy.to(device)
            gt = gt.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                output = model(hazy)
                loss = criterion(output, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        val_psnr, val_ssim = validate_model(model, val_loader, device=device)
        psnr_history.append(val_psnr)
        ssim_history.append(val_ssim)

        print(
            f"Epoch {epoch} | Loss: {avg_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}"
        )
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            no_improve_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, "best_eenet.pth"))
        else:
            no_improve_counter += 1

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"eenet_epoch_{epoch}.pth"))

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "loss_history": loss_history,
                "psnr_history": psnr_history,
                "ssim_history": ssim_history,
            },
            checkpoint_file,
        )

        if no_improve_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

        if plot_every and (epoch % plot_every == 0 or epoch == epochs):
            plot_training_curves(loss_history, psnr_history, ssim_history, best_epoch)

    return model


def plot_training_curves(loss_history, psnr_history, ssim_history, best_epoch=0):
    import matplotlib.pyplot as plt

    plt.clf()
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))

    ax[0].plot(loss_history, label="Training Loss")
    ax[0].set_title("Loss Curve")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("L1 Loss")

    ax[1].plot(psnr_history, label="Validation PSNR", color="green")
    ax[1].set_title("Validation PSNR")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("PSNR (dB)")

    ax[2].plot(ssim_history, label="Validation SSIM", color="blue")
    ax[2].set_title("Validation SSIM")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("SSIM")

    for axis in ax:
        if best_epoch:
            axis.axvline(best_epoch, color="red", linestyle="--", label="Best")
        axis.legend()
        axis.grid(True)

    plt.tight_layout()
    plt.pause(0.01)
    plt.show()
