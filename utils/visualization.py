import matplotlib.pyplot as plt


def show_batch(loader):
    batch = next(iter(loader))

    if len(batch) == 3:
        images, targets, _ = batch
    elif len(batch) == 2:
        images, targets = batch
    else:
        images = batch
        targets = None

    cols = len(images)
    fig, axs = plt.subplots(2, cols, figsize=(15, 5), squeeze=False)
    for i in range(cols):
        axs[0, i].imshow(images[i].permute(1, 2, 0).cpu())
        axs[0, i].set_title("Hazy")
        axs[0, i].axis("off")

        if targets is not None:
            axs[1, i].imshow(targets[i].permute(1, 2, 0).cpu())
            axs[1, i].set_title("Ground Truth")
        axs[1, i].axis("off")

    plt.tight_layout()
    plt.show()
