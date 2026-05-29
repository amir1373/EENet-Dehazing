import csv
import os

import torch
from torchvision.utils import save_image

from utils.train_utils import get_device, psnr, ssim


def _first_filename(fname, fallback):
    if isinstance(fname, (list, tuple)):
        return fname[0]
    return fname if isinstance(fname, str) else str(fallback)


def test_model(model, dataloader, save_dir="dehaze_results", log_csv=True, device=None):
    device = get_device(device)
    model = model.to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    comparisons_dir = os.path.join(save_dir, "comparisons")
    os.makedirs(comparisons_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "metrics_log.csv") if log_csv else None
    html_path = os.path.join(save_dir, "report.html")

    if log_csv:
        with open(log_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Filename", "PSNR", "SSIM"])

    html_lines = [
        "<html><head><title>Dehaze Evaluation Report</title></head><body>",
        "<h1>Dehazing Model Results</h1>",
        "<table border='1'><tr><th>Image</th><th>Comparison</th><th>PSNR</th><th>SSIM</th></tr>",
    ]

    psnr_total = 0.0
    ssim_total = 0.0
    count = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if len(data) == 3:
                hazy, gt, fname = data
                has_gt = True
            else:
                hazy, fname = data
                gt = None
                has_gt = False

            hazy = hazy.to(device)
            output = model(hazy).clamp(0, 1)

            fname_str = _first_filename(fname, i)
            out_name = f"dehazed_{fname_str}"
            cmp_name = f"compare_{fname_str}"
            out_path = os.path.join(save_dir, out_name)
            cmp_path = os.path.join(comparisons_dir, cmp_name)
            save_image(output, out_path)

            psnr_val = "-"
            ssim_val = "-"
            if has_gt:
                gt = gt.to(device)
                comparison = torch.cat([hazy, output, gt], dim=-1)
                save_image(comparison, cmp_path)

                psnr_val = psnr(output, gt)
                ssim_val = ssim(output, gt)
                psnr_total += psnr_val
                ssim_total += ssim_val
                count += 1

                if log_csv:
                    with open(log_path, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([fname_str, f"{psnr_val:.2f}", f"{ssim_val:.4f}"])

            comparison_cell = (
                f"<img src='comparisons/{cmp_name}' width='768'>"
                if has_gt
                else "No ground truth"
            )
            psnr_cell = psnr_val if psnr_val == "-" else f"{psnr_val:.2f} dB"
            ssim_cell = ssim_val if ssim_val == "-" else f"{ssim_val:.4f}"
            html_lines.append(
                f"<tr><td><img src='{out_name}' width='256'></td>"
                f"<td>{comparison_cell}</td><td>{psnr_cell}</td><td>{ssim_cell}</td></tr>"
            )

    html_lines.append("</table></body></html>")
    with open(html_path, "w") as file:
        file.write("\n".join(html_lines))

    if count > 0:
        avg_psnr = psnr_total / count
        avg_ssim = ssim_total / count
        print("Evaluation complete")
        print(f"Avg PSNR: {avg_psnr:.2f} dB")
        print(f"Avg SSIM: {avg_ssim:.4f}")
        print(f"HTML report saved at: {html_path}")
        return {"psnr": avg_psnr, "ssim": avg_ssim, "report": html_path}

    print("Test mode only: no ground truth available, skipping metrics.")
    print(f"HTML report saved at: {html_path}")
    return {"psnr": None, "ssim": None, "report": html_path}
