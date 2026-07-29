from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "submission" / "figures"
V4 = ROOT / "report" / "v4" / "figures"
METRICS = ROOT / "report" / "v4" / "tables" / "pretrained_p000.csv"

NAVY = "#0B2545"
BLUE = "#2E74B5"
GOLD = "#A67118"


def copy_core_figures():
    for name in (
        "vggt_architecture_corrected.png",
        "experimental_ladder.png",
        "tartanair_nested_subsets.png",
        "eth3d_overlap_pair_comparison.png",
        "p000_absrel.png",
        "p000_confidence.png",
        "adaptation_validation.png",
        "pretrained_vs_adapted.png",
    ):
        shutil.copy2(V4 / name, OUT / name)


def representative_rgb(contact_sheet):
    image = Image.open(contact_sheet).convert("RGB")
    return image.crop((0, int(image.height * 0.12), int(image.width * 0.32), image.height))


def dark_point_cloud(path):
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).copy()
    white = np.min(arr, axis=2) > 238
    arr[white] = np.array([17, 25, 35], dtype=np.uint8)
    # Increase visibility of non-background splats without changing geometry.
    fg = ~white
    arr[fg] = np.clip(arr[fg].astype(np.float32) * 1.18 + 8, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def scene_page(scene, folder):
    sources = [
        representative_rgb(folder / "contact_sheet.jpg"),
        Image.open(folder / "depth_view0.png").convert("RGB"),
        Image.open(folder / "point_confidence_view0.png").convert("RGB"),
        dark_point_cloud(folder / "point_cloud_confidence_filtered_preview.png"),
    ]
    labels = ["Representative RGB", "Predicted depth", "Point confidence", "Confidence-filtered point cloud"]
    panel = (770, 430)
    canvas = Image.new("RGB", (1580, 970), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (source, label) in enumerate(zip(sources, labels)):
        x = 10 + (index % 2) * 790
        y = 55 + (index // 2) * 460
        image = ImageOps.contain(source, panel, Image.Resampling.LANCZOS)
        background = Image.new("RGB", panel, (17, 25, 35) if index == 3 else "white")
        background.paste(image, ((panel[0] - image.width) // 2, (panel[1] - image.height) // 2))
        canvas.paste(background, (x, y))
        draw.text((x + 8, y - 31), label, fill=NAVY)
    canvas.save(OUT / f"eth3d_{scene}_evidence.png", quality=96)


def runtime_memory():
    with METRICS.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    views = [int(row["views"]) for row in rows]
    runtime = [float(row["inference_seconds"]) for row in rows]
    memory = [float(row["peak_allocated_gib"]) for row in rows]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(8.4, 4.2))
    ax2 = ax1.twinx()
    ax1.plot(views, runtime, "o-", color=BLUE, lw=2.5, label="Inference time")
    ax2.plot(views, memory, "s--", color=GOLD, lw=2.5, label="Peak allocated VRAM")
    ax1.set_xlabel("Input views")
    ax1.set_ylabel("Single recorded inference (s)", color=BLUE)
    ax2.set_ylabel("Peak allocated VRAM (GiB)", color=GOLD)
    lines = ax1.lines + ax2.lines
    ax1.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "runtime_memory.png", dpi=220)
    plt.close(fig)


def equation_image(name, expression, width=8.0, fontsize=24):
    fig, ax = plt.subplots(figsize=(width, 0.72))
    ax.axis("off")
    ax.text(0.5, 0.5, f"${expression}$", ha="center", va="center", fontsize=fontsize, color=NAVY)
    fig.savefig(OUT / f"eq_{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.08, transparent=True)
    plt.close(fig)


def equations():
    equation_image("camera_mapping", r"x_c = R x_w + t")
    equation_image("camera_center", r"C = -R^{\mathsf{T}}t")
    equation_image("sim3", r"\widehat{C}_i = sQ C_i + b")
    equation_image("rotation_gauge", r"A=R^{gt}_0(R^{pred}_0)^{\mathsf{T}}")
    equation_image("rotation_error", r"\Delta R_i=(R^{gt}_i)^{\mathsf{T}} A R^{pred}_i,\quad e_{R,i}=\cos^{-1}\!\left(\mathrm{clip}\!\left(\frac{\mathrm{tr}(\Delta R_i)-1}{2},-1,1\right)\right)", 10.5, 19)
    equation_image("depth_scale", r"\alpha=\frac{\mathrm{median}(d^{gt})}{\mathrm{median}(d^{pred})},\qquad \widehat d=\alpha d^{pred}")
    equation_image("absrel", r"\mathrm{AbsRel}=\frac{1}{N}\sum_i\frac{|\widehat d_i-d_i|}{d_i}")
    equation_image("rmse", r"\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_i(\widehat d_i-d_i)^2}")
    equation_image("delta", r"\delta_{1.25}=\frac{1}{N}\sum_i\mathbf{1}\!\left[\max\!\left(\frac{\widehat d_i}{d_i},\frac{d_i}{\widehat d_i}\right)<1.25\right]")
    equation_image("spearman", r"\rho\!\left(\mathrm{confidence},\,|\widehat d-d|\right)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    copy_core_figures()
    scene_page(
        "delivery_area",
        ROOT / "outputs/predictions/phase7_eth3d_view_count/delivery_area/S10_overlap_aware_nested_original/visualizations",
    )
    scene_page(
        "courtyard",
        ROOT / "outputs/predictions/phase8_eth3d_view_count/courtyard/S10_overlap_aware_nested_original/visualizations",
    )
    runtime_memory()
    equations()
    print(f"Prepared {len(list(OUT.glob('*.png')))} submission assets")


if __name__ == "__main__":
    main()
