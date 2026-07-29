from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/v4/figures"
BASE = ROOT / "outputs/experiments/v3_tartanair_pretrained_20260729_sdpa_corrected/metrics.csv"
ADAPTED = ROOT / "outputs/experiments/v3_tartanair_adapted_step15_20260729/metrics.csv"
TRAIN = ROOT / "outputs/experiments/v3_tartanair_finetune_20260729/history.csv"
AUDIT = ROOT / "outputs/analysis/v3_tartanair_audit_20260729/selected_contact_sheet.jpg"

NAVY = "#0B2545"
BLUE = "#2E74B5"
CYAN = "#DDF3F5"
GOLD = "#A67118"
RED = "#9B1C1C"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def flow_figure(labels, filename, title=None):
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.text(0.5, 0.975, title, ha="center", va="top", fontsize=15, weight="bold", color=NAVY)
    top, gap, height = 0.87, 0.135, 0.082
    for index, label in enumerate(labels):
        y = top - index * gap
        color = CYAN if index < len(labels) - 1 else "#E8EEF5"
        patch = FancyBboxPatch(
            (0.13, y - height / 2),
            0.74,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            edgecolor=BLUE,
            facecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(patch)
        ax.text(0.5, y, label, ha="center", va="center", fontsize=11, weight="bold", color=NAVY)
        if index < len(labels) - 1:
            ax.annotate("", xy=(0.5, y - gap + height / 2 + 0.01), xytext=(0.5, y - height / 2 - 0.01),
                        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.8))
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def crop_input(path):
    image = Image.open(path).convert("RGB")
    # Contact sheets have a title band; take a representative left segment below it.
    return image.crop((0, int(image.height * 0.12), int(image.width * 0.32), image.height))


def tile(image, size=(430, 270)):
    return ImageOps.fit(image.convert("RGB"), size, Image.Resampling.LANCZOS)


def eth3d_figure():
    scenes = {
        "delivery_area": ROOT / "outputs/predictions/phase7_eth3d_view_count/delivery_area/S10_overlap_aware_nested_original/visualizations",
        "courtyard": ROOT / "outputs/predictions/phase8_eth3d_view_count/courtyard/S10_overlap_aware_nested_original/visualizations",
    }
    labels = ["Representative RGB", "Predicted depth", "Point confidence", "Filtered point cloud"]
    width, height = 430, 270
    header, row_gap, label_h = 62, 28, 32
    canvas = Image.new("RGB", (width * 4, header + (height + label_h) * 2 + row_gap), "white")
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(labels):
        draw.text((col * width + 12, 18), label, fill=NAVY)
    for row, (scene, folder) in enumerate(scenes.items()):
        images = [
            crop_input(folder / "contact_sheet.jpg"),
            Image.open(folder / "depth_view0.png"),
            Image.open(folder / "point_confidence_view0.png"),
            Image.open(folder / "point_cloud_confidence_filtered_preview.png"),
        ]
        y = header + row * (height + label_h + row_gap)
        for col, image in enumerate(images):
            canvas.paste(tile(image, (width, height)), (col * width, y))
        draw.rectangle((0, y + height, canvas.width, y + height + label_h), fill="#E8EEF5")
        draw.text((14, y + height + 7), f"{scene} | S10 overlap-aware nested inputs", fill=NAVY)
    canvas.save(OUT / "eth3d_two_scene_results.png", quality=95)


def overlap_figure():
    poor = ROOT / "outputs/predictions/phase6_eth3d_smoke/delivery_area/2_views_evenly_spaced_original/visualizations/contact_sheet.jpg"
    good = ROOT / "outputs/predictions/phase6_2_eth3d_overlap_smoke/delivery_area/S2_overlap_aware_nested_original/visualizations/contact_sheet.jpg"
    canvas = Image.new("RGB", (1600, 620), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 18), "Endpoint pair [0, 43] - limited shared visibility", fill=RED)
    draw.text((820, 18), "Overlap-aware pair [0, 6] - stronger shared evidence", fill=NAVY)
    canvas.paste(tile(Image.open(poor), (760, 540)), (20, 62))
    canvas.paste(tile(Image.open(good), (760, 540)), (820, 62))
    canvas.save(OUT / "eth3d_overlap_pair_comparison.png", quality=95)


def nested_subsets():
    sheet = Image.open(AUDIT).convert("RGB")
    # Extract 5x2 tiles from the known contact-sheet layout.
    tile_w, tile_h = sheet.width // 5, sheet.height // 2
    frames = []
    for index in range(10):
        x, y = (index % 5) * tile_w, (index // 5) * tile_h
        frames.append(ImageOps.fit(sheet.crop((x, y, x + tile_w, y + tile_h)), (150, 150)))
    left = 140
    canvas = Image.new("RGB", (left + 1500, 150 + 5 * 46 + 35), "white")
    draw = ImageDraw.Draw(canvas)
    for i, frame in enumerate(frames):
        canvas.paste(frame, (left + i * 150, 0))
        draw.text((left + i * 150 + 48, 154), str(20 + i), fill=NAVY)
    colors = ["#BFD7EA", "#9BC1E2", "#69A7D4", "#3F88C5", "#0B5FA5"]
    for row, count in enumerate((2, 4, 6, 8, 10)):
        y = 185 + row * 46
        draw.text((30, y + 9), f"S{count}", fill=NAVY)
        for i in range(10):
            fill = colors[row] if i < count else "#F2F4F7"
            draw.rounded_rectangle((left + i * 150 + 6, y, left + (i + 1) * 150 - 6, y + 34),
                                   radius=7, fill=fill, outline="#D0D6DC")
    canvas.save(OUT / "tartanair_nested_subsets.png", quality=95)


def charts():
    base, adapted = read_csv(BASE), read_csv(ADAPTED)
    views = [int(row["views"]) for row in base]
    plt.style.use("seaborn-v0_8-whitegrid")
    for key, ylabel, filename, note in [
        ("depth_abs_rel", "AbsRel", "p000_absrel.png", None),
        ("confidence_error_spearman", "Confidence/error Spearman rho", "p000_confidence.png", "More negative is better"),
    ]:
        fig, ax = plt.subplots(figsize=(7.6, 3.8))
        ax.plot(views, [float(r[key]) for r in base], "o-", lw=2.5, label="Official checkpoint")
        if key == "confidence_error_spearman":
            ax.axhline(0, color="#444444", lw=0.8)
        ax.set_xlabel("Input views")
        ax.set_ylabel(ylabel)
        if note:
            ax.text(0.02, 0.96, note, transform=ax.transAxes, va="top", weight="bold", color=NAVY)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    metrics = [("depth_abs_rel", "AbsRel"), ("depth_delta1", "delta1")]
    for ax, (key, label) in zip(axes, metrics):
        ax.plot(views, [float(r[key]) for r in base], "o-", lw=2.3, label="Official")
        ax.plot(views, [float(r[key]) for r in adapted], "s--", lw=2.3, label="Adapted step 15")
        ax.set_xlabel("Input views")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "pretrained_vs_adapted.png", dpi=200)
    plt.close(fig)

    history = read_csv(TRAIN)
    validation = [(int(r["step"]), float(r["validation_objective"])) for r in history if r.get("validation_objective")]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.plot([int(r["step"]) for r in history], [float(r["objective"]) for r in history],
            color="#78909C", alpha=0.65, label="Training-sample objective")
    ax.plot([x for x, _ in validation], [y for _, y in validation], "o-", color=GOLD,
            lw=2.5, label="P006 validation objective")
    ax.axvline(15, color=RED, ls="--", label="Selected step 15")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Objective")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "adaptation_validation.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    flow_figure(
        [
            "MULTI-VIEW RGB IMAGES",
            "DINOv2 IMAGE-TOKEN ENCODER",
            "ALTERNATING FRAME / GLOBAL ATTENTION AGGREGATOR",
            "FIRST-CAMERA REFERENCE REPRESENTATION",
            "CAMERA  |  DEPTH  |  POINT MAP  |  TRACKING",
        ],
        "vggt_architecture_corrected.png",
    )
    flow_figure(
        [
            "OFFICIAL VGGT REPRODUCTION",
            "ETH3D REAL-SCENE VARIED-INPUT STUDY",
            "NEED FOR EXACT METRIC GROUND TRUTH",
            "TARTANAIR HELD-OUT QUANTITATIVE EVALUATION",
            "BOUNDED CAMERA / DEPTH-HEAD ADAPTATION",
            "PRETRAINED VERSUS ADAPTED COMPARISON",
        ],
        "experimental_ladder.png",
        "Experimental ladder",
    )
    eth3d_figure()
    overlap_figure()
    nested_subsets()
    charts()
    print(f"Created {len(list(OUT.glob('*.png')))} figures in {OUT}")


if __name__ == "__main__":
    main()
