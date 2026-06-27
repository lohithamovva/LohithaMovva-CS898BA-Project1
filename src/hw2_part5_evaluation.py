"""Homework Two — Part 5: Ground-truth evaluation, metrics, and comparison visualization."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_IMAGE = PROJECT_ROOT / "data" / "HW1_IMG_CS898BA.png"
NORMALIZED_IMAGE = PROJECT_ROOT / "output" / "hw2" / "part2" / "01_multi_channel_normalized.png"
PART3_DIR = PROJECT_ROOT / "output" / "hw2" / "part3"
PART4_DIR = PROJECT_ROOT / "output" / "hw2" / "part4"
OUTPUT_DIR = PROJECT_ROOT / "output" / "hw2" / "part5"
README_PLOT_DIR = PROJECT_ROOT / "output" / "readme_plots"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "ground_truth_mask.png"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

# Manually defined pseudo-ground-truth: ellipse model of head + torso/legs.
GROUND_TRUTH_ELLIPSES = [
    {"center": (1780, 500), "axes": (95, 120), "angle": 0},
    {"center": (1780, 820), "axes": (130, 380), "angle": 0},
]


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def create_ground_truth_mask(shape: tuple[int, int]) -> np.ndarray:
    """Build pseudo-ground-truth binary mask from hand-defined ellipses."""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for ellipse in GROUND_TRUTH_ELLIPSES:
        cv2.ellipse(
            mask,
            ellipse["center"],
            ellipse["axes"],
            ellipse["angle"],
            0,
            360,
            255,
            -1,
        )
    return mask


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 127).astype(np.uint8)


def iou_score(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_bin = binarize_mask(pred)
    truth_bin = binarize_mask(truth)
    intersection = np.logical_and(pred_bin, truth_bin).sum()
    union = np.logical_or(pred_bin, truth_bin).sum()
    return float(intersection / union) if union > 0 else 0.0


def dice_score(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_bin = binarize_mask(pred)
    truth_bin = binarize_mask(truth)
    intersection = np.logical_and(pred_bin, truth_bin).sum()
    total = pred_bin.sum() + truth_bin.sum()
    return float(2 * intersection / total) if total > 0 else 0.0


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return mask


def otsu_on_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def make_comparison_plot(
    original: np.ndarray,
    normalized: np.ndarray,
    otsu_mask: np.ndarray,
    adaptive_mask: np.ndarray,
    kmeans_mask: np.ndarray,
    ground_truth: np.ndarray,
    out_path: Path,
) -> None:
    """Six-panel comparison for README (original, normalized, 3 methods, ground truth)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    panels = [
        (original, "Original Image", "RGB"),
        (normalized, "Multi-Channel Normalized", "RGB"),
        (otsu_mask, "Otsu Mask", "gray"),
        (adaptive_mask, "Adaptive Mask", "gray"),
        (kmeans_mask, "K-Means Mask (K=selected)", "gray"),
        (ground_truth, "Pseudo-Ground Truth", "gray"),
    ]

    for ax, (image, title, cmap) in zip(axes.ravel(), panels):
        if cmap == "RGB":
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        "Homework Two: Segmentation Comparison — Doorbell Figure Isolation",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    original = cv2.imread(str(INPUT_IMAGE))
    normalized = cv2.imread(str(NORMALIZED_IMAGE))
    if original is None or normalized is None:
        raise FileNotFoundError("Missing original or normalized image. Run prior HW2 scripts.")

    ground_truth = create_ground_truth_mask(original.shape[:2])
    save_image(GROUND_TRUTH_PATH, ground_truth)
    save_image(OUTPUT_DIR / "00_ground_truth_mask.png", ground_truth)

    otsu_mask = load_mask(PART3_DIR / "01_otsu_mask.png")
    adaptive_mask = load_mask(PART3_DIR / "03_adaptive_mask.png")
    kmeans_mask = load_mask(PART4_DIR / "00_kmeans_mask.png")

    raw_otsu_mask = otsu_on_image(original)
    save_image(OUTPUT_DIR / "01_raw_otsu_mask.png", raw_otsu_mask)

    methods = {
        "Otsu (normalized)": otsu_mask,
        "Adaptive (normalized)": adaptive_mask,
        "K-Means (normalized)": kmeans_mask,
    }

    print("=== Homework Two — Part 5: Evaluation ===\n")
    print("Quantitative Comparison vs Pseudo-Ground Truth")
    print(f"{'Method':<28} {'IoU':>8} {'Dice':>8}")
    print("-" * 46)

    metrics: dict[str, dict[str, float]] = {}
    for name, mask in methods.items():
        iou = iou_score(mask, ground_truth)
        dice = dice_score(mask, ground_truth)
        metrics[name] = {"iou": iou, "dice": dice}
        print(f"{name:<28} {iou:8.4f} {dice:8.4f}")

    raw_iou = iou_score(raw_otsu_mask, ground_truth)
    raw_dice = dice_score(raw_otsu_mask, ground_truth)
    print(f"\nReference — Otsu on RAW image (HW1 baseline): IoU={raw_iou:.4f}, Dice={raw_dice:.4f}")

    comparison_path = README_PLOT_DIR / "hw2_segmentation_comparison.png"
    make_comparison_plot(
        original,
        normalized,
        otsu_mask,
        adaptive_mask,
        kmeans_mask,
        ground_truth,
        comparison_path,
    )
    save_image(OUTPUT_DIR / "02_segmentation_comparison.png", cv2.imread(str(comparison_path)))

    manifest = {
        "ground_truth_path": str(GROUND_TRUTH_PATH.relative_to(PROJECT_ROOT)),
        "figure_ellipses": GROUND_TRUTH_ELLIPSES,
        "metrics_vs_ground_truth": metrics,
        "raw_otsu_baseline": {"iou": raw_iou, "dice": raw_dice},
        "comparison_plot": str(comparison_path.relative_to(PROJECT_ROOT)),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print(f"\nGround truth saved to {GROUND_TRUTH_PATH}")
    print(f"Comparison plot saved to {comparison_path}")


if __name__ == "__main__":
    main()
