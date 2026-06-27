"""Homework Two — Part 3: Otsu and adaptive (Gaussian) threshold segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NORMALIZED_IMAGE = PROJECT_ROOT / "output" / "hw2" / "part2" / "01_multi_channel_normalized.png"
OUTPUT_DIR = PROJECT_ROOT / "output" / "hw2" / "part3"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

ADAPTIVE_BLOCK_SIZE = 51
ADAPTIVE_C = 8


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def extract_foreground(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply binary mask to source image (white = keep foreground)."""
    binary = (mask > 0).astype(np.uint8)
    return cv2.bitwise_and(source, source, mask=binary)


def segment_otsu(gray: np.ndarray) -> tuple[np.ndarray, float]:
    otsu_threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask, float(otsu_threshold)


def segment_adaptive(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )


def main() -> None:
    normalized = cv2.imread(str(NORMALIZED_IMAGE))
    if normalized is None:
        raise FileNotFoundError(
            f"Normalized image not found: {NORMALIZED_IMAGE}. Run hw2_part2_normalization.py first."
        )

    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

    otsu_mask, otsu_threshold = segment_otsu(gray)
    adaptive_mask = segment_adaptive(gray)

    otsu_foreground = extract_foreground(normalized, otsu_mask)
    adaptive_foreground = extract_foreground(normalized, adaptive_mask)

    save_image(OUTPUT_DIR / "01_otsu_mask.png", otsu_mask)
    save_image(OUTPUT_DIR / "02_otsu_foreground.png", otsu_foreground)
    save_image(OUTPUT_DIR / "03_adaptive_mask.png", adaptive_mask)
    save_image(OUTPUT_DIR / "04_adaptive_foreground.png", adaptive_foreground)
    save_image(OUTPUT_DIR / "00_grayscale_normalized.png", gray)

    manifest = {
        "input_image": str(NORMALIZED_IMAGE.relative_to(PROJECT_ROOT)),
        "otsu_threshold": otsu_threshold,
        "adaptive_block_size": ADAPTIVE_BLOCK_SIZE,
        "adaptive_c": ADAPTIVE_C,
        "outputs": {
            "grayscale": str((OUTPUT_DIR / "00_grayscale_normalized.png").relative_to(PROJECT_ROOT)),
            "otsu_mask": str((OUTPUT_DIR / "01_otsu_mask.png").relative_to(PROJECT_ROOT)),
            "otsu_foreground": str((OUTPUT_DIR / "02_otsu_foreground.png").relative_to(PROJECT_ROOT)),
            "adaptive_mask": str((OUTPUT_DIR / "03_adaptive_mask.png").relative_to(PROJECT_ROOT)),
            "adaptive_foreground": str((OUTPUT_DIR / "04_adaptive_foreground.png").relative_to(PROJECT_ROOT)),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print("=== Homework Two — Part 3: Threshold Segmentation ===")
    print(f"Otsu threshold: {otsu_threshold:.2f}")
    print(f"Adaptive: block={ADAPTIVE_BLOCK_SIZE}, C={ADAPTIVE_C}")
    print(f"Saved masks and foreground extractions to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
