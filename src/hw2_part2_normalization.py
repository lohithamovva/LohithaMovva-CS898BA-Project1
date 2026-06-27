"""Homework Two — Part 2: Multi-channel color normalization via histogram equalization."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_IMAGE = PROJECT_ROOT / "data" / "HW1_IMG_CS898BA.png"
OUTPUT_DIR = PROJECT_ROOT / "output" / "hw2" / "part2"
NORMALIZED_PATH = OUTPUT_DIR / "01_multi_channel_normalized.png"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def multi_channel_equalize(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization independently to each BGR channel."""
    channels = cv2.split(image)
    equalized = [cv2.equalizeHist(channel) for channel in channels]
    return cv2.merge(equalized)


def main() -> None:
    image = cv2.imread(str(INPUT_IMAGE))
    if image is None:
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

    normalized = multi_channel_equalize(image)
    save_image(NORMALIZED_PATH, normalized)

    manifest = {
        "input_image": str(INPUT_IMAGE.relative_to(PROJECT_ROOT)),
        "output_image": str(NORMALIZED_PATH.relative_to(PROJECT_ROOT)),
        "method": "Independent histogram equalization on B, G, and R channels",
        "shape": list(image.shape),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print("=== Homework Two — Part 2: Multi-Channel Normalization ===")
    print(f"Input:  {INPUT_IMAGE}")
    print(f"Output: {NORMALIZED_PATH}")
    print(f"Saved normalized color image ({normalized.shape[1]}×{normalized.shape[0]})")


if __name__ == "__main__":
    main()
