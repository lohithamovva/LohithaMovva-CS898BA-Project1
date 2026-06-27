"""Homework Two — Part 4: HSV color-space K-Means clustering segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NORMALIZED_IMAGE = PROJECT_ROOT / "output" / "hw2" / "part2" / "01_multi_channel_normalized.png"
OUTPUT_DIR = PROJECT_ROOT / "output" / "hw2" / "part4"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

K_VALUES = [3, 4, 5]
KMEANS_ATTEMPTS = 10
KMEANS_MAX_ITER = 100
KMEANS_EPS = 1.0

# Expected figure region (fractions of height/width) for cluster scoring.
FIGURE_ROI = (0.35, 0.85, 0.55, 0.78)  # y0, y1, x0, x1


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def extract_foreground(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    return cv2.bitwise_and(source, source, mask=binary)


def run_kmeans(hsv: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Cluster pixels in HSV space; return label map and cluster centers."""
    height, width = hsv.shape[:2]
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, KMEANS_MAX_ITER, KMEANS_EPS)
    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        KMEANS_ATTEMPTS,
        cv2.KMEANS_PP_CENTERS,
    )
    label_map = labels.reshape(height, width)
    return label_map, centers


def score_cluster(label_map: np.ndarray, cluster_id: int, height: int, width: int) -> float:
    """Higher score = cluster more likely to be the central figure."""
    mask = (label_map == cluster_id).astype(np.uint8)
    y0, y1, x0, x1 = FIGURE_ROI
    roi = mask[int(height * y0) : int(height * y1), int(width * x0) : int(width * x1)]
    roi_coverage = roi.sum() / max(roi.size, 1)

    total_area = mask.sum() / max(mask.size, 1)
    # Prefer moderate area (figure-sized) with strong ROI presence.
    area_penalty = abs(total_area - 0.04) * 2.0
    return roi_coverage - area_penalty


def select_figure_cluster(label_map: np.ndarray, k: int) -> int:
    height, width = label_map.shape
    scores = [score_cluster(label_map, cluster_id, height, width) for cluster_id in range(k)]
    return int(np.argmax(scores))


def cluster_colored_preview(label_map: np.ndarray, k: int) -> np.ndarray:
    rng = np.random.default_rng(898)
    colors = rng.integers(0, 255, size=(k, 3), dtype=np.uint8)
    preview = colors[label_map]
    return cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)


def main() -> None:
    normalized = cv2.imread(str(NORMALIZED_IMAGE))
    if normalized is None:
        raise FileNotFoundError(
            f"Normalized image not found: {NORMALIZED_IMAGE}. Run hw2_part2_normalization.py first."
        )

    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
    k_results: list[dict] = []
    best_k = K_VALUES[0]
    best_score = float("-inf")
    best_label_map: np.ndarray | None = None
    best_cluster_id = 0

    for k in K_VALUES:
        label_map, centers = run_kmeans(hsv, k)
        figure_cluster = select_figure_cluster(label_map, k)
        height, width = label_map.shape
        score = score_cluster(label_map, figure_cluster, height, width)

        k_dir = OUTPUT_DIR / f"k_{k}"
        save_image(k_dir / "00_cluster_preview.png", cluster_colored_preview(label_map, k))

        for cluster_id in range(k):
            cluster_mask = np.where(label_map == cluster_id, 255, 0).astype(np.uint8)
            save_image(k_dir / f"01_cluster_{cluster_id}_mask.png", cluster_mask)

        figure_mask = np.where(label_map == figure_cluster, 255, 0).astype(np.uint8)
        save_image(k_dir / "02_figure_cluster_mask.png", figure_mask)
        save_image(k_dir / "03_figure_cluster_foreground.png", extract_foreground(normalized, figure_mask))

        k_results.append(
            {
                "k": k,
                "figure_cluster_id": figure_cluster,
                "figure_score": score,
                "centers_hsv": centers.tolist(),
            }
        )

        if score > best_score:
            best_score = score
            best_k = k
            best_label_map = label_map
            best_cluster_id = figure_cluster

    assert best_label_map is not None
    final_mask = np.where(best_label_map == best_cluster_id, 255, 0).astype(np.uint8)
    save_image(OUTPUT_DIR / "00_kmeans_mask.png", final_mask)
    save_image(OUTPUT_DIR / "01_kmeans_foreground.png", extract_foreground(normalized, final_mask))
    save_image(OUTPUT_DIR / "02_kmeans_cluster_preview.png", cluster_colored_preview(best_label_map, best_k))

    manifest = {
        "input_image": str(NORMALIZED_IMAGE.relative_to(PROJECT_ROOT)),
        "k_values_tested": K_VALUES,
        "selected_k": best_k,
        "selected_cluster_id": best_cluster_id,
        "selection_score": best_score,
        "figure_roi_fractions": {
            "y0": FIGURE_ROI[0],
            "y1": FIGURE_ROI[1],
            "x0": FIGURE_ROI[2],
            "x1": FIGURE_ROI[3],
        },
        "k_results": k_results,
        "outputs": {
            "kmeans_mask": str((OUTPUT_DIR / "00_kmeans_mask.png").relative_to(PROJECT_ROOT)),
            "kmeans_foreground": str((OUTPUT_DIR / "01_kmeans_foreground.png").relative_to(PROJECT_ROOT)),
            "cluster_preview": str((OUTPUT_DIR / "02_kmeans_cluster_preview.png").relative_to(PROJECT_ROOT)),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print("=== Homework Two — Part 4: K-Means Segmentation ===")
    for result in k_results:
        print(
            f"K={result['k']}: figure cluster={result['figure_cluster_id']}, "
            f"score={result['figure_score']:.4f}"
        )
    print(f"Selected K={best_k} (cluster {best_cluster_id}, score={best_score:.4f})")
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
