"""Part 3: Subset selection, edge detection, and comparison plots."""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PART2_MANIFEST = PROJECT_ROOT / "output" / "part2" / "manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "part3"
SUBSET_DIR = OUTPUT_DIR / "01_subset"
EDGE_DIR = OUTPUT_DIR / "02_edges"
PLOTS_DIR = OUTPUT_DIR / "03_plots"
README_PLOTS_DIR = PROJECT_ROOT / "output" / "readme_plots"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

EDGE_METHODS = ("sobel", "laplacian", "canny", "prewitt")
SELECTED_SUBSET_INDEX = 0
RANDOM_SEED = 898
NUM_README_PLOTS = 6


def prewitt_edges(gray: np.ndarray) -> np.ndarray:
    """Apply Prewitt edge detection using separable kernels."""
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    grad_x = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel_y)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def detect_edges(gray: np.ndarray, method: str) -> np.ndarray:
    """Run one edge detection method on a grayscale image."""
    if method == "sobel":
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if method == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        return cv2.convertScaleAbs(lap)
    if method == "canny":
        v = float(np.median(gray))
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        return cv2.Canny(gray, lower, upper)
    if method == "prewitt":
        return prewitt_edges(gray)
    raise ValueError(f"Unknown edge method: {method}")


def overlay_edges(base_bgr: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Draw white edges on top of the original image."""
    if len(base_bgr.shape) == 2:
        overlay = cv2.cvtColor(base_bgr, cv2.COLOR_GRAY2BGR)
    else:
        overlay = base_bgr.copy()
    overlay[edges > 0] = (255, 255, 255)
    return overlay


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def load_part2_blur_images() -> list[dict]:
    manifest = json.loads(PART2_MANIFEST.read_text())
    return manifest["blur_records"]


def split_into_subsets(items: list[dict], seed: int = RANDOM_SEED) -> list[list[dict]]:
    shuffled = items.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    subset_size = len(shuffled) // 4
    return [shuffled[i * subset_size : (i + 1) * subset_size] for i in range(4)]


def save_subset(subset: list[dict], subset_index: int) -> list[dict]:
    records = []
    subset_folder = SUBSET_DIR / f"subset_{subset_index}"
    subset_folder.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(subset):
        src = PROJECT_ROOT / item["path"]
        image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Missing image: {src}")

        filename = f"{idx:02d}_{Path(item['filename']).name}"
        out_path = subset_folder / filename
        cv2.imwrite(str(out_path), image)
        records.append(
            {
                "subset_index": subset_index,
                "subset_filename": filename,
                "subset_path": str(out_path.relative_to(PROJECT_ROOT)),
                "source_blur_record": item,
            }
        )
    return records


def edge_strength_score(edges: np.ndarray) -> dict:
    edge_pixels = int(np.count_nonzero(edges))
    total_pixels = edges.size
    return {
        "edge_pixel_count": edge_pixels,
        "edge_density": edge_pixels / total_pixels,
        "mean_edge_intensity": float(edges.mean()),
    }


def process_subset(subset_records: list[dict]) -> tuple[list[dict], dict]:
    edge_records: list[dict] = []
    method_scores: dict[str, list[float]] = {method: [] for method in EDGE_METHODS}

    for record in subset_records:
        image_path = PROJECT_ROOT / record["subset_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Missing subset image: {image_path}")

        gray = to_grayscale(image)
        stem = Path(record["subset_filename"]).stem

        before_path = EDGE_DIR / "before" / f"{stem}.png"
        before_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(before_path), image)

        for method in EDGE_METHODS:
            edges = detect_edges(gray, method)
            after_path = EDGE_DIR / method / f"{stem}__{method}.png"
            after_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(after_path), overlay_edges(image, edges))

            scores = edge_strength_score(edges)
            method_scores[method].append(scores["edge_density"])

            edge_records.append(
                {
                    "subset_filename": record["subset_filename"],
                    "before_path": str(before_path.relative_to(PROJECT_ROOT)),
                    "after_path": str(after_path.relative_to(PROJECT_ROOT)),
                    "method": method,
                    "processing": record["source_blur_record"]["processing"],
                    "scores": scores,
                }
            )

    summary = {
        method: {
            "average_edge_density": float(np.mean(values)),
            "std_edge_density": float(np.std(values)),
        }
        for method, values in method_scores.items()
    }
    return edge_records, summary


def create_comparison_plots(subset_records: list[dict], edge_records: list[dict]) -> list[str]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    README_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    edge_lookup: dict[str, dict[str, str]] = {}
    for record in edge_records:
        edge_lookup.setdefault(record["subset_filename"], {})[record["method"]] = record["after_path"]

    plot_paths: list[str] = []
    titles = ["Input", "Sobel", "Laplacian", "Canny", "Prewitt"]

    for record in subset_records:
        filename = record["subset_filename"]
        input_path = PROJECT_ROOT / record["subset_path"]
        input_img = cv2.imread(str(input_path))
        input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        axes[0].imshow(input_rgb)
        axes[0].set_title("Input")
        axes[0].axis("off")

        for axis, method, title in zip(axes[1:], EDGE_METHODS, titles[1:]):
            edge_path = PROJECT_ROOT / edge_lookup[filename][method]
            edge_img = cv2.imread(str(edge_path))
            edge_rgb = cv2.cvtColor(edge_img, cv2.COLOR_BGR2RGB)
            axis.imshow(edge_rgb)
            axis.set_title(title)
            axis.axis("off")

        processing = record["source_blur_record"]["processing"]
        fig.suptitle(f"Edge Detection Comparison\n{processing}", fontsize=11)
        fig.tight_layout()

        plot_name = f"plot_{Path(filename).stem}.png"
        plot_path = PLOTS_DIR / plot_name
        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(str(plot_path.relative_to(PROJECT_ROOT)))

    random.seed(RANDOM_SEED)
    selected = random.sample(plot_paths, NUM_README_PLOTS)
    for rel_path in selected:
        src = PROJECT_ROOT / rel_path
        dst = README_PLOTS_DIR / Path(rel_path).name
        dst.write_bytes(src.read_bytes())

    return selected


def main() -> None:
    if not PART2_MANIFEST.exists():
        raise FileNotFoundError("Run part2_processing.py first to generate blurred images.")

    blur_images = load_part2_blur_images()
    if len(blur_images) != 168:
        raise ValueError(f"Expected 168 blurred images, found {len(blur_images)}")

    subsets = split_into_subsets(blur_images)
    subset_manifests = []
    for idx, subset in enumerate(subsets):
        subset_manifests.append(save_subset(subset, idx))

    selected_subset = subset_manifests[SELECTED_SUBSET_INDEX]
    edge_records, score_summary = process_subset(selected_subset)
    readme_plot_paths = create_comparison_plots(selected_subset, edge_records)

    total_saved = len(selected_subset) + len(edge_records)
    manifest = {
        "selected_subset_index": SELECTED_SUBSET_INDEX,
        "subset_count": len(subsets),
        "images_per_subset": len(subsets[0]),
        "selected_subset_size": len(selected_subset),
        "edge_methods": list(EDGE_METHODS),
        "edge_output_count": len(edge_records),
        "total_part3_saved_images": total_saved,
        "score_summary": score_summary,
        "readme_plot_paths": readme_plot_paths,
        "selected_subset": selected_subset,
        "edge_records": edge_records,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print(f"Created 4 subsets of {len(subsets[0])} images each.")
    print(f"Selected subset {SELECTED_SUBSET_INDEX} for edge detection.")
    print(f"Saved {total_saved} images in part 3 output folders.")
    print(f"Copied {len(readme_plot_paths)} plots to {README_PLOTS_DIR}")
    print("\nAverage edge density by method:")
    for method, values in score_summary.items():
        print(f"  {method}: {values['average_edge_density']:.6f}")


if __name__ == "__main__":
    main()
