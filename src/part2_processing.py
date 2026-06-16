"""Part 2: Basic image analysis, color spaces, affine transforms, and Gaussian blur."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_IMAGE = PROJECT_ROOT / "data" / "HW1_IMG_CS898BA.png"
OUTPUT_DIR = PROJECT_ROOT / "output" / "part2"
BASE_DIR = OUTPUT_DIR / "01_base"
AFFINE_DIR = OUTPUT_DIR / "02_affine"
BLUR_DIR = OUTPUT_DIR / "03_blur"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

SIGMA_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

# Two unique affine transforms per base image (14 total). Each entry is unique.
AFFINE_TRANSFORMS = [
    {"name": "translate_x30_y-15", "type": "translation", "matrix": np.float32([[1, 0, 30], [0, 1, -15]])},
    {"name": "translate_x-20_y25", "type": "translation", "matrix": np.float32([[1, 0, -20], [0, 1, 25]])},
    {"name": "rotate_90", "type": "rotation", "angle": 90},
    {"name": "rotate_186", "type": "rotation", "angle": 186},
    {"name": "scale_0.85", "type": "scale", "scale": 0.85},
    {"name": "scale_1.25", "type": "scale", "scale": 1.25},
    {"name": "shear_x0.25", "type": "shear", "shear": (0.25, 0.0)},
    {"name": "shear_y-0.18", "type": "shear", "shear": (0.0, -0.18)},
    {"name": "rotate_45", "type": "rotation", "angle": 45},
    {"name": "translate_x10_y40", "type": "translation", "matrix": np.float32([[1, 0, 10], [0, 1, 40]])},
    {"name": "scale_0.65", "type": "scale", "scale": 0.65},
    {"name": "shear_x-0.12_y0.08", "type": "shear", "shear": (-0.12, 0.08)},
    {"name": "rotate_270", "type": "rotation", "angle": 270},
    {"name": "translate_x-35_y-5", "type": "translation", "matrix": np.float32([[1, 0, -35], [0, 1, -5]])},
]


def channel_statistics(channel: np.ndarray, channel_name: str) -> dict:
    """Compute per-channel image statistics."""
    flat = channel.astype(np.float64).ravel()
    mode_result = stats.mode(flat, keepdims=False)
    mode_value = int(mode_result.mode) if np.isscalar(mode_result.mode) else int(mode_result.mode[0])

    return {
        "channel": channel_name,
        "min": int(flat.min()),
        "max": int(flat.max()),
        "average": float(flat.mean()),
        "median": float(np.median(flat)),
        "mode": mode_value,
        "skew": float(stats.skew(flat)),
        "range": int(flat.max() - flat.min()),
        "standard_deviation": float(flat.std(ddof=0)),
        "variance": float(flat.var(ddof=0)),
    }


def print_statistics(image: np.ndarray) -> list[dict]:
    """Print and return statistics for BGR channels."""
    channel_names = ["Blue", "Green", "Red"]
    results = []
    print("\n=== Original Image Channel Statistics ===")
    for idx, name in enumerate(channel_names):
        stats_dict = channel_statistics(image[:, :, idx], name)
        results.append(stats_dict)
        print(f"\n{name} channel:")
        for key, value in stats_dict.items():
            if key != "channel":
                print(f"  {key}: {value}")
    return results


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def build_base_images(image: np.ndarray) -> dict[str, np.ndarray]:
    """Create the seven required base images."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    hsv_eq = hsv.copy()
    hsv_eq[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    hsv_normalized_rgb = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    images = {
        "00_original": image,
        "01_grayscale": gray,
        "02_binary": binary,
        "03_hsv": hsv,
        "04_cielab": lab,
        "05_hls": hls,
        "06_hsv_v_equalized_rgb": hsv_normalized_rgb,
    }

    for name, img in images.items():
        save_image(BASE_DIR / f"{name}.png", img)

    return images


def apply_affine(image: np.ndarray, transform: dict) -> np.ndarray:
    """Apply a single affine transformation."""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    if transform["type"] == "translation":
        matrix = transform["matrix"]
    elif transform["type"] == "rotation":
        matrix = cv2.getRotationMatrix2D(center, transform["angle"], 1.0)
    elif transform["type"] == "scale":
        matrix = cv2.getRotationMatrix2D(center, 0, transform["scale"])
    elif transform["type"] == "shear":
        shx, shy = transform["shear"]
        matrix = np.float32([[1, shx, 0], [shy, 1, 0]])
    else:
        raise ValueError(f"Unknown transform type: {transform['type']}")

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def apply_affine_transforms(base_images: dict[str, np.ndarray]) -> list[dict]:
    """Apply two unique affine transforms to each base image."""
    records: list[dict] = []
    transform_idx = 0

    for base_name, base_image in base_images.items():
        for local_idx in range(2):
            transform = AFFINE_TRANSFORMS[transform_idx]
            transformed = apply_affine(base_image, transform)
            out_name = f"{base_name}__{transform['name']}.png"
            out_path = AFFINE_DIR / out_name
            save_image(out_path, transformed)
            records.append(
                {
                    "source": base_name,
                    "filename": out_name,
                    "path": str(out_path.relative_to(PROJECT_ROOT)),
                    "transform": transform["name"],
                    "transform_type": transform["type"],
                }
            )
            transform_idx += 1

    return records


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur using sigma (kernel size derived from sigma)."""
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def collect_all_part2_images(base_images: dict[str, np.ndarray], affine_records: list[dict]) -> list[dict]:
    """Collect 21 images (7 base + 14 affine) before blur stage."""
    images: list[dict] = []

    for name, img in base_images.items():
        filename = f"{name}.png"
        images.append(
            {
                "filename": filename,
                "path": str((BASE_DIR / filename).relative_to(PROJECT_ROOT)),
                "source_stage": "base",
                "processing": name.replace("_", " "),
            }
        )

    for record in affine_records:
        images.append(
            {
                "filename": record["filename"],
                "path": record["path"],
                "source_stage": "affine",
                "processing": f"{record['source']} + {record['transform']} ({record['transform_type']})",
            }
        )

    return images


def apply_blurs(all_images: list[dict]) -> list[dict]:
    """Blur each of the 21 images at seven sigma levels.

    Also retain the 21 unblurred images so the full Part 2 set is 168 images.
    """
    blur_records: list[dict] = []

    for image_info in all_images:
        image_path = PROJECT_ROOT / image_info["path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        stem = Path(image_info["filename"]).stem

        unblurred_name = f"{stem}__blur_sigma_0.0.png"
        unblurred_path = BLUR_DIR / unblurred_name
        save_image(unblurred_path, image)
        blur_records.append(
            {
                "filename": unblurred_name,
                "path": str(unblurred_path.relative_to(PROJECT_ROOT)),
                "source_image": image_info["filename"],
                "sigma": 0.0,
                "processing": f"{image_info['processing']} + no blur (sigma=0.0)",
            }
        )

        for sigma in SIGMA_LEVELS:
            blurred = gaussian_blur(image, sigma)
            out_name = f"{stem}__blur_sigma_{sigma:.1f}.png"
            out_path = BLUR_DIR / out_name
            save_image(out_path, blurred)
            blur_records.append(
                {
                    "filename": out_name,
                    "path": str(out_path.relative_to(PROJECT_ROOT)),
                    "source_image": image_info["filename"],
                    "sigma": sigma,
                    "processing": f"{image_info['processing']} + Gaussian blur (sigma={sigma})",
                }
            )

    return blur_records


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(INPUT_IMAGE))
    if image is None:
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

    stats_results = print_statistics(image)

    base_images = build_base_images(image)
    affine_records = apply_affine_transforms(base_images)
    all_images = collect_all_part2_images(base_images, affine_records)
    blur_records = apply_blurs(all_images)

    manifest = {
        "input_image": str(INPUT_IMAGE.relative_to(PROJECT_ROOT)),
        "statistics": stats_results,
        "base_image_count": len(base_images),
        "affine_image_count": len(affine_records),
        "pre_blur_image_count": len(all_images),
        "blurred_image_count": len(blur_records),
        "total_part2_outputs": len(blur_records),
        "sigma_levels": SIGMA_LEVELS,
        "base_images": [str((BASE_DIR / f"{name}.png").relative_to(PROJECT_ROOT)) for name in base_images],
        "affine_records": affine_records,
        "blur_records": blur_records,
        "all_pre_blur_images": all_images,
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print(f"\nSaved 7 base images to {BASE_DIR}")
    print(f"Saved 14 affine images to {AFFINE_DIR}")
    print(f"Saved {len(blur_records)} blurred images to {BLUR_DIR}")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
