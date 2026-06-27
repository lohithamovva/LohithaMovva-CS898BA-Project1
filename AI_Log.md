# AI Usage Log — CS 898BA Project 1

| Date and Time | Prompt | Tool | Response Synopsis | Change |
|---------------|--------|------|-------------------|--------|
| 06/12/2026 12:07 AM | Full Homework One assignment prompt for CS 898BA (Parts 1–4): create git repo LohithaMovva-CS898BA-Project1, Hello World script, AI_Log.md, incremental commits, OpenCV pipeline for image stats/color spaces/affine transforms/Gaussian blur/edge detection/plots, and comprehensive README. | Gemini | Generated complete Python project structure with `hello_world.py`, `src/part2_processing.py`, `src/part3_edge_detection.py`, copied `HW1_IMG_CS898BA.png`, produced all required outputs (168 Part 2 images, 210 Part 3 images, 42 comparison plots, 6 README plots), and drafted README.md with setup, results, and discussion. | Entire codebase, documentation, and output pipeline design. |
| 06/12/2026 12:07 AM | Confirm author and repository naming as Lohitha Movva / LohithaMovva-CS898BA-Project1. | Gemini | Updated repository name, README author line, and AI_Log to use Lohitha Movva consistently. | Updated README.md and AI_Log.md. |
| 06/12/2026 12:07 AM | Review edge detection results, Gaussian blur discussion, and README plot selection for Part 3 analysis. | Gemini | Refined written analysis of Sobel, Laplacian, Canny, and Prewitt; confirmed Canny as best method for low-light noisy doorbell image set. | README discussion and edge detection comparison sections. |
| 06/27/2026 07:45 PM | Full Homework Two assignment prompt for CS 898BA (Parts 1–6): create Feature-Segmentation branch, multi-channel histogram equalization, Otsu/adaptive thresholding, HSV K-Means (K=3–5), pseudo-ground-truth mask, IoU/Dice metrics, comparison plot in README, incremental commits, AI_Log update. | Cursor (Claude) | Implemented `src/hw2_part2_normalization.py` through `hw2_part5_evaluation.py`, generated segmentation outputs, pseudo-ground-truth ellipses, quantitative metrics, and README Homework Two section with qualitative analysis. | HW2 segmentation pipeline, evaluation scripts, README, AI_Log, ground truth mask, comparison plot. |

## Detailed Prompt (06/27/2026 07:45 PM)

```
Homework Two — CS 898BA Image Segmentation

Part 1: Feature-Segmentation branch, AI_Log.md, incremental commits.
Part 2: Multi-channel BGR histogram equalization → normalized color image.
Part 3: Otsu global + adaptive Gaussian threshold on normalized grayscale; save masks and foreground extractions.
Part 4: HSV K-Means clustering (K=3,4,5); select optimal K; isolate figure cluster.
Part 5: Qualitative analysis, pseudo-ground-truth mask, IoU and Dice vs 3 methods, comparison plot in README.
Part 6: Submit updated GitHub repository link.
```

## Design Decisions Influenced by AI (Homework Two)

1. **Multi-channel equalization on BGR** (not just HSV-V) to satisfy the assignment’s full-spectrum normalization requirement.
2. **K-Means cluster selection** uses a figure ROI score (coverage in right-center region minus area penalty) rather than manual label picking.
3. **Pseudo-ground-truth** defined as two hand-placed ellipses (head + torso/legs) for reproducible biomass-region annotation.
4. **Raw vs. normalized Otsu baseline** included in evaluation to quantify normalization impact vs. Homework One.
5. **HW2 generated outputs** gitignored under `output/hw2/`; comparison plot committed to `output/readme_plots/`.

## Design Decisions Influenced by AI

```
Homework One — CS 898BA Image Analysis and Computer Vision

Purpose: To apply basic image analysis and processing techniques.

Situation: You are working at Meta, and your supervisor's supervisor walks in, gasping for air. You don't often interact with him, so this visit is strange. He opens a folder on his personal computer and shows you an image he thinks is an alien, captured by his doorbell camera. You are not convinced and think he is just seeing things. He came to see you because you heard you took an image analysis course in college and wants you to clean up the image so it is easier to make out what is in it.

Part 1: Create git repository FirstnameLastname-CS898BA-Project1, Hello World script, AI_Log.md, incremental commits, comprehensive README.

Part 2: Channel statistics, color space conversions (grayscale, binary, HSV, CIELAB, HLS), HSV V histogram equalization, 14 affine transforms (2 per 7 images), Gaussian blur at sigma 0.5–3.5 (168 total images).

Part 3: Split into 4 subsets of 42, choose one, apply Sobel/Laplacian/Canny/Prewitt edge detection (210 images), create 42 five-image plots and include 6 in README.

Part 4: Submit GitHub link on Blackboard.
```

## Design Decisions Influenced by AI

1. **168-image blur set**: Each of the 21 pre-blur images is saved once at sigma=0.0 plus seven blurred variants (21 × 8 = 168).
2. **Fixed random seed (898)** for subset shuffling and README plot selection to keep results reproducible.
3. **Canny thresholds** use median-based auto thresholds (`0.66×median`, `1.33×median`) for adaptability to low-light images.
4. **Edge overlay output**: “After” images overlay detected edges in white on the original for visual comparison in plots.
5. **Large generated outputs** are gitignored under `output/part2/` and `output/part3/`; sample plots in `output/readme_plots/` are committed for README display. Full outputs regenerate via the scripts.
