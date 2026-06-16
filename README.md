# COMP430 — Turkish License Plate Recognition via Hybrid Multiscale Morphological Analysis

**Final Term Project — COMP430 Digital Image Processing**

A classical, training-free image processing pipeline for **Turkish license plate localization and OCR**. No deep learning required — pure OpenCV morphology + Tesseract.

---

## Results (100-image benchmark)

| Metric | Baseline (Raw Tesseract) | Proposed Algorithm |
|---|---|---|
| Localization Accuracy (IoU > 0.5) | — | **73.00%** |
| Mean Character Error Rate (CER) | 99.86% | **45.77%** |

---

## How It Works

The pipeline has 4 stages:

1. **Dual-Channel Binary Map Generation**  
   - Sobel-X edge detection → Otsu threshold → `B_s`  
   - Black Hat morphological transform (15×15 SE) → Otsu threshold → `B_bh`

2. **Multiscale Candidate Extraction**  
   Both maps are closed at 5 kernel widths `k ∈ {9, 17, 29, 47, 65}` px to handle plates at any distance. Candidate bounding boxes are filtered by aspect ratio (1.5–7.0) and area (400–70,000 px).

3. **Aligned Character Scoring**  
   Each candidate is scored by the number, height uniformity, and vertical alignment of character-like sub-contours inside it:  
   `score = 15·n_chars + 30·U_h − 5·|r − 4.5|`  
   The highest-scoring region is selected as the plate.

4. **Localized OCR**  
   The selected crop is locally binarized (Black Hat + Otsu) and passed to Tesseract OCR (PSM 7, whitelist: A–Z, 0–9).

---

## Requirements & Installation

### 1. Python libraries

```bash
pip install opencv-python numpy pytesseract python-Levenshtein
```

### 2. Tesseract OCR Engine

You **must** install Tesseract separately — `pytesseract` is only a Python wrapper.

**Windows:**
1. Download from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install to the default path: `C:\Program Files\Tesseract-OCR`

**Linux / Raspberry Pi:**
```bash
sudo apt-get install tesseract-ocr
```
> On Linux, remove or comment out the `pytesseract.pytesseract.tesseract_cmd` line at the top of `imagefinal.py`.

---

## Dataset Structure

The `dataset/` folder must contain `.jpg` images and their corresponding YOLO-format `.txt` annotation files in the same directory:

```
imagefinal/
├── imagefinal.py
└── dataset/
    ├── image_001.jpg
    ├── image_001.txt    ← YOLO format: class x_center y_center width height
    ├── image_002.jpg
    └── image_002.txt
```

The dataset used in this project is the [Turkish Number Plates v2](https://universe.roboflow.com/plakatanima-vnt3k/turkish-number-plates/dataset/2) dataset from Roboflow Universe (CC BY 4.0).

---

## How to Run

```bash
python imagefinal.py
```

A window will open for each image showing:
- 🟦 **Blue box** — Ground truth YOLO annotation
- 🟩 **Green box** — Proposed pipeline prediction
- 🟠 **Orange boxes** — Individual detected character regions
- HUD overlay with Ground Truth text, OCR output, and IoU score

Press **any key** to advance to the next image. Final statistics are printed to the terminal after all images are processed.

---

## Project Structure

```
imagefinal/
├── imagefinal.py   # Main pipeline: localization + OCR + evaluation loop
├── dataset/        # 100 benchmark images + YOLO labels (manually annotated GT)
├── README.md
└── .gitignore
```

---

## Evaluation Metrics

- **Localization Accuracy**: Fraction of images where predicted box achieves IoU ≥ 0.5 with ground truth
- **Character Error Rate (CER)**: `edit_distance(predicted, ground_truth) / len(ground_truth)` — lower is better
