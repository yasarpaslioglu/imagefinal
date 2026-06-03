# Edge-Optimized Automatic License Plate Recognition (ALPR)

##  Project Overview
This project implements a lightweight, highly optimized Computer Vision pipeline for Automatic License Plate Recognition (ALPR). Unlike modern computationally heavy deep-learning models (like YOLO or CRNN), this project utilizes **Mathematical Morphology (Black Hat Transformation)** and **Dynamic Geometric Clustering** to successfully localize and read license plates. 

It is specifically designed to be robust against real-world environmental degradations (shadows, dirt, motion blur) while maintaining a low computational footprint suitable for resource-constrained edge devices (e.g., Raspberry Pi) or rapid PC deployment.

###  Key Features
* **Zero Deep-Learning Localization:** Achieves 100% plate localization on the test set using purely traditional computer vision techniques.
* **Morphological Pre-processing:** Utilizes Black Hat transformations and localized Otsu Binarization to defeat severe shadow gradients and glare.
* **Dynamic Heuristics:** Employs Y-axis alignment, X-axis clustering, and Macro Aspect Ratio filters to eliminate background noise (trees, taillights, logos).
* **Visual Debugging HUD:** Includes an interactive UI that draws bounding boxes around predictions (Green), individual characters (Orange), and ground truth (Blue) for real-time failure mode analysis.

---

## ⚙️ Requirements & Installation

### 1. Python Libraries
This project requires Python 3.7+ and the following libraries. You can install them via pip:

```bash
pip install opencv-python numpy pytesseract python-Levenshtein
```
### 2. Tesseract OCR Engine (CRUCIAL)
The `pytesseract` library is just a Python wrapper. You **must** install the actual Tesseract OCR engine on your system for the OCR to work.

* **For Windows:** 1. Download the installer from the [UB-Mannheim Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki).
  2. Install it to the default directory (`C:\Program Files\Tesseract-OCR`).
  3. *Note: The Python script is already configured to point to this exact Windows directory.*
* **For Linux (Raspberry Pi / Ubuntu):**
  ```bash
  sudo apt-get update
  sudo apt-get install tesseract-ocr
  ```
  *(If using Linux, remove or comment out the `pytesseract.pytesseract.tesseract_cmd` line at the top of the Python script).*

---

## 📂 Dataset Structure

The algorithm is built to process images and evaluate them against **YOLO-format** annotations. Ensure your dataset folder contains both the `.jpg` images and their corresponding `.txt` YOLO bounding box files in the same directory.

```text
C:\Users\YourName\Desktop\imagefinal\
│
├── imagefinal.py          <-- Main Python script
└── dataset\               <-- Folder containing your images and labels
    ├── car_01.jpg
    ├── car_01.txt         <-- YOLO format (class x_center y_center width height)
    ├── car_02.jpg
    └── car_02.txt
```

---

## 🚀 How to Run

1. Open the `imagefinal.py` script in your preferred IDE or text editor.
2. Scroll to the bottom of the script to the `if __name__ == "__main__":` block.
3. Update the `dataset_folder` variable with the absolute path to your dataset folder.
4. *(Optional)* Update the `manual_ground_truth` dictionary with the exact text of the license plates to calculate the exact Character Error Rate (CER).
5. Run the script:

```bash
python imagefinal.py
```
### Interacting with the Visual Debugger
Once the script runs, a window will pop up showing the first processed image. 
* Look at the **Heads Up Display (HUD)** in the top left for real-time Ground Truth vs. OCR Output and Localization (IoU) scores.
* **Press any key** on your keyboard to close the current image and process the next one.
* After all images are processed, the terminal will print a final Scientific Experiment Results report.

---
