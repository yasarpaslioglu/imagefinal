import cv2
import numpy as np
import pytesseract
import Levenshtein
import os
import glob

# ==========================================
# WINDOWS TESSERACT PATH
# ==========================================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ==========================================
# 1. EVALUATION METRICS
# ==========================================

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) for two bounding boxes [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def calculate_cer(predicted_text, ground_truth_text):
    """Calculates Character Error Rate (CER) using Levenshtein distance."""
    if len(ground_truth_text) == 0:
        return 1.0
    
    predicted_text = predicted_text.strip().upper()
    ground_truth_text = ground_truth_text.strip().upper()
    
    distance = Levenshtein.distance(predicted_text, ground_truth_text)
    cer = distance / len(ground_truth_text)
    return cer

# ==========================================
# 2. YOLO LABEL PARSER
# ==========================================

def parse_yolo_label(txt_path, img_width, img_height):
    """Converts YOLO normalized format to pixel coordinates [x, y, w, h]."""
    if not os.path.exists(txt_path):
        return None
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                _, x_center, y_center, width, height = map(float, parts)
                w_abs = int(width * img_width)
                h_abs = int(height * img_height)
                x_abs = int((x_center * img_width) - (w_abs / 2))
                y_abs = int((y_center * img_height) - (h_abs / 2))
                return [x_abs, y_abs, w_abs, h_abs]
    return None

# ==========================================
# 3. IMAGE PROCESSING PIPELINES
# ==========================================

def baseline_pipeline(image_path):
    """Baseline method: Passing raw image directly to OCR."""
    img = cv2.imread(image_path)
    if img is None:
        return ""
    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(img, config=config)
    return text.strip()

def proposed_pipeline(image_path, kernel_size=(15, 15)):
    """
    FINAL ALGORITHM: Black Hat -> Otsu -> Dilate -> Geo Filter -> Y-Align -> X-Cluster -> Macro Ratio -> Local Binarized OCR
    """
    img = cv2.imread(image_path)
    if img is None:
        return "", [], None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Morphological Black Hat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 2. Otsu Binarization & Dilation
    _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)
    
    # 3. Contour Extraction
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    unsorted_bboxes = []
    
    # STEP A: Micro Geometric Size and Shape Filtering
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w) if w > 0 else 0
        area = cv2.contourArea(cnt)
        solidity = area / float(w * h) if (w * h) > 0 else 0
        
        if 0.8 < aspect_ratio < 5.0 and 80 < area < 4000 and solidity > 0.30:
            unsorted_bboxes.append([x, y, w, h])

    # STEP B: Y-Axis Alignment (Rejects scattered noise)
    aligned_bboxes = []
    if len(unsorted_bboxes) > 0:
        median_y = np.median([b[1] + (b[3] / 2) for b in unsorted_bboxes])
        median_h = np.median([b[3] for b in unsorted_bboxes])
        
        for b in unsorted_bboxes:
            y_center = b[1] + (b[3] / 2)
            if abs(y_center - median_y) < (median_h * 0.8):
                aligned_bboxes.append(b)
    
    # STEP C: X-Axis Clustering
    sorted_bboxes = sorted(aligned_bboxes, key=lambda b: b[0])
    valid_groups = []
    
    if len(sorted_bboxes) > 0:
        med_w = np.median([b[2] for b in sorted_bboxes])
        current_group = [sorted_bboxes[0]]
        
        for i in range(1, len(sorted_bboxes)):
            prev_box = current_group[-1]
            curr_box = sorted_bboxes[i]
            gap = curr_box[0] - (prev_box[0] + prev_box[2])
            
            if gap < (med_w * 4.0):
                current_group.append(curr_box)
            else:
                valid_groups.append(current_group)
                current_group = [curr_box]
        valid_groups.append(current_group)
        
    filtered_groups = [g for g in valid_groups if len(g) >= 3]
    
    if not filtered_groups:
        return "", [], None 
        
    best_group = max(filtered_groups, key=len)
    
    # STEP D: Macro Aspect Ratio Filter (DESTROY TREES AND POLES)
    x_coords = [b[0] for b in best_group]
    y_coords = [b[1] for b in best_group]
    max_x_coords = [b[0] + b[2] for b in best_group]
    max_y_coords = [b[1] + b[3] for b in best_group]
    
    px = min(x_coords)
    py = min(y_coords)
    pw = max(max_x_coords) - px
    ph = max(max_y_coords) - py
    
    # If the overall box is a vertical rectangle (like a tree), reject it!
    macro_ratio = pw / float(ph) if ph > 0 else 0
    if macro_ratio < 1.5 or macro_ratio > 8.0:
        return "", [], None

    # Padding for OCR
    pad = 5
    px_pad = max(0, px - pad)
    py_pad = max(0, py - pad)
    pw_pad = min(gray.shape[1] - px_pad, pw + 2*pad)
    ph_pad = min(gray.shape[0] - py_pad, ph + 2*pad)
    
    final_plate_box = [px_pad, py_pad, pw_pad, ph_pad]
    
    # STEP E: Localized Binarized OCR (Fixes Shadow Blindness)
    roi_gray = gray[py_pad:py_pad+ph_pad, px_pad:px_pad+pw_pad]
    
    # Apply Otsu specifically to the cropped plate to force pure black text on white background
    _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Pad with white (255) pixels
    roi_padded = cv2.copyMakeBorder(roi_thresh, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255,255,255])
    
    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    plate_text = pytesseract.image_to_string(roi_padded, config=config).strip()
    plate_text = ''.join(e for e in plate_text if e.isalnum())

    return plate_text, best_group, final_plate_box

# ==========================================
# 4. EXPERIMENT LOOP & VISUALIZATION
# ==========================================

def run_experiment(dataset_path, correct_labels_dict=None):
    image_files = glob.glob(os.path.join(dataset_path, "*.jpg"))
    total_images = len(image_files)
    
    if total_images == 0:
        print(f"❌ ERROR: No .jpg files found in '{dataset_path}'")
        return

    baseline_cer_list = []
    proposed_cer_list = []
    correct_localizations = 0
    
    print(f" Final Academic Experiment Started. Processing {total_images} images...\n")
    print("ATTENTION: Images will pop up. Press any key on your keyboard to proceed to the next image!\n")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        ground_truth_text = correct_labels_dict.get(filename, "UNKNOWN") if correct_labels_dict else base_name.split('_')[0].upper()
            
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w, _ = img.shape
        display_img = img.copy() 
        
        txt_path = img_path.replace(".jpg", ".txt")
        real_plate_box = parse_yolo_label(txt_path, img_w, img_h)
        
        base_text = baseline_pipeline(img_path)
        prop_text, char_boxes, pred_plate_box = proposed_pipeline(img_path, kernel_size=(15, 15))
        
        baseline_cer_list.append(calculate_cer(base_text, ground_truth_text))
        proposed_cer_list.append(calculate_cer(prop_text, ground_truth_text))
        
        iou_score = 0.0
        
        # VISUALIZE GROUND TRUTH (BLUE)
        if real_plate_box is not None:
            rx, ry, rw, rh = real_plate_box
            cv2.rectangle(display_img, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)
            cv2.putText(display_img, "Ground Truth (YOLO)", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # VISUALIZE PREDICTIONS (GREEN AND ORANGE)
        if pred_plate_box is not None:
            px, py, pw, ph = pred_plate_box
            cv2.rectangle(display_img, (px, py), (px+pw, py+ph), (0, 255, 0), 2)
            cv2.putText(display_img, "Proposed Pipeline", (px, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            for cb in char_boxes:
                cx, cy, cw, ch = cb
                cv2.rectangle(display_img, (cx, cy), (cx+cw, cy+ch), (0, 165, 255), 1)

            if real_plate_box is not None:
                iou_score = calculate_iou(pred_plate_box, real_plate_box)
                if iou_score > 0.5:
                    correct_localizations += 1

        print(f"[{filename}] Ground Truth: {ground_truth_text} | Proposed OCR: '{prop_text}' | IoU: {iou_score:.2f}")

        # HUD DISPLAY OVERLAY
        cv2.rectangle(display_img, (0, 0), (700, 70), (0, 0, 0), -1)
        cv2.putText(display_img, f"Ground Truth: {ground_truth_text} | Output: {prop_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_img, f"Localization Score (IoU): {iou_score:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Algorithm Visual Debugger (Press Any Key to Continue)", display_img)
        cv2.waitKey(0) 

    cv2.destroyAllWindows() 

    loc_acc = (correct_localizations / total_images) * 100 if total_images > 0 else 0
    avg_base_cer = np.mean(baseline_cer_list) * 100 if baseline_cer_list else 100
    avg_prop_cer = np.mean(proposed_cer_list) * 100 if proposed_cer_list else 100

    print("\n" + "="*55)
    print("  EXPERIMENT RESULTS ")
    print("="*55)
    print(f"Total Images Evaluated           : {total_images}")
    print(f"Localization Accuracy (IoU > 0.5): {loc_acc:.2f}%")
    print(f"Baseline Tesseract CER           : {avg_base_cer:.2f}%")
    print(f"Proposed Algorithm CER           : {avg_prop_cer:.2f}%")
    print("="*55)

if __name__ == "__main__":
    dataset_folder = r"C:\Users\Yasar\Desktop\imagefinal\dataset"
    
    manual_ground_truth = {
        "814_jpg.rf.5dc4faceca5a5f0fc2d64081dfa3aca0.jpg": "42AJR028", 
        "820_jpg.rf.540a35fd2fdba45829510e740fc3cfad.jpg": "50ND202",
        "788_jpg.rf.e515070467bd836e4dee71c4645d4e58.jpg": "58DF351",
        "835_jpg.rf.ea96e479c2a025a6c62fc97e09166a50.jpg": "01EK217",
        "846_jpg.rf.8c81fe5cf4f14477da2c6932811e7a22.jpg": "66KE469", 
        "847_jpg.rf.3a6f1b879676370f38dd07ba598bacd5.jpg": "01BOA50",
        "855_jpg.rf.86c38065ff4ef111a9f6a2673717b617.jpg": "66AAH987",
        "867_jpg.rf.c75677a03124f25afe96662d1d136b78.jpg": "18AAT811",
        "781_jpg.rf.a639665383529ab0daeccd804a31d114.jpg": "66AAH987",
        "943_jpg.rf.bf8c4d9c026f6aaa1a0c2a62dfb724e3.jpg": "06BZL949"
    }
    
    run_experiment(dataset_folder, manual_ground_truth)