import os
import glob
import cv2
import numpy as np

# We import the manual_ground_truth dictionary from imagefinal.py by executing it
import sys
sys.path.append(r"c:\Users\Yasar\Desktop\imagefinal")
with open(r"c:\Users\Yasar\Desktop\imagefinal\imagefinal.py", "r", encoding="utf-8") as f:
    code = f.read()
code = code.replace("run_experiment(dataset_folder, manual_ground_truth)", "")
local_vars = {"__name__": "__main__"}
exec(code, local_vars)
correct_labels_dict = local_vars['manual_ground_truth']

dataset_folder = r"C:\Users\Yasar\Desktop\imagefinal\dataset"
image_files = sorted(glob.glob(os.path.join(dataset_folder, "*.jpg")))

def parse_yolo_label(txt_path, img_width, img_height):
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

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

failed_images = []

for img_path in image_files:
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None: continue
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    txt_path = img_path.replace(".jpg", ".txt")
    real_plate_box = parse_yolo_label(txt_path, w, h)
    if not real_plate_box: continue
    
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    _, thresh_sobel = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
    _, thresh_bh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    candidates = []
    scales = [9, 17, 29, 47, 65]
    for s_w in scales:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (s_w, 3))
        closed_sobel = cv2.morphologyEx(thresh_sobel, cv2.MORPH_CLOSE, kernel)
        contours_sobel, _ = cv2.findContours(closed_sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        closed_bh = cv2.morphologyEx(thresh_bh, cv2.MORPH_CLOSE, kernel)
        contours_bh, _ = cv2.findContours(closed_bh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours_sobel + contours_bh:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect_ratio = w_box / float(h_box) if h_box > 0 else 0
            area = w_box * h_box
            if 1.5 < aspect_ratio < 7.0 and 400 < area < 70000:
                candidates.append([x, y, w_box, h_box])
                
    unique_candidates = []
    for c in candidates:
        is_dup = False
        for uc in unique_candidates:
            if calculate_iou(c, uc) > 0.8:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(c)
            
    # Check if the real plate was generated as a candidate
    has_real_candidate = False
    best_candidate_iou = 0
    for c in unique_candidates:
        iou = calculate_iou(c, real_plate_box)
        if iou > 0.5:
            has_real_candidate = True
        if iou > best_candidate_iou:
            best_candidate_iou = iou
            
    # Score
    best_candidate = None
    best_score = -1
    for c in unique_candidates:
        cx, cy, cw, ch = c
        pad_x = int(cw * 0.05)
        pad_y = int(ch * 0.1)
        x1 = max(0, cx - pad_x)
        y1 = max(0, cy - pad_y)
        x2 = min(w, cx + cw + pad_x)
        y2 = min(h, cy + ch + pad_y)
        
        roi_gray = gray[y1:y2, x1:x2]
        if roi_gray.size == 0: continue
        
        roi_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        roi_blackhat = cv2.morphologyEx(roi_gray, cv2.MORPH_BLACKHAT, roi_kernel)
        _, roi_thresh = cv2.threshold(roi_blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        roi_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        char_cnts = []
        roi_h = y2 - y1
        for r_cnt in roi_contours:
            rx, ry, rw, rh = cv2.boundingRect(r_cnt)
            r_aspect = rh / float(rw) if rw > 0 else 0
            if 0.8 < r_aspect < 5.0 and 0.3 * roi_h < rh < 0.95 * roi_h:
                char_cnts.append([rx, ry, rw, rh])
                
        if len(char_cnts) >= 3:
            char_cnts = sorted(char_cnts, key=lambda b: b[0])
            median_cy = np.median([b[1] + b[3]/2 for b in char_cnts])
            median_ch = np.median([b[3] for b in char_cnts])
            
            aligned_chars = []
            for b in char_cnts:
                cy_center = b[1] + b[3]/2
                if abs(cy_center - median_cy) < median_ch * 0.4:
                    aligned_chars.append(b)
            n_chars = len(aligned_chars)
            if n_chars >= 3:
                heights = [b[3] for b in aligned_chars]
                std_h = np.std(heights)
                mean_h = np.mean(heights)
                h_uniformity = 1.0 - (std_h / mean_h) if mean_h > 0 else 0
                crop_aspect = cw / float(ch) if ch > 0 else 0
                aspect_penalty = abs(crop_aspect - 4.5)
                score = n_chars * 15.0 + h_uniformity * 30.0 - aspect_penalty * 5.0
            else:
                score = 0
        else:
            score = 0
        if score > best_score:
            best_score = score
            best_candidate = c
            
    final_iou = 0.0
    if best_candidate:
        final_iou = calculate_iou(best_candidate, real_plate_box)
        
    if final_iou < 0.5:
        failed_images.append((filename, has_real_candidate, best_candidate_iou, len(unique_candidates)))

print(f"Failed to localize {len(failed_images)} images:")
for fn, has_cand, best_iou, n_cands in failed_images[:15]:
    print(f"  {fn} -> Candidate found? {has_cand} (Best candidate IoU: {best_iou:.2f}) | Total cands: {n_cands}")
