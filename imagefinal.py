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
    IMPROVED HYBRID ALGORITHM: Multiscale Sobel & Black Hat -> Aligned Character-based scoring -> Localized OCR
    """
    img = cv2.imread(image_path)
    if img is None:
        return "", [], None
    
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge detection using Sobel X
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    _, thresh_sobel = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 2. Black Hat
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
    _, thresh_bh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    candidates = []
    # Multiscale closing kernels
    scales = [9, 17, 29, 47, 65]
    for s_w in scales:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (s_w, 3))
        
        # Sobel close
        closed_sobel = cv2.morphologyEx(thresh_sobel, cv2.MORPH_CLOSE, kernel)
        contours_sobel, _ = cv2.findContours(closed_sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Blackhat close
        closed_bh = cv2.morphologyEx(thresh_bh, cv2.MORPH_CLOSE, kernel)
        contours_bh, _ = cv2.findContours(closed_bh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours_sobel + contours_bh:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect_ratio = w_box / float(h_box) if h_box > 0 else 0
            area = w_box * h_box
            if 1.5 < aspect_ratio < 7.0 and 400 < area < 70000:
                candidates.append([x, y, w_box, h_box])
                
    # Deduplicate candidates
    unique_candidates = []
    for c in candidates:
        is_dup = False
        for uc in unique_candidates:
            # Simple IoU overlap check
            xA = max(c[0], uc[0])
            yA = max(c[1], uc[1])
            xB = min(c[0] + c[2], uc[0] + uc[2])
            yB = min(c[1] + c[3], uc[1] + uc[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = c[2] * c[3]
            boxBArea = uc[2] * uc[3]
            iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
            if iou > 0.8:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(c)
            
    best_candidate = None
    best_score = -1
    best_roi_padded = None
    best_char_boxes = []
    
    for c in unique_candidates:
        cx, cy, cw, ch = c
        aligned_chars = []
        pad_x = int(cw * 0.05)
        pad_y = int(ch * 0.1)
        x1 = max(0, cx - pad_x)
        y1 = max(0, cy - pad_y)
        x2 = min(w, cx + cw + pad_x)
        y2 = min(h, cy + ch + pad_y)
        
        roi_gray = gray[y1:y2, x1:x2]
        if roi_gray.size == 0:
            continue
            
        # Local Black Hat thresholding for character extraction inside crop
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
            
            for b in char_cnts:
                cy_center = b[1] + b[3]/2
                if abs(cy_center - median_cy) < median_ch * 0.4:
                    # Convert coordinates back to global image coordinates
                    aligned_chars.append([b[0] + x1, b[1] + y1, b[2], b[3]])
                    
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
                score = -100
        else:
            score = -100
            
        if score > 0 and score > best_score:
            best_score = score
            best_candidate = c
            best_char_boxes = aligned_chars
            # Pad with white (255) pixels for OCR
            best_roi_padded = cv2.copyMakeBorder(roi_thresh, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            
    plate_text = ""
    if best_roi_padded is not None:
        config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        plate_text = pytesseract.image_to_string(best_roi_padded, config=config).strip()
        plate_text = ''.join(e for e in plate_text if e.isalnum())
        
    return plate_text, best_char_boxes, best_candidate

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
        "833_jpg.rf.18c9a4aad4ff1a3d8b381d126145c0c3.jpg": "34TE6244",
        "788_jpg.rf.e515070467bd836e4dee71c4645d4e58.jpg": "58DF351",
        "835_jpg.rf.ea96e479c2a025a6c62fc97e09166a50.jpg": "01EK217",
        "846_jpg.rf.8c81fe5cf4f14477da2c6932811e7a22.jpg": "66KE469", 
        "847_jpg.rf.3a6f1b879676370f38dd07ba598bacd5.jpg": "01BOA50",
        "855_jpg.rf.86c38065ff4ef111a9f6a2673717b617.jpg": "66AAH987",
        "867_jpg.rf.c75677a03124f25afe96662d1d136b78.jpg": "18AAT811",
        "781_jpg.rf.a639665383529ab0daeccd804a31d114.jpg": "66AAH987",
        "943_jpg.rf.bf8c4d9c026f6aaa1a0c2a62dfb724e3.jpg": "06BZL949",
        "1003_jpg.rf.78eaaaf563b861a4675b451c803201fc.jpg": "66ABA699",
        "1013_jpg.rf.b66a4f59d286bc10a86f890c5e4ec92f.jpg": "07KDF22",
        "1015_jpg.rf.7b06a5bf4d7ffe97cab0d7b755e5f826.jpg": "34RV0795",
        "1017_jpg.rf.603785f19fa306ae250d643a8b2b9e7b.jpg": "66KR787",
        "101_jpg.rf.cb65e35c8d629be7178ca2b4944adb42.jpg": "19SN241",
        "1022_jpg.rf.9bbc6bb10660af68a3c365a7599b96d6.jpg": "34YN3515",
        "1023_jpg.rf.12db799035bf5a1cc822138abbaa8664.jpg": "34KM0956",
        "1031_jpg.rf.f1a8c7a077abb43d56ff6781cd3195e2.jpg": "06DT6345",
        "1034_jpg.rf.209c84961f64410b5af7fcad28bc1a7c.jpg": "14BZ939",
        "1039_jpg.rf.99385f82465ad4dfed03dfcb581ae3b6.jpg": "16SCD40",
        "1048_jpg.rf.7e1c832dbcbf847b7c85984131ea5e1f.jpg": "55LY911",
        "1058_jpg.rf.10ba2cdc59348039e5b844a1cfbb328b.jpg": "16CUU21",
        "1061_jpg.rf.20d0529b35a958cf0a092132415bc763.jpg": "54EG064",
        "1064_jpg.rf.7dffc171b80a950fa8f353bb2d012c05.jpg": "42DHE25",
        "1069_jpg.rf.daaaa8f1ce0ecce22a65de8d1b34e15b.jpg": "06CIS18",
        "1072_jpg.rf.83d84c625aff59698b41ce37207a0c88.jpg": "16KE930",
        "1078_jpg.rf.068b55a2c2517118b5e634ef31f581e8.jpg": "05DM745",
        "1083_jpg.rf.2fe47b1a8f2404b5bdf44ec319ee8f1d.jpg": "34DG5102",
        "1084_jpg.rf.844ddff623f79f158c78516c0560efa2.jpg": "16FM491",
        "1086_jpg.rf.5e19e33717bad85aeb09f048f8390701.jpg": "23EA380",
        "1089_jpg.rf.dbb1647a7421d219f2d5d8a60eecd7ae.jpg": "34ZRG88",
        "1090_jpg.rf.a1b992f75f6f9d743fcb8c1a70a46f3a.jpg": "35PLS64",
        "1095_jpg.rf.d62c91c58f5641a0cd125e280c3afca0.jpg": "06NEB06",
        "1096_jpg.rf.cdde5b1f3e40ee52bad6cb870fcaee89.jpg": "48SN426",
        "1103_jpg.rf.42d7485e5977792137e3c07f91cbf4bd.jpg": "31AD873",
        "1105_jpg.rf.9d4ce96f55481693f36d4221086c38d2.jpg": "52TB650",
        "1108_jpg.rf.19f1f3bd3c3ebc492cc633e9b3d92cdb.jpg": "06ATB76",
        "1112_jpg.rf.28707f1bff841271319d5a21cfa35edb.jpg": "06AOL20",
        "1113_jpg.rf.80a9689c4e464ccdc973d1217a2ea321.jpg": "48V8174",
        "1124_jpg.rf.09610135c1978b735dfeaf3de5a99ee9.jpg": "35S0035",
        "1130_jpg.rf.d386eaf398330363fd848c1f7e810d7f.jpg": "34BBC129",
        "1132_jpg.rf.64fd884dacea581e05cc06c2c94a3649.jpg": "34NC9123",
        "1140_jpg.rf.4d3c988c8698caf60ef1d3ed75b8884e.jpg": "34J6306",
        "1142_jpg.rf.43a6bcfca40e9d308c1db2ddd7235265.jpg": "67DB417",
        "1143_jpg.rf.d34ccf79d9929bc2e774c02849e4854a.jpg": "34LY4851",
        "1148_jpg.rf.06a353b8e9aa25a0933d63fdd34831ee.jpg": "78DG147",
        "1149_jpg.rf.f9580c6a6d1d646d112db4b87f3a396d.jpg": "03UY028",
        "1152_jpg.rf.adc559799aafcfc6f33393da0f849455.jpg": "35DIG15",
        "1157_jpg.rf.74ca4f61a29e9bf62a7f5891aedd2c5d.jpg": "27EG364",
        "115_jpg.rf.8ad56780aa1c8ae3077265bc0f2fd560.jpg": "66AAY895",
        "1163_jpg.rf.34ba65f9a6bd7d4c9296695b5671d736.jpg": "27TU032",
        "1164_jpg.rf.1233bf2bf7cc66ec22fcfc7713c63ffa.jpg": "33NYF39",
        "1166_jpg.rf.7eb6871382a959a9fd3f208af8c43358.jpg": "42JN506",
        "1171_jpg.rf.20efaafb3f987f3194bb7db579cf55b4.jpg": "42EOV01",
        "1172_jpg.rf.0e4276d825447064e6f5640f92b7857d.jpg": "45UD879",
        "1177_jpg.rf.5a5d10d0717b4ae737d81f78c873bca2.jpg": "26EN559",
        "1179_jpg.rf.8fd9295afc1c4c8a0dc099e8b53bdbe7.jpg": "39SY105",
        "1182_jpg.rf.858e907a8133ca66dad0e345bb038287.jpg": "35KKL16",
        "1184_jpg.rf.6c4940ee1abd6c09ad73153d2d3b5a39.jpg": "06EU5892",
        "1186_jpg.rf.78047645f6d02e93f56c860e7d001b88.jpg": "46BU281",
        "1188_jpg.rf.3a2e9e688f1277fc6daeae03fd0378ed.jpg": "06AC5533",
        "1190_jpg.rf.a0f631da3dc4baf74a2d85b20725c56c.jpg": "50DV803",
        "1194_jpg.rf.9212918b34d841dcced26ed5390ed952.jpg": "21AAE048",
        "1196_jpg.rf.03d588c8e7dd46b0327d1e319fa4f879.jpg": "54KZ781",
        "1209_jpg.rf.942d1444fb46c739290d0a6974ece94e.jpg": "06LA656",
        "1211_jpg.rf.3ae9f37e85a7b86e5a4719d68cb6bd75.jpg": "34HT8553",
        "1216_jpg.rf.867372ab2b28d813594911204498b46c.jpg": "38TD649",
        "1219_jpg.rf.d4a8c556f73661d42b31083686f3eacc.jpg": "68AAD641",
        "1224_jpg.rf.cbaae940c2c8bd018d46b9af9594619b.jpg": "07VZ702",
        "1226_jpg.rf.c0f278c426f66c6a75f68e78a3354cfb.jpg": "34HZ6393",
        "1228_jpg.rf.d73ff78586a2a2a4d11244c7ecefff1e.jpg": "66AAD365",
        "1234_jpg.rf.342189ac891909e9cb33ca949a8ae491.jpg": "16YG032",
        "1243_jpg.rf.39e6b9d40a27fee6522bb26ceb3466ed.jpg": "27ACA50",
        "1244_jpg.rf.9e926ebf24519b5a3f092e9d5742baf7.jpg": "39TB552",
        "1259_jpg.rf.db87180fc317109d9598b7c057d8237b.jpg": "06EF3018",
        "125_jpg.rf.0ce03ae7ba434593921d5273ee13b252.jpg": "71FC102",
        "1263_jpg.rf.3ec7465e6430834a65e962fc9a603083.jpg": "06RPV47",
        "1273_jpg.rf.2f478ce0f504c6c126ae1b86bbf8fa9e.jpg": "01PJ457",
        "1279_jpg.rf.aaa5c105c64e0c7ad627391b5c4d033c.jpg": "06BAG855",
        "1286_jpg.rf.b0eaa8f51a7590c7947cf817e2035adc.jpg": "19ES210",
        "1295_jpg.rf.81e8f8025f75b0d96fc9ed823203c7eb.jpg": "66LJ564",
        "1306_jpg.rf.ad386ea28ea384b81939645105f7c11a.jpg": "06RJP16",
        "1307_jpg.rf.a7727f9b52d0625a90207e22d4b6c6bf.jpg": "06V9858",
        "1318_jpg.rf.1c3627fd151ed76ccdd4896543cfeb07.jpg": "35NCH13",
        "1319_jpg.rf.fcfabb9929a8b06c2f72c3ba5b2ddd70.jpg": "34ZAV50",
        "1321_jpg.rf.3933caadb5ab7645e75bf8ceb3e38870.jpg": "39TA649",
        "1322_jpg.rf.45fc14b32dec13a3e3ac825a3efce955.jpg": "34DNK05",
        "1323_jpg.rf.ae1441ee58cc27b424b5271e1092e90b.jpg": "06KA4312",
        "1326_jpg.rf.861f6b6696ac2608f1f7510e5c0e4a59.jpg": "31DL647",
        "1334_jpg.rf.ffc6e4654b2888f7fedbdb7a23fc3005.jpg": "27AD155",
        "1343_jpg.rf.b7eda2ad8e59cbb85822870f95cc2ed8.jpg": "27BKG93",
        "1361_jpg.rf.1a48ad5635a286026ee2dc376d082292.jpg": "34ESR77",
        "1363_jpg.rf.6de23f003670c23a8d4d3c605838d3e8.jpg": "06DB2185",
        "1369_jpg.rf.bb87883fc11b576b910e249e70c9eaac.jpg": "06FE1848",
        "1375_jpg.rf.8b9e08a7f80f10fd076b18f838b37832.jpg": "34RFA70",
        "1376_jpg.rf.88b382c9b41254184409946f8c9eb2e9.jpg": "60ET900",
        "1385_jpg.rf.162ed250cf08e6f3574bfe2e6918d99d.jpg": "31AYM58",
        "1386_jpg.rf.8c63f99b481846a7d2fec829ebf15707.jpg": "35AV9877",
        "1399_jpg.rf.54541120552d70c2d5e3c451ea2b71a2.jpg": "27BKJ90",
        "1400_jpg.rf.c077f86f46f3441784c6a965b1ff4a34.jpg": "68AAF915"
    }
    
    run_experiment(dataset_folder, manual_ground_truth)