"""
Creates figure images for the IEEE report:
  Fig 2 - Sample dataset images (6 examples with GT labels and bounding boxes drawn)
  Fig 3 - Processing stages (pipeline visualization for one image)
  Fig 4 - Success cases (3 images with predicted boxes and OCR output)
  Fig 5 - Failure cases (3 images showing why localization failed)
  Fig 6 - CER distribution histogram
"""

import cv2
import numpy as np
import os
import glob

DATASET   = r"C:\Users\Yasar\Desktop\imagefinal\dataset"
ARTIFACT  = r"C:\Users\Yasar\.gemini\antigravity\brain\c70e79db-84f8-4c74-aa09-c43062f30048"

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
    "1400_jpg.rf.c077f86f46f3441784c6a965b1ff4a34.jpg": "68AAF915",
}

def parse_yolo(txt_path, img_w, img_h):
    if not os.path.exists(txt_path):
        return None
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, xc, yc, w, h = map(float, parts)
                W = int(w * img_w); H = int(h * img_h)
                X = int(xc * img_w - W/2); Y = int(yc * img_h - H/2)
                return (X, Y, W, H)
    return None

def add_label(img, text, x, y, color=(0,255,0)):
    """Draw text with black outline for visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 2
    cv2.putText(img, text, (x, y), font, scale, (0,0,0), thick+2)
    cv2.putText(img, text, (x, y), font, scale, color, thick)

def resize_to_height(img, H):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * H / h), H))

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Sample Dataset Images (2 rows × 3 cols, GT box + label)
# ─────────────────────────────────────────────────────────────────────────────
sample_keys = [
    "1022_jpg.rf.9bbc6bb10660af68a3c365a7599b96d6.jpg",
    "1058_jpg.rf.10ba2cdc59348039e5b844a1cfbb328b.jpg",
    "1083_jpg.rf.2fe47b1a8f2404b5bdf44ec319ee8f1d.jpg",
    "1143_jpg.rf.d34ccf79d9929bc2e774c02849e4854a.jpg",
    "1219_jpg.rf.d4a8c556f73661d42b31083686f3eacc.jpg",
    "1326_jpg.rf.861f6b6696ac2608f1f7510e5c0e4a59.jpg",
]

ROW_H = 200
cells = []
for key in sample_keys:
    path = os.path.join(DATASET, key)
    gt   = manual_ground_truth.get(key, "?")
    img  = cv2.imread(path)
    if img is None:
        img = np.ones((ROW_H, 320, 3), dtype=np.uint8) * 180
    txt_path = path.replace(".jpg", ".txt")
    box = parse_yolo(txt_path, img.shape[1], img.shape[0])
    if box:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 60, 60), 3)
        add_label(img, "GT: " + gt, x, max(y-8, 18), (255, 60, 60))
    img = resize_to_height(img, ROW_H)
    # Add white bottom bar with GT text
    bar = np.ones((28, img.shape[1], 3), dtype=np.uint8) * 245
    cv2.putText(bar, "GT: " + gt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30,30,30), 2)
    img = np.vstack([img, bar])
    cells.append(img)

# Pad all to same width
max_w = max(c.shape[1] for c in cells)
padded = [np.hstack([c, np.ones((c.shape[0], max_w - c.shape[1], 3), dtype=np.uint8)*245]) for c in cells]
row1 = np.hstack(padded[:3])
row2 = np.hstack(padded[3:])
sep  = np.ones((6, row1.shape[1], 3), dtype=np.uint8) * 245
fig2 = np.vstack([row1, sep, row2])
fig2_path = os.path.join(ARTIFACT, "fig2_sample_images.png")
cv2.imwrite(fig2_path, fig2)
print("Fig 2 saved:", fig2_path)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Processing Pipeline Stages (one image, 4 panels)
# ─────────────────────────────────────────────────────────────────────────────
demo_key  = "1022_jpg.rf.9bbc6bb10660af68a3c365a7599b96d6.jpg"
demo_path = os.path.join(DATASET, demo_key)
orig      = cv2.imread(demo_path)

PANEL_H = 200
PANEL_W = int(orig.shape[1] * PANEL_H / orig.shape[0])

gray      = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
sobel     = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
_, bs     = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
bh        = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
_, bbh    = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Merge for visualization
merged = cv2.bitwise_or(bs, bbh)
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (29, 3))
closed = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel_close)

# Panel 1: original
p1 = cv2.resize(orig, (PANEL_W, PANEL_H))
# Panel 2: Sobel-X binary
p2 = cv2.resize(cv2.cvtColor(bs, cv2.COLOR_GRAY2BGR), (PANEL_W, PANEL_H))
# Panel 3: Black Hat binary
p3 = cv2.resize(cv2.cvtColor(bbh, cv2.COLOR_GRAY2BGR), (PANEL_W, PANEL_H))
# Panel 4: after closing (merged)
p4 = cv2.resize(cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR), (PANEL_W, PANEL_H))

def add_panel_label(panel, text):
    bar = np.ones((26, panel.shape[1], 3), dtype=np.uint8) * 40
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
    return np.vstack([panel, bar])

p1 = add_panel_label(p1, "(a) Original Input")
p2 = add_panel_label(p2, "(b) Sobel-X Binary (Bs)")
p3 = add_panel_label(p3, "(c) Black Hat Binary (Bbh)")
p4 = add_panel_label(p4, "(d) Merged + Closing (k=29)")

sep3 = np.ones((p1.shape[0], 4, 3), dtype=np.uint8) * 245
fig3 = np.hstack([p1, sep3, p2, sep3, p3, sep3, p4])
fig3_path = os.path.join(ARTIFACT, "fig3_pipeline_stages.png")
cv2.imwrite(fig3_path, fig3)
print("Fig 3 saved:", fig3_path)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — CER distribution bar chart
# ─────────────────────────────────────────────────────────────────────────────
import Levenshtein as lev
import sys
sys.path.insert(0, r"C:\Users\Yasar\Desktop\imagefinal")
from imagefinal import proposed_pipeline, baseline_pipeline

cer_proposed = []
cer_baseline = []
all_files    = glob.glob(os.path.join(DATASET, "*.jpg"))

print("Computing CER for all images...")
for img_path in all_files:
    fn = os.path.basename(img_path)
    gt = manual_ground_truth.get(fn, None)
    if gt is None:
        continue
    gt = gt.strip().upper()
    
    base_text, _, _ = proposed_pipeline(img_path) if False else ("", None, None)
    prop_text, _, _ = proposed_pipeline(img_path)
    base_text       = baseline_pipeline(img_path)
    
    def cer(p, g):
        p = p.strip().upper(); g = g.strip().upper()
        return min(lev.distance(p, g) / len(g), 2.0) if g else 1.0
    
    cer_proposed.append(cer(prop_text, gt))
    cer_baseline.append(cer(base_text, gt))

# Create bar chart with matplotlib-free approach using OpenCV
bins     = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
bin_labels = ["0-25%","25-50%","50-75%","75-100%","100-150%","150-200%+"]

def count_bins(data, bins):
    counts = []
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        counts.append(sum(1 for v in data if lo <= v < hi))
    return counts

prop_counts = count_bins(cer_proposed, bins)
base_counts = count_bins(cer_baseline, bins)

# Draw chart on canvas
CW = 700; CH = 320
chart = np.ones((CH, CW, 3), dtype=np.uint8) * 255
n = len(bin_labels)
bar_w   = 50
gap     = 18
group_w = 2 * bar_w + gap
margin_l = 60; margin_b = 60
max_count = max(max(prop_counts), max(base_counts), 1)
avail_h   = CH - margin_b - 20

colors = [(52, 152, 219), (231, 76, 60)]  # blue = proposed, red = baseline

for i, (pc, bc) in enumerate(zip(prop_counts, base_counts)):
    x0 = margin_l + i * (group_w + 15)
    
    # Proposed bar (blue)
    ph = int(pc / max_count * avail_h)
    y1 = CH - margin_b - ph
    cv2.rectangle(chart, (x0, y1), (x0 + bar_w, CH - margin_b), colors[0], -1)
    cv2.rectangle(chart, (x0, y1), (x0 + bar_w, CH - margin_b), (30,30,30), 1)
    if pc > 0:
        cv2.putText(chart, str(pc), (x0 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20,20,20), 1)
    
    # Baseline bar (red)
    bh2 = int(bc / max_count * avail_h)
    y2  = CH - margin_b - bh2
    x0b = x0 + bar_w + 2
    cv2.rectangle(chart, (x0b, y2), (x0b + bar_w, CH - margin_b), colors[1], -1)
    cv2.rectangle(chart, (x0b, y2), (x0b + bar_w, CH - margin_b), (30,30,30), 1)
    if bc > 0:
        cv2.putText(chart, str(bc), (x0b + 4, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20,20,20), 1)
    
    # X label
    cv2.putText(chart, bin_labels[i], (x0 - 5, CH - margin_b + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30,30,30), 1)

# Axes
cv2.line(chart, (margin_l - 5, 15), (margin_l - 5, CH - margin_b), (0,0,0), 2)
cv2.line(chart, (margin_l - 5, CH - margin_b), (CW - 10, CH - margin_b), (0,0,0), 2)
cv2.putText(chart, "Count", (2, CH//2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

# Y grid lines
for frac in [0.25, 0.5, 0.75, 1.0]:
    y = CH - margin_b - int(frac * avail_h)
    cv2.line(chart, (margin_l - 5, y), (CW - 10, y), (200,200,200), 1)
    cv2.putText(chart, str(int(frac * max_count)), (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1)

# Legend
cv2.rectangle(chart, (CW-200, 18), (CW-185, 33), colors[0], -1)
cv2.putText(chart, "Proposed", (CW-180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
cv2.rectangle(chart, (CW-200, 42), (CW-185, 57), colors[1], -1)
cv2.putText(chart, "Baseline", (CW-180, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

fig4_path = os.path.join(ARTIFACT, "fig4_cer_distribution.png")
cv2.imwrite(fig4_path, chart)
print("Fig 4 saved:", fig4_path)
print("Proposed CER counts:", prop_counts)
print("Baseline CER counts:", base_counts)
