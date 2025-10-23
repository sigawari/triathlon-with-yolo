from ultralytics import YOLO
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Konfigurasi
# =========================
input_folder = "test_bike"
file_name = "test10.png"  # gambar berisi sepeda + ArUco
aruco_image_path = os.path.join(input_folder, file_name)

# Ukuran fisik sisi marker ArUco (cm) - ganti sesuai marker yang dicetak
MARKER_LENGTH_CM = 5.0

# Kamus ArUco (6x6)
ARUCO_DICT = cv2.aruco.DICT_6X6_250

# Output
output_folder = "predict_from_images"
os.makedirs(output_folder, exist_ok=True)

# Model YOLO
model = YOLO("best-960.pt")

# Mapping class
class_map = {
    0: "saddle",
    1: "f_tire",
    2: "crank",
    3: "r_tire"
}

print("[STEP] Baca gambar:", aruco_image_path)
img = cv2.imread(aruco_image_path)
if img is None:
    raise FileNotFoundError(f"Gambar tidak ditemukan: {aruco_image_path}")

# =========================
# Kalibrasi ArUco di gambar yang sama
# =========================
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

corners, ids, _ = detector.detectMarkers(img)
if ids is None or len(ids) == 0:
    raise RuntimeError("Tidak menemukan ArUco 6x6 pada gambar. Pastikan marker terlihat dengan jelas.")

# Ambil marker pertama
c = corners[0][0]  # (4,2)
edges = [
    np.linalg.norm(c[0] - c[1]),
    np.linalg.norm(c[1] - c[2]),
    np.linalg.norm(c[2] - c[3]),
    np.linalg.norm(c[3] - c[0]),
]
avg_side_px = float(np.mean(edges))
if avg_side_px <= 0:
    raise RuntimeError("Edge length ArUco tidak valid.")

scale_cm_per_px = MARKER_LENGTH_CM / avg_side_px
print(f"[CALIB] avg_side_px = {avg_side_px:.2f} px")
print(f"[CALIB] scale_cm_per_px = {scale_cm_per_px:.6f} cm/px")

# Visualisasi ArUco di gambar
c_int = c.astype(int)
cv2.polylines(img, [c_int], True, (0, 255, 255), 3)
cx, cy = int(c[:,0].mean()), int(c[:,1].mean())
cv2.putText(img, f"Scale: {scale_cm_per_px:.4f} cm/px (Aruco)",
            (max(10, cx - 150), max(30, cy - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

# =========================
# Prediksi YOLO di gambar yang sama
# =========================
results = model.predict(
    source=img,
    show=False,
    save=False,
    conf=0.7,
    line_width=2,
    classes=[0, 1, 2, 3]
)

bboxes = {}
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy()
    for box, ccls in zip(boxes, cls):
        label = class_map.get(int(ccls), str(ccls))
        bboxes[label] = box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(f"[DETECT] {label}: ({x1},{y1})-({x2},{y2})")

# =========================
# Pengukuran memakai scale_cm_per_px
# =========================
if scale_cm_per_px is not None:
    # Wheelbase
    if "f_tire" in bboxes and "r_tire" in bboxes:
        x1_f, y1_f, x2_f, y2_f = bboxes["f_tire"]
        x1_r, y1_r, x2_r, y2_r = bboxes["r_tire"]
        cf = (int((x1_f + x2_f) / 2), int((y1_f + y2_f) / 2))
        cr = (int((x1_r + x2_r) / 2), int((y1_r + y2_r) / 2))
        import math
        wheelbase_px = math.hypot(cf[0] - cr[0], cf[1] - cr[1])
        wheelbase_cm = wheelbase_px * scale_cm_per_px
        print(f"Wheelbase: {wheelbase_cm:.1f} cm")
        cv2.line(img, cf, cr, (255, 0, 0), 4)
        cv2.putText(img, f"Wheelbase: {wheelbase_cm:.1f} cm",
                    (cr[0], cr[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Orientasi
        orientation = "Right-facing" if cf[0] > cr[0] else "Left-facing"
        cv2.putText(img, f"Orientation: {orientation}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 200), 2)
        print(f"Orientation: {orientation}")

        # Panjang total
        if orientation == "Right-facing":
            front = int(max(x1_f, x2_f))
            rear = int(min(x1_r, x2_r))
        else:
            front = int(min(x1_f, x2_f))
            rear = int(max(x1_r, x2_r))
        y_base = int(max(y2_f, y2_r)) + 50
        length_px = abs(rear - front)
        length_cm = length_px * scale_cm_per_px
        print(f"Length: {length_cm:.1f} cm")
        if length_cm <= 185:
            print("Length OK")
        else:
            print("The bike is not legal")
        cv2.line(img, (front, y_base), (rear, y_base), (0, 0, 255), 4)
        cv2.putText(img, f"Length: {length_cm:.1f} cm",
                    (front, y_base - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Crank -> sumbu vertikal ban depan
    if "crank" in bboxes and "f_tire" in bboxes:
        x1_c, y1_c, x2_c, y2_c = bboxes["crank"]
        crank_center = (int((x1_c + x2_c) / 2), int((y1_c + y2_c) / 2))
        x1_f, y1_f, x2_f, y2_f = bboxes["f_tire"]
        center_f = (int((x1_f + x2_f) / 2), int((y1_f + y2_f) / 2))
        vertical_x = center_f[0]
        dist_px = abs(crank_center[0] - vertical_x)
        dist_cm = dist_px * scale_cm_per_px
        print(f"Crank->F_tire axis: {dist_cm:.1f} cm")
        if 54 <= dist_cm <= 65:
            print("Crank to F_Tire OK")
        else:
            print("The bike is not legal")
        cv2.line(img, crank_center, (vertical_x, crank_center[1]), (0, 165, 255), 4)
        cv2.putText(img, f"Crank->F_tire axis: {dist_cm:.1f} cm",
                    (crank_center[0], crank_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        cv2.line(img, (vertical_x, 0), (vertical_x, img.shape[0]), (200, 200, 200), 2)

    # Crank -> ground (aproksimasi)
    if "crank" in bboxes and "f_tire" in bboxes:
        x1_c, y1_c, x2_c, y2_c = bboxes["crank"]
        crank_center = (int((x1_c + x2_c) / 2), int((y1_c + y2_c) / 2))
        x1_f, y1_f, x2_f, y2_f = bboxes["f_tire"]
        ground_y = int(max(y2_f, y2_f))
        dist_px = abs(crank_center[1] - ground_y)
        dist_cm = dist_px * scale_cm_per_px
        print(f"Crank->Ground: {dist_cm:.1f} cm")
        if 24 <= dist_cm <= 30:
            print("Crank to Ground OK")
        else:
            print("The bike is not legal")
        cv2.line(img, crank_center, (crank_center[0], ground_y), (0, 100, 255), 4)

    # Diameter ban belakang
    if "r_tire" in bboxes:
        x1_r, y1_r, x2_r, y2_r = bboxes["r_tire"]
        width_r = abs(x2_r - x1_r)
        height_r = abs(y2_r - y1_r)
        diameter_r_px = (width_r + height_r) / 2
        diameter_r_cm = diameter_r_px * scale_cm_per_px
        print(f"R_tire diameter: {diameter_r_cm:.1f} cm")
        center_r = (int((x1_r + x2_r) / 2), int((y1_r + y2_r) / 2))
        cv2.putText(img, f"R_tire dia: {diameter_r_cm:.1f} cm",
                    (center_r[0], center_r[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.circle(img, center_r, int(diameter_r_px/2), (0, 255, 255), 3)

# Simpan hasil anotasi
out_file = os.path.join(output_folder, f"output_{file_name}")
cv2.imwrite(out_file, img)
print(f"[INFO] Hasil anotasi disimpan: {out_file}")

# =========================
# Tampilkan dengan Matplotlib
# =========================
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # konversi untuk plt.imshow
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.axis('off')
plt.tight_layout()
plt.show()  # letakkan terakhir agar semua print sudah tampil
