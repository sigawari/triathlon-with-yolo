from ultralytics import YOLO
import cv2
import os
import numpy as np
import time
import math

# =========================
# Konfigurasi
# =========================
MARKER_LENGTH_CM = 5.0
ARUCO_DICT = cv2.aruco.DICT_6X6_250

# Model YOLO
model = YOLO("best-960.pt")

output_folder = "predict_camera"
os.makedirs(output_folder, exist_ok=True)

# Kamera
cap = cv2.VideoCapture(0)
# Lebar/tinggi capture tinggi untuk tampilan, tapi inferensi akan di-downscale
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

class_map = {0: "saddle", 1: "f_tire", 2: "crank", 3: "r_tire"}

print("Press 'q' to quit, 's' to save current frame")

# State
frame_count = 0
saved_count = 0
scale_cm_per_px = None
calibrated = False
calib_stability_counter = 0
CALIB_STABILITY_N = 8
last_scale_value = None

# Interval inferensi YOLO agar FPS naik
YOLO_INTERVAL = 4  # lakukan prediksi setiap 4 frame
last_bboxes = {}   # cache bbox dari prediksi terakhir untuk digambar di frame antara
last_metrics = {}  # cache metrik terakhir untuk overlay stabil

# Ukuran downscale untuk inferensi (lebih kecil = lebih cepat)
infer_w, infer_h = 960, 540

# Detektor ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Smoothing sederhana (EMA) untuk panjang
ema_length = None
ALPHA = 0.3  # 0.0-1.0; makin kecil makin halus

def emavg(prev, new, alpha=ALPHA):
    return new if prev is None else (alpha * new + (1 - alpha) * prev)

def infer_yolo_with_downscale(img_bgr):
    h, w = img_bgr.shape[:2]
    # Resize untuk inferensi
    img_small = cv2.resize(img_bgr, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
    scale_x = w / infer_w
    scale_y = h / infer_h
    results = model.predict(
        source=img_small,
        show=False,
        save=False,
        conf=0.7,
        line_width=2,
        classes=[0, 1, 2, 3],
        imgsz=max(infer_w, infer_h)  # bantu hint ukuran
    )
    bboxes_scaled = {}
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        for box, ccls in zip(boxes, cls):
            x1_s, y1_s, x2_s, y2_s = box
            # Skala balik ke ukuran frame asli
            x1 = int(x1_s * scale_x); y1 = int(y1_s * scale_y)
            x2 = int(x2_s * scale_x); y2 = int(y2_s * scale_y)
            label = class_map.get(int(ccls), str(ccls))
            bboxes_scaled[label] = np.array([x1, y1, x2, y2], dtype=np.int32)
    return bboxes_scaled

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to capture frame from camera")
        break

    h, w = img.shape[:2]

    if not calibrated:
        # MODE KALIBRASI: ArUco
        corners, ids, _ = detector.detectMarkers(img)
        if ids is not None and len(ids) > 0:
            c = corners[0][0]
            edges = [
                np.linalg.norm(c[0] - c[1]),
                np.linalg.norm(c[1] - c[2]),
                np.linalg.norm(c[2] - c[3]),
                np.linalg.norm(c[3] - c[0]),
            ]
            avg_side_px = float(np.mean(edges))
            if avg_side_px > 0:
                current_scale = MARKER_LENGTH_CM / avg_side_px
                if last_scale_value is None or abs(current_scale - last_scale_value) < (0.02 * (last_scale_value or current_scale)):
                    calib_stability_counter += 1
                else:
                    calib_stability_counter = 0
                last_scale_value = current_scale

                c_int = c.astype(int)
                cv2.polylines(img, [c_int], True, (0, 255, 255), 2)
                cx = int(c[:,0].mean()); cy = int(c[:,1].mean())
                cv2.putText(img, f"ArUco scale: {current_scale:.4f} cm/px",
                            (cx, max(30, cy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                print(f"[CALIB] avg_side_px={avg_side_px:.1f} -> scale={current_scale:.6f} ({calib_stability_counter}/{CALIB_STABILITY_N})")

                if calib_stability_counter >= CALIB_STABILITY_N:
                    scale_cm_per_px = current_scale
                    calibrated = True
                    print(f"[CALIB] Locked scale_cm_per_px = {scale_cm_per_px:.6f} cm/px")
            else:
                print("[CALIB] Invalid edge length")
        else:
            # Jangan spam terminal tiap frame; print per sekian frame
            if frame_count % 15 == 0:
                print("[CALIB] No ArUco detected. Show a 6x6 marker in view.")

        cv2.putText(img, "Calibration mode: show 6x6 ArUco",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 220, 220), 2)

    else:
        # MODE PENGUKURAN: jalankan YOLO periodik
        run_yolo = (frame_count % YOLO_INTERVAL == 0)
        if run_yolo:
            bboxes = infer_yolo_with_downscale(img)
            last_bboxes = bboxes
        else:
            bboxes = last_bboxes  # pakai hasil terakhir untuk render cepat

        # Gambar bbox (ringan)
        for label, box in bboxes.items():
            x1, y1, x2, y2 = box.tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if run_yolo:
                print(f"[DETECT] {label}: ({x1},{y1})-({x2},{y2})")

        # Hitung metrik hanya saat run_yolo agar print tidak membanjiri
        if scale_cm_per_px is not None and bboxes:
            metrics = {}

            # Wheelbase
            if "f_tire" in bboxes and "r_tire" in bboxes:
                x1_f, y1_f, x2_f, y2_f = bboxes["f_tire"]
                x1_r, y1_r, x2_r, y2_r = bboxes["r_tire"]
                cf = (int((x1_f + x2_f) / 2), int((y1_f + y2_f) / 2))
                cr = (int((x1_r + x2_r) / 2), int((y1_r + y2_r) / 2))
                wheelbase_px = math.hypot(cf[0] - cr[0], cf[1] - cr[1])
                wheelbase_cm = wheelbase_px * scale_cm_per_px
                metrics["wheelbase_cm"] = wheelbase_cm
                # Orientasi
                orientation = "Right-facing" if cf[0] > cr[0] else "Left-facing"
                metrics["orientation"] = orientation

                # Panjang total (proyeksi x)
                if orientation == "Right-facing":
                    front = int(max(x1_f, x2_f))
                    rear  = int(min(x1_r, x2_r))
                else:
                    front = int(min(x1_f, x2_f))
                    rear  = int(max(x1_r, x2_r))
                length_px = abs(rear - front)
                length_cm = length_px * scale_cm_per_px
                # Smoothing agar tidak jitter
                ema_length = emavg(ema_length, length_cm)
                metrics["length_cm"] = length_cm
                metrics["length_cm_smooth"] = ema_length

                # Overlay ringan
                y_base = int(max(y2_f, y2_r)) + 40
                cv2.line(img, (front, y_base), (rear, y_base), (0, 0, 255), 3)
                cv2.putText(img, f"Length: {ema_length:.1f} cm", (front, y_base - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(img, f"Wheelbase: {wheelbase_cm:.1f} cm", (cr[0], max(20, cr[1] - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                cv2.putText(img, f"Orientation: {orientation}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,200), 2)

                if run_yolo:
                    print(f"Wheelbase: {wheelbase_cm:.1f} cm")
                    print(f"Length(raw): {length_cm:.1f} cm | Length(smooth): {ema_length:.1f} cm")
                    if length_cm <= 185:
                        print("Length OK")
                    else:
                        print("The bike is not legal")

            # Crank -> F_tire axis
            if "crank" in bboxes and "f_tire" in bboxes:
                x1_c, y1_c, x2_c, y2_c = bboxes["crank"]
                crank_center = (int((x1_c + x2_c) / 2), int((y1_c + y2_c) / 2))
                x1_f, y1_f, x2_f, y2_f = bboxes["f_tire"]
                center_f = (int((x1_f + x2_f) / 2), int((y1_f + y2_f) / 2))
                vertical_x = center_f[0]
                dist_px = abs(crank_center[0] - vertical_x)
                dist_cm = dist_px * scale_cm_per_px
                cv2.line(img, crank_center, (vertical_x, crank_center[1]), (0,165,255), 3)
                cv2.putText(img, f"Crank->F_tire: {dist_cm:.1f} cm",
                            (crank_center[0], max(20, crank_center[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,165,255), 2)
                if run_yolo:
                    print(f"Crank->F_tire axis: {dist_cm:.1f} cm")
                    print("Crank to F_Tire OK" if 54 <= dist_cm <= 65 else "The bike is not legal")

            # Crank -> Ground
            if "crank" in bboxes and "f_tire" in bboxes:
                x1_c, y1_c, x2_c, y2_c = bboxes["crank"]
                crank_center = (int((x1_c + x2_c) / 2), int((y1_c + y2_c) / 2))
                x1_f, y1_f, x2_f, y2_f = bboxes["f_tire"]
                ground_y = int(max(y2_f, y2_f))
                dist_px = abs(crank_center[1] - ground_y)
                dist_cm = dist_px * scale_cm_per_px
                cv2.line(img, crank_center, (crank_center[0], ground_y), (0,100,255), 3)
                if run_yolo:
                    print(f"Crank->Ground: {dist_cm:.1f} cm")
                    print("Crank to Ground OK" if 24 <= dist_cm <= 30 else "The bike is not legal")

            # R_tire diameter
            if "r_tire" in bboxes:
                x1_r, y1_r, x2_r, y2_r = bboxes["r_tire"]
                width_r = abs(x2_r - x1_r); height_r = abs(y2_r - y1_r)
                diameter_r_px = (width_r + height_r) / 2
                diameter_r_cm = diameter_r_px * scale_cm_per_px
                center_r = (int((x1_r + x2_r) / 2), int((y1_r + y2_r) / 2))
                cv2.putText(img, f"R_tire dia: {diameter_r_cm:.1f} cm",
                            (center_r[0], center_r[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
                cv2.circle(img, center_r, int(diameter_r_px/2), (0,255,255), 2)
                if run_yolo:
                    print(f"R_tire diameter: {diameter_r_cm:.1f} cm")

            last_metrics = metrics

        # Tampilkan status skala
        if scale_cm_per_px is not None:
            cv2.putText(img, f"Scale: {scale_cm_per_px:.4f} cm/px (Aruco)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,0), 2)

    # Tampilkan frame
    cv2.imshow('Bike Measurement - Live Camera', img)

    # Keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        out_path = os.path.join(output_folder, f"camera_frame_{saved_count:04d}.jpg")
        cv2.imwrite(out_path, img)
        print(f"[INFO] Frame saved to {out_path}")
        saved_count += 1

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed.")