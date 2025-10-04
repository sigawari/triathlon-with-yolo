from ultralytics import YOLO
import cv2
import os

# Muat model
model = YOLO("best.pt")

# Skala wajib (ganti sesuai kalibrasi nyata)
scale_cm_per_px = 0.261  # contoh: 1 px = 0.05 cm

# Tentukan path gambar menggunakan os
source_image_path = os.path.join("test_bike", "test5.jpeg")

# Jalankan prediksi
results = model.predict(
    source=source_image_path,
    show=False,
    save=False,
    conf=0.7,
    line_width=2,
    classes=[0, 1, 2, 3]  # hanya objek relevan
)

# Mapping class
class_map = {
    0: "saddle",
    1: "f_tire",
    2: "crank",
    3: "r_tire"
}

# Baca gambar asli
img = cv2.imread(os.path.join("test_bike", "test5.jpeg"))

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy()

    bboxes = {}
    for box, c in zip(boxes, cls):
        label = class_map.get(int(c), str(c))
        bboxes[label] = box

        # --- Gambar bounding box ---
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, label, (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # --- Wheelbase ---
    if "f_tire" in bboxes and "r_tire" in bboxes:
        x1_f, y1_f, x2_f, y2_f = bboxes["f_tire"]
        x1_r, y1_r, x2_r, y2_r = bboxes["r_tire"]

        center_f = (int((x1_f + x2_f) / 2), int((y1_f + y2_f) / 2))
        center_r = (int((x1_r + x2_r) / 2), int((y1_r + y2_r) / 2))

        wheelbase_px = ((center_f[0] - center_r[0]) ** 2 + (center_f[1] - center_r[1]) ** 2) ** 0.5
        wheelbase_cm = wheelbase_px * scale_cm_per_px

        cv2.line(img, center_f, center_r, (255, 0, 0), 4)
        cv2.putText(img, f"Wheelbase: {wheelbase_cm:.1f} cm",
                    (center_r[0], center_r[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        # Tentukan orientasi sepeda
        orientation = "Right-facing" if center_f[0] > center_r[0] else "Left-facing"
        cv2.putText(img, f"Orientation: {orientation}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 200), 3)

        # --- Panjang total sepeda ---
        front_left = int(min(x1_f, x2_f))
        rear_right = int(max(x1_r, x2_r))
        y_base = int(max(y2_f, y2_r)) + 50

        length_px = abs(rear_right - front_left)
        length_cm = length_px * scale_cm_per_px

        cv2.line(img, (front_left, y_base), (rear_right, y_base), (0, 0, 255), 4)
        cv2.putText(img, f"Length: {length_cm:.1f} cm",
                    (front_left, y_base - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # --- Crank ke sumbu vertikal ban depan ---
    if "crank" in bboxes and "f_tire" in bboxes:
        x1_c, y1_c, x2_c, y2_c = bboxes["crank"]
        crank_center = (int((x1_c + x2_c) / 2), int((y1_c + y2_c) / 2))

        x1_f, y1_f, x2_f, y2_f = bboxes["f_tire"]
        center_f = (int((x1_f + x2_f) / 2), int((y1_f + y2_f) / 2))

        vertical_x = center_f[0]

        dist_px = abs(crank_center[0] - vertical_x)
        dist_cm = dist_px * scale_cm_per_px

        cv2.line(img, crank_center, (vertical_x, crank_center[1]), (0, 165, 255), 4)
        cv2.putText(img, f"Crank->F_tire axis: {dist_cm:.1f} cm",
                    (crank_center[0], crank_center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

        cv2.line(img, (vertical_x, 0), (vertical_x, img.shape[0]), (200, 200, 200), 2)

    # --- Saddle depan ke sumbu crank ---
    if "saddle" in bboxes and "crank" in bboxes and "f_tire" in bboxes and "r_tire" in bboxes:
        x1_s, y1_s, x2_s, y2_s = bboxes["saddle"]
        x1_c, y1_c, x2_c, y2_c = bboxes["crank"]

        # tentukan orientasi sepeda
        x1_f, y1_f, x2_f, y2_f = bboxes["f_tire"]
        x1_r, y1_r, x2_r, y2_r = bboxes["r_tire"]
        center_f = (int((x1_f + x2_f) / 2), int((y1_f + y2_f) / 2))
        center_r = (int((x1_r + x2_r) / 2), int((y1_r + y2_r) / 2))

        if center_f[0] > center_r[0]:  # sepeda hadap kanan
            saddle_front = (int(x2_s), int((y1_s + y2_s) / 2))  # sisi kanan = ujung depan
        else:  # sepeda hadap kiri
            saddle_front = (int(x1_s), int((y1_s + y2_s) / 2))  # sisi kiri = ujung depan

        crank_center = (int((x1_c + x2_c) / 2), int((y1_c + y2_c) / 2))
        vertical_x = crank_center[0]

        dist_px = abs(saddle_front[0] - vertical_x)
        dist_cm = dist_px * scale_cm_per_px

        cv2.line(img, saddle_front, (vertical_x, saddle_front[1]), (128, 0, 128), 4)
        cv2.putText(img, f"Saddle front->Crank axis: {dist_cm:.1f} cm",
                    (saddle_front[0], saddle_front[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 0, 128), 3)

        # garis vertikal sumbu crank
        cv2.line(img, (vertical_x, 0), (vertical_x, img.shape[0]), (180, 180, 255), 2)

    # --- Diameter ban depan ---
    if "f_tire" in bboxes:
        x1_f, y1_f, x2_f, y2_f = bboxes["f_tire"]
        width_f = abs(x2_f - x1_f)
        height_f = abs(y2_f - y1_f)
        diameter_f_px = (width_f + height_f) / 2
        diameter_f_cm = diameter_f_px * scale_cm_per_px

        center_f = (int((x1_f + x2_f) / 2), int((y1_f + y2_f) / 2))
        cv2.putText(img, f"F_tire dia: {diameter_f_cm:.1f} cm",
                    (center_f[0], center_f[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.circle(img, center_f, int(diameter_f_px/2), (0, 255, 255), 3)

    # --- Diameter ban belakang ---
    if "r_tire" in bboxes:
        x1_r, y1_r, x2_r, y2_r = bboxes["r_tire"]
        width_r = abs(x2_r - x1_r)
        height_r = abs(y2_r - y1_r)
        diameter_r_px = (width_r + height_r) / 2
        diameter_r_cm = diameter_r_px * scale_cm_per_px

        center_r = (int((x1_r + x2_r) / 2), int((y1_r + y2_r) / 2))
        cv2.putText(img, f"R_tire dia: {diameter_r_cm:.1f} cm",
                    (center_r[0], center_r[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.circle(img, center_r, int(diameter_r_px/2), (0, 255, 255), 3)


# Simpan hasil
cv2.imwrite("output_with_measurements.jpg", img)
print("[INFO] Hasil disimpan di output_with_measurements.jpg")
