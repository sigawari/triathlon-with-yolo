from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt

# Muat model
model = YOLO("best.pt") 

# Skala cm/px akan di-set dari f_tire
scale_cm_per_px = None 

# Folder input/output
input_folder = "sepedaku"
output_folder = "predict_nopreprocess"
os.makedirs(output_folder, exist_ok=True)  

file_sepeda = "sepeda_Ultramarathon.jpeg"
source_image_path = os.path.join(input_folder, file_sepeda)

# Mapping class
class_map = {
    0: "saddle",
    1: "f_tire",
    2: "crank",
    3: "r_tire"
} 

# Baca gambar
img = cv2.imread(source_image_path)  # BGR 
if img is None:
    raise FileNotFoundError(f"Image not found: {source_image_path}")

print(f"[INFO] Memproses: {source_image_path}")  

# Prediksi
results = model.predict(
    source=source_image_path,
    show=False,
    save=False,
    conf=0.7,
    line_width=2,
    classes=[0, 1, 2, 3]
) 

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy() 
    cls = r.boxes.cls.cpu().numpy()    

    bboxes = {}
    for box, c in zip(boxes, cls):
        label = class_map.get(int(c), str(c)) 
        bboxes[label] = box

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3) 
        cv2.putText(img, label, (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3) 
        print(f"Detected {label}: ({x1}, {y1}), ({x2}, {y2})")  

    # Diameter ban depan & scale
    if "f_tire" in bboxes:
        x1_f, y1_f, x2_f, y2_f = bboxes["f_tire"]
        width_f = abs(x2_f - x1_f)
        height_f = abs(y2_f - y1_f)
        diameter_f_px = (width_f + height_f) / 2
        scale_cm_per_px = 67.2 / diameter_f_px  # asumsi 700c 672 mm 
        print("cm/px = ", scale_cm_per_px)  
        diameter_f_cm = diameter_f_px * scale_cm_per_px
        print(f"F_tire diameter: {diameter_f_cm:.1f} cm")  

        center_f = (int((x1_f + x2_f) / 2), int((y1_f + y2_f) / 2))
        cv2.putText(img, f"F_tire dia: {diameter_f_cm:.1f} cm",
                    (center_f[0], center_f[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3) 
        cv2.circle(img, center_f, int(diameter_f_px/2), (0, 255, 255), 3) 

    # Pengukuran lain jika scale ada
    if scale_cm_per_px is not None:
        # Wheelbase
        if "f_tire" in bboxes and "r_tire" in bboxes:
            x1_f, y1_f, x2_f, y2_f = bboxes["f_tire"]
            x1_r, y1_r, x2_r, y2_r = bboxes["r_tire"]

            center_f = (int((x1_f + x2_f) / 2), int((y1_f + y2_f) / 2))
            center_r = (int((x1_r + x2_r) / 2), int((y1_r + y2_r) / 2))

            wheelbase_px = ((center_f[0] - center_r[0]) ** 2 + (center_f[1] - center_r[1]) ** 2) ** 0.5
            wheelbase_cm = wheelbase_px * scale_cm_per_px
            print(f"Wheelbase: {wheelbase_cm:.1f} cm")  

            cv2.line(img, center_f, center_r, (255, 0, 0), 4) 
            cv2.putText(img, f"Wheelbase: {wheelbase_cm:.1f} cm",
                        (center_r[0], center_r[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3) 

            # Orientasi sepeda
            orientation = "Right-facing" if center_f[0] > center_r[0] else "Left-facing"
            print(f"Orientation: {orientation}")  
            cv2.putText(img, f"Orientation: {orientation}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 200), 3) 

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
                        (front, y_base - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3) 

        # Crank ke sumbu vertikal ban depan
        if "crank" in bboxes and "f_tire" in bboxes:
            x1_c, y1_c, x2_c, y2_c = bboxes["crank"]
            crank_center = (int((x1_c + x2_c) / 2), int((y1_c + y2_c) / 2))

            x1_f, y1_f, x2_f, y2_f = bboxes["f_tire"]
            center_f = (int((x1_f + x2_f) / 2), int((y1_f + y2_f) / 2))
            vertical_x = center_f[0]

            dist_px = abs(crank_center[0] - vertical_x)
            dist_cm = dist_px * scale_cm_per_px
            print(f"Crank->F_tire axis: {dist_cm:.1f} cm")  

            cv2.line(img, crank_center, (vertical_x, crank_center[1]), (0, 165, 255), 4) 
            cv2.putText(img, f"Crank->F_tire axis: {dist_cm:.1f} cm",
                        (crank_center[0], crank_center[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3) 
            cv2.line(img, (vertical_x, 0), (vertical_x, img.shape[0]), (200, 200, 200), 2) 

        # Saddle depan ke sumbu crank
        if "saddle" in bboxes and "crank" in bboxes and "f_tire" in bboxes and "r_tire" in bboxes:
            x1_s, y1_s, x2_s, y2_s = bboxes["saddle"]
            x1_c, y1_c, x2_c, y2_c = bboxes["crank"]

            x1_f, y1_f, x2_f, y2_f = bboxes["f_tire"]
            x1_r, y1_r, x2_r, y2_r = bboxes["r_tire"]
            center_f = (int((x1_f + x2_f) / 2), int((y1_f + y2_f) / 2))
            center_r = (int((x1_r + x2_r) / 2), int((y1_r + y2_r) / 2))
            if center_f[0] > center_r[0]:
                saddle_front = (int(x2_s), int((y1_s + y2_s) / 2))
            else:
                saddle_front = (int(x1_s), int((y1_s + y2_s) / 2))

            crank_center = (int((x1_c + x2_c) / 2), int((y1_c + y2_c) / 2))
            vertical_x = crank_center[0]

            dist_px = abs(saddle_front[0] - vertical_x)
            dist_cm = dist_px * scale_cm_per_px
            print(f"Saddle front->Crank axis: {dist_cm:.1f} cm")  

            cv2.line(img, saddle_front, (vertical_x, saddle_front[1]), (128, 0, 128), 4) 
            cv2.putText(img, f"Saddle front->Crank axis: {dist_cm:.1f} cm",
                        (saddle_front[0], saddle_front[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 0, 128), 3) 
            cv2.line(img, (vertical_x, 0), (vertical_x, img.shape[0]), (180, 180, 255), 2) 

        # Crank ke tanah
        if "crank" in bboxes and "f_tire" in bboxes:
            x1_c, y1_c, x2_c, y2_c = bboxes["crank"]
            crank_center = (int((x1_c + x2_c) / 2), int((y1_c + y2_c) / 2))

            x1_f, y1_f, x2_f, y2_f = bboxes["f_tire"]
            ground_y = int(max(y2_f, y2_f))  # bawah f_tire 

            dist_px = abs(crank_center[1] - ground_y)
            dist_cm = dist_px * scale_cm_per_px
            print(f"Crank->Ground: {dist_cm:.1f} cm")  
            if 24 <= dist_cm <= 30:
                print("Crank to Ground OK")  
            else:
                print("The bike is not legal")  

            cv2.line(img, crank_center, (crank_center[0], ground_y), (0, 100, 255), 4) 
            label_text = f"Crank->Ground:\n{dist_cm:.1f} cm"
            for i, line in enumerate(label_text.split("\n")):
                cv2.putText(img, line, (crank_center[0] + 60, crank_center[1] + 60 + i*35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 3) 
                font_scale = 0.7
                thickness = 2
                (text_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                x_right = crank_center[0] + 200
                x_aligned = x_right - text_width
                y_aligned = crank_center[1] + 60 + i*25
                cv2.putText(img, line, (x_aligned, y_aligned),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 100, 255), thickness) 

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
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3) 
            cv2.circle(img, center_r, int(diameter_r_px/2), (0, 255, 255), 3) 

# Simpan hasil anotasi
output_path = os.path.join(output_folder, f"output_{file_sepeda}")
cv2.imwrite(output_path, img)
print(f"[INFO] Hasil disimpan di {output_path}")  

# Tampilkan dengan Matplotlib (BGR->RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.axis('off')
plt.tight_layout()


plt.show()
