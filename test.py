import os
import cv2
import math
import sys
from ultralytics import YOLO

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def process_image(img_path, model):
    print(f"Processing: {img_path}")
    img = cv2.imread(img_path)
    results = model(img)
    segmented_img = results[0].plot()

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    centers = []
    print(f"Detected objects in {os.path.basename(img_path)}:")
    for i, (bbox, conf, cls) in enumerate(zip(boxes, scores, classes)):
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        centers.append((int(x_center), int(y_center)))
        print(f" Object {i}: Class={int(cls)}, Confidence={conf:.2f}, Center=({x_center:.1f}, {y_center:.1f})")

    # Hitung jarak antar center
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dist = euclidean_distance(centers[i], centers[j])
            print(f" Distance between object {i} and {j}: {dist:.1f} pixels")

            # Gambar garis dan jarak di foto
            cv2.line(segmented_img, centers[i], centers[j], (0, 255, 255), 2)
            mid_point = ((centers[i][0]+centers[j][0])//2, (centers[i][1]+centers[j][1])//2)
            cv2.putText(segmented_img, f"{dist:.1f}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Tampilkan hasil
    window_name = f"Segmented - {os.path.basename(img_path)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1080)
    cv2.imshow(window_name, segmented_img)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'): 
            cv2.destroyAllWindows()
            exit()
        elif key == 13: 
            cv2.destroyAllWindows()
            break

def main():
    model = YOLO("best-960.pt")
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if os.path.isfile(arg):
                process_image(arg, model)
            elif os.path.isdir(arg):
                for f in os.listdir(arg):
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        process_image(os.path.join(arg, f), model)
            else:
                print(f"Not found: {arg}")
    else:
        folder_path = os.getcwd()
        for file_name in os.listdir(folder_path):
            if file_name.lower().startswith("test") and file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                process_image(os.path.join(folder_path, file_name), model)


if __name__ == "__main__":
    main()
