# Testing hasil nilai JSON Kalibrasi

import cv2 as cv
import numpy as np
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))           
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))  
JSON_NAME = "camera_calibration_charuco.json"
JSON_PATH = os.path.join(ROOT_DIR, JSON_NAME)

# Jika file JSON di subfolder, misalnya "calibration":
# JSON_PATH = os.path.join(BASE_DIR, "calibration", JSON_NAME)

CAM_INDEX = 0

def load_calib(json_path):
    # Jika path relatif diberikan, pastikan absolut
    json_path = os.path.abspath(json_path)
    if not os.path.isfile(json_path):
        # Fallback: coba cari di BASE_DIR
        alt = os.path.join(BASE_DIR, os.path.basename(json_path))
        if os.path.isfile(alt):
            json_path = alt
        else:
            raise FileNotFoundError(f"File kalibrasi tidak ditemukan: {json_path}")
    with open(json_path, "r") as f:
        calib = json.load(f)
    w = int(calib["image_width"])
    h = int(calib["image_height"])
    K = np.array(calib["camera_matrix"], dtype=np.float64)
    d = np.array(calib["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    return (w, h, K, d, json_path)

def main():
    w, h, K, d, used_path = load_calib(JSON_PATH)
    print(f"Muat JSON: {used_path}")

    cap = cv.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Tidak bisa membuka kamera.")
        sys.exit(1)

    cap.set(cv.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

    ret, frame = cap.read()
    if not ret:
        print("Tidak bisa membaca kamera.")
        cap.release()
        sys.exit(1)

    H, W = frame.shape[:2]
    if (W, H) != (w, h):
        print(f"Peringatan: kamera memberi {W}x{H}, bukan {w}x{h} dari JSON. "
              "Peta undistort mungkin tidak akurat.")

    newK, roi = cv.getOptimalNewCameraMatrix(K, d, (W, H), alpha=0)
    map1, map2 = cv.initUndistortRectifyMap(K, d, None, newK, (W, H), cv.CV_16SC2)

    print("Kalibrasi dimuat. ESC untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        undist = cv.remap(frame, map1, map2, interpolation=cv.INTER_LINEAR)

        x, y, rw, rh = roi
        view = undist[y:y+rh, x:x+rw] if (rw > 0 and rh > 0) else undist

        cv.imshow("undistorted", view)
        if (cv.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
