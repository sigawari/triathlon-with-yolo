# Kalibrasi ChArUco 16x9 dengan auto-detect dictionary 6x6
import cv2 as cv
import numpy as np
import json, glob, os

# ====== Konfigurasi papan (SESUIKAN dengan cetakanmu 9x16) ======
SQUARES_X = 16
SQUARES_Y = 9
SQUARE_LENGTH = 0.017
MARKER_LENGTH = 0.012

# ====== Folder input ======
input_dir = "calibrator_image"
image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")) +
                     glob.glob(os.path.join(input_dir, "*.jpg")) +
                     glob.glob(os.path.join(input_dir, "*.jpeg")))
assert len(image_paths) > 0, f"Tidak ada gambar di folder: {input_dir}"

# ====== 1) Auto-detect dictionary terbaik ======
candidates = [
    cv.aruco.DICT_6X6_50,
    cv.aruco.DICT_6X6_100,
    cv.aruco.DICT_6X6_250,
    cv.aruco.DICT_6X6_1000,
]

def detector_with_params(dictionary_id):
    dic = cv.aruco.getPredefinedDictionary(dictionary_id)
    params = cv.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 23
    params.minMarkerPerimeterRate = 0.02
    det = cv.aruco.ArucoDetector(dic, params)
    return dic, det

# Ambil maksimal 5 sampel awal untuk voting
sample_paths = image_paths[:min(5, len(image_paths))]
scores = {}
for cid in candidates:
    dic, det = detector_with_params(cid)
    total = 0
    for p in sample_paths:
        img = cv.imread(p, cv.IMREAD_GRAYSCALE)
        if img is None: 
            continue
        corners, ids, _ = det.detectMarkers(img)
        total += 0 if ids is None else len(ids)
    scores[cid] = total

# Pilih dictionary dengan total marker terbanyak
best_dict_id = max(scores, key=scores.get)
best_total = scores[best_dict_id]
print("Skor kandidat:", {int(k): v for k, v in scores.items()})
print(f"Dictionary terpilih: DICT_6X6_{int(str(best_dict_id).split('_')[-1])} (total markers={best_total} pada {len(sample_paths)} sampel)")

DICTIONARY, DETECTOR = detector_with_params(best_dict_id)
BOARD = cv.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, DICTIONARY)

# ====== 2) Proses semua gambar + preview ======
all_corners, all_ids = [], []
image_size = None

cv.namedWindow("preview", cv.WINDOW_NORMAL)
cv.resizeWindow("preview", 960, 720)

MIN_CHARUCO = 8  # ambang minimal sudut agar frame dipakai

for p in image_paths:
    img = cv.imread(p, cv.IMREAD_COLOR)
    if img is None:
        print(f"Lewati (gagal baca): {p}")
        continue
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = (gray.shape[1], gray.shape[0])

    corners, ids, _ = DETECTOR.detectMarkers(gray)
    overlay = img.copy()
    n_markers = 0 if ids is None else len(ids)
    if n_markers > 0:
        cv.aruco.drawDetectedMarkers(overlay, corners, ids, borderColor=(0,255,0))

    retval, ch_corners, ch_ids = cv.aruco.interpolateCornersCharuco(corners, ids, gray, BOARD)
    n_charuco = 0 if ch_ids is None else len(ch_ids)
    if n_charuco > 0:
        cv.aruco.drawDetectedCornersCharuco(overlay, ch_corners, ch_ids, (255,0,0))

    info = f"{os.path.basename(p)} | dict={best_dict_id} markers={n_markers} charuco={n_charuco}"
    cv.rectangle(overlay, (0,0), (overlay.shape[1], 36), (0,0,0), -1)
    cv.putText(overlay, info, (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv.LINE_AA)

    cv.imshow("preview", overlay)
    key = cv.waitKey(200) & 0xFF
    if key == 27:
        print("Dihentikan oleh pengguna.")
        cv.destroyAllWindows()
        raise SystemExit
    elif key == ord(' '):
        while True:
            k2 = cv.waitKey(0) & 0xFF
            if k2 in (ord(' '), 13):
                break
            if k2 == 27:
                print("Dihentikan oleh pengguna.")
                cv.destroyAllWindows()
                raise SystemExit

    if n_markers == 0 or n_charuco < MIN_CHARUCO:
        print(f"Skip: markers={n_markers} charuco={n_charuco} -> {os.path.basename(p)}")
        continue

    all_corners.append(ch_corners)
    all_ids.append(ch_ids)

cv.destroyWindow("preview")

print(f"Gambar valid untuk kalibrasi: {len(all_corners)} dari {len(image_paths)}")
assert len(all_corners) >= 10, "Kumpulkan >=10 gambar valid untuk hasil stabil"

# ====== 3) Kalibrasi ======
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv.aruco.calibrateCameraCharuco(
    charucoCorners=all_corners,
    charucoIds=all_ids,
    board=BOARD,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

print("Reprojection error:", float(ret))
print("Camera matrix:\n", cameraMatrix)
print("Distortion coeffs:", distCoeffs.ravel())

# ====== 4) Simpan JSON ======
dict_label = f"DICT_6X6_{int(str(best_dict_id).split('_')[-1])}"
output_json = {
    "image_width": int(image_size[0]),
    "image_height": int(image_size[1]),
    "reprojection_error": float(ret),
    "camera_matrix": cameraMatrix.tolist(),
    "distortion_coefficients": distCoeffs.reshape(-1).tolist(),
    "model": "opencv_aruco_charuco",
    "board": {
        "dictionary": dict_label,
        "squares_x": SQUARES_X,
        "squares_y": SQUARES_Y,
        "square_length_m": SQUARE_LENGTH,
        "marker_length_m": MARKER_LENGTH
    }
}

with open("camera_calibration_charuco.json", "w") as f:
    json.dump(output_json, f, indent=2)

print("Tersimpan -> camera_calibration_charuco.json")
