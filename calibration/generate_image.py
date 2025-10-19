# Mengambil gambar untuk file kalibrasi (manual shutter + overlay)
import cv2
import os
from datetime import datetime

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to read camera feed")
    exit(1)

output_dir = "calibrator_image"
os.makedirs(output_dir, exist_ok=True)

img_counter = 0
armed = False       # mode siap jepret
show_help = True    # overlay bantuan (hanya di layar)

win_name = "webcam kalibrasi"
cv2.namedWindow(win_name)

def draw_hud(frame, armed, img_counter, show_help):
    h, w = frame.shape[:2]
    status = f"MODE: {'ARMED' if armed else 'IDLE'} | Saved: {img_counter}"
    color = (0, 255, 0) if armed else (0, 165, 255)
    cv2.rectangle(frame, (0, 0), (w, 48), (0,0,0), -1)
    cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    if show_help:
        lines = [
            "Controls: A=Arm/Disarm  S=Snap  H=Help  R=Reset  ESC/Close=Exit",
            "Tips: Variasikan jarak & sudut; ambil 12-25 pose unik."
        ]
        y0 = h - 42
        cv2.rectangle(frame, (0, h-64), (w, h), (0,0,0), -1)
        for i, t in enumerate(lines):
            cv2.putText(frame, t, (12, y0 + i*26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Salinan untuk overlay; frame asli tetap bersih
    display = frame.copy()
    draw_hud(display, armed, img_counter, show_help)
    cv2.imshow(win_name, display)

    # Deteksi jika user menekan tombol close (X) di window
    # getWindowProperty akan < 1 jika jendela ditutup/minimized depending backend
    prop = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)
    if prop < 1:
        print("Jendela ditutup, menutup program....")
        break

    k = cv2.waitKey(1) & 0xFF
    if k == 27:                     # ESC: keluar
        print("Keluar (ESC).")
        break
    elif k == ord('h'):             # H: toggle bantuan
        show_help = not show_help
    elif k == ord('a'):             # A: toggle ARMED/IDLE
        armed = not armed
        print("MODE:", "ARMED" if armed else "IDLE")
    elif k == ord('r'):             # R: reset counter
        img_counter = 0
        print("Counter direset ke 0")
    elif k == ord('s'):             # S: jepret hanya jika ARMED
        if armed:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_name = os.path.join(output_dir, f"frame_{img_counter:03d}_{ts}.png")
            if cv2.imwrite(img_name, frame):   # simpan frame mentah (tanpa overlay)
                print(f"Tersimpan: {img_name}")
                img_counter += 1
            else:
                print("Gagal menyimpan gambar.")
        else:
            print("Belum ARMED. Tekan 'A' untuk siap jepret.")

cap.release()
cv2.destroyAllWindows()
