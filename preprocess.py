import cv2
import os
import glob
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Folder paths
input_folder = "test_bike"
output_folder = "preprocess_result"
os.makedirs(output_folder, exist_ok=True)

target_size = 1280

# Get all image files in test_bike (jpg, jpeg, png)
image_paths = glob.glob(os.path.join(input_folder, '*.*'))
image_paths = [p for p in image_paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]

for source_image_path in image_paths:
    img = cv2.imread(source_image_path)
    if img is None:
        print(f"Image not found or unreadable: {source_image_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    # Convert back to BGR for saving
    resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    filename = os.path.basename(source_image_path)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, resized_bgr)
    print(f"Saved resized image to {output_path}")