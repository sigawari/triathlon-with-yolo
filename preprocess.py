import cv2
import os
import tensorflow as tf

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
# from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Path file img & inisialize
file_sepeda = "element.jpg"
source_image_path = os.path.join("test_bike", file_sepeda)
target_size = 1280

# Read img & convert to RGB
img = cv2.imread(source_image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at: {source_image_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get current size
h, w = img.shape[:2]
scale = target_size / max(h, w)

# Resize but keep aspect ratio
new_w, new_h = int(w * scale), int(h * scale)
resized = cv2.resize(img, (new_w, new_h))

cv2.imshow("Original", img)
cv2.imshow("Resized with Ratio", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()