import cv2
import cv2.aruco as aruco

# pilih dictionary
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# bikin marker ID=23
markerImage = aruco.generateImageMarker(dictionary, 20, 200)

# simpan gambar
cv2.imwrite("marker20(6).png", markerImage)