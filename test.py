import os
import cv2
from src.feature_extraction import calculate_hu_moments_custom, calculate_hu_moments

image = cv2.imread(f"shapes\square\\0.png", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (50, 50))
f = open("demofile2.txt", "w")
for i in range(len(image)):
    for j in range(len(image[i])):
        f.write(f"{image[i][j]} ")
    f.write("\n")
f.close()

moments = calculate_hu_moments(image).flatten()
moments_custom = calculate_hu_moments_custom(image).flatten()