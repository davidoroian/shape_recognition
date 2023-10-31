import cv2
import numpy as np

# Calculate Hu moments for an image
def calculate_hu_moments(image):
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)
    return -np.sign(hu_moments) * np.log10(np.abs(hu_moments))