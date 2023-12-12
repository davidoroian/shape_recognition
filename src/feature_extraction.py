import cv2
import numpy as np

def custom_moments(image):
    rows, cols = image.shape
    m00 = np.sum(image)
    m10 = np.sum(np.arange(cols).reshape(1, -1) * image)
    m01 = np.sum(np.arange(rows).reshape(-1, 1) * image)

    x_bar = m10 / m00
    y_bar = m01 / m00

    m20 = np.sum((np.arange(cols).reshape(1, -1) - x_bar) ** 2 * image)
    m11 = np.sum((np.arange(rows).reshape(-1, 1) - y_bar) *
                 (np.arange(cols).reshape(1, -1) - x_bar) * image)
    m02 = np.sum((np.arange(rows).reshape(-1, 1) - y_bar) ** 2 * image)

    m30 = np.sum((np.arange(cols).reshape(1, -1) - x_bar) ** 3 * image)
    m21 = np.sum((np.arange(rows).reshape(-1, 1) - y_bar) *
                 (np.arange(cols).reshape(1, -1) - x_bar) ** 2 * image)
    m12 = np.sum((np.arange(rows).reshape(-1, 1) - y_bar) ** 2 *
                 (np.arange(cols).reshape(1, -1) - x_bar) * image)
    m03 = np.sum((np.arange(rows).reshape(-1, 1) - y_bar) ** 3 * image)

    mu20 = m20 - x_bar * m10
    mu11 = m11 - x_bar * m01
    mu02 = m02 - y_bar * m01

    mu30 = m30 - 3 * x_bar * mu20 + 2 * x_bar**2 * m10
    mu21 = m21 - 2 * y_bar * mu11 - x_bar * mu20 + 2 * x_bar**2 * m01
    mu12 = m12 - 2 * x_bar * mu11 - y_bar * mu02 + 2 * y_bar**2 * m10
    mu03 = m03 - 3 * y_bar * mu02 + 2 * y_bar**2 * m01

    nu20 = mu20 / m00**2
    nu11 = mu11 / (m00**2.5)
    nu02 = mu02 / m00**2

    nu30 = mu30 / (m00**2.5)
    nu21 = mu21 / (m00**2.5)
    nu12 = mu12 / (m00**2.5)
    nu03 = mu03 / (m00**2.5)

    return {
        'm00': m00,
        'm10': m10,
        'm01': m01,
        'm20': m20,
        'm11': m11,
        'm02': m02,
        'm30': m30,
        'm21': m21,
        'm12': m12,
        'm03': m03,
        'mu20': mu20,
        'mu11': mu11,
        'mu02': mu02,
        'mu30': mu30,
        'mu21': mu21,
        'mu12': mu12,
        'mu03': mu03,
        'nu20': nu20,
        'nu11': nu11,
        'nu02': nu02,
        'nu30': nu30,
        'nu21': nu21,
        'nu12': nu12,
        'nu03': nu03
    }

# Calculate Hu moments for an image
def calculate_hu_moments(image):
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)
    return -np.sign(hu_moments) * np.log10(np.abs(hu_moments))

def calculate_hu_moments_custom(image):
    moments = custom_moments(image)
    hu_moments = cv2.HuMoments(moments)
    return -np.sign(hu_moments) * np.log10(np.abs(hu_moments))