import cv2
import numpy as np
import statistics


def is_blurred(img_path, threshold=50):
    # show img
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # compute laplacian variance
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # TODO: valutare parametri
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()

    return laplacian_var < threshold, laplacian_var


def is_dark(img_path, threshold=50):
    # show image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # get brightness from HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = img_hsv[..., 2].mean()

    return brightness < threshold, brightness

def fix_darkness(self, img, threshold = 50):
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    intensity = imgYCC[:,:,0]

    y_mean = np.mean(intensity)
    y_mode = statistics.mode(intensity.ravel())
    y_std = np.std(intensity)

    skewness = (y_mean - y_mode) / y_std

    is_dark = False

    if skewness > 0:
        # low key
        if y_mean < threshold:
            is_dark = True
            intensity = adaptive_gamma_correction(intensity)
            imgYCC[:,:,0] = intensity
    
    return is_dark, cv2.cvtColor(imgYCC, cv2.COLOR_YCrCb2BGR)
    
def adaptive_gamma_correction(self, intensity):

    mask = cv2.bilateralFilter(intensity, d=2, sigmaColor=7, sigmaSpace=7)
    mask = 255 - mask

    intensity = intensity.astype(float)
    mask = mask.astype(float)

    gamma = (2- np.mean(intensity/255))**((128 - mask) / 128)
    intensity = 255*((intensity/255)**gamma)
    intensity = np.uint8(intensity)

    return intensity


def has_clear_margins(img_path, margin=1):
    img = cv2.imread(img_path)
    img = cv2.bilateralFilter(
        img, 45, 200, 40, cv2.BORDER_REFLECT)
    # extract stripes
    size = len(img)
    top_border = img[0:margin, :]
    bottom_border = img[size-margin:size, :]
    left_border = img[:, 0:margin]
    right_border = img[:, size-margin:size]
    # check edges presence in the borders
    top_canny = cv2.Canny(top_border, 100,  150)
    bottom_canny = cv2.Canny(bottom_border, 100,  150)
    left_canny = cv2.Canny(left_border, 100,  150)
    right_canny = cv2.Canny(right_border, 100,  150)
    concatenated = np.concatenate(
        [top_canny, bottom_canny, np.transpose(left_canny), np.transpose(right_canny)])
    # edges pixels are represented as 255 in canny, while the non-edges are 0
    return concatenated.max() == 0
