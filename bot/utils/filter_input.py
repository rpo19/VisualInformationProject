import cv2
import numpy as np
import statistics
import tempfile
import tensorflow as tf



def is_blurred_old(img_path, threshold=50):
    # show img
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # compute laplacian variance
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # TODO: valutare parametri
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()

    return laplacian_var < threshold, laplacian_var

def is_blurred(img_path, model):
    # load img
    img = load_image(img_path)
    # check blurriness
    pred = model.predict(img)[0][0]
    is_blurred = pred > 0.5 
    print('is_blur prediction', pred)
    return is_blurred

def load_image(path_to_img):
  img = tf.keras.preprocessing.image.load_img(path_to_img, target_size=(224, 224))
  img = tf.keras.preprocessing.image.img_to_array(img)
  img /= 255
  img = np.expand_dims(img, 0)
  return img

def is_dark(img_path, threshold=50):
    # show image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # get brightness from HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = img_hsv[..., 2].mean()

    return brightness < threshold, brightness

def fix_darkness(img_path, threshold = 50):
    img = cv2.imread(img_path)
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
            print('fixing darkness...')
            is_dark = True
            intensity = adaptive_gamma_correction(intensity, threshold)
            imgYCC[:,:,0] = intensity
            

    img_final = cv2.cvtColor(imgYCC, cv2.COLOR_YCrCb2BGR)
    

    return is_dark, img_final
    
def adaptive_gamma_correction(intensity, threshold):
    # cale values
    threshold /= 255
    y_mean = np.mean(intensity/255)

    mask = cv2.bilateralFilter(intensity, d=2, sigmaColor=7, sigmaSpace=7)
    mask = 255 - mask

    intensity = intensity.astype(float)
    mask = mask.astype(float)

    # 2- y_mean/threshold to regulate the correction
    base = 2- y_mean/threshold*0.8
    gamma = (base)**((128 - mask) / 128)
    print('gamma base', base)
    intensity = 255*((intensity/255)**gamma)
    intensity = np.uint8(intensity)

    return intensity


def has_clear_margins(img_path, margin=1, bilateral=True):
    img = cv2.imread(img_path)
    if bilateral:
        img = cv2.bilateralFilter(
            img, 45, 200, 40, cv2.BORDER_REFLECT)
    # extract stripes
    width = img.shape[1]
    height = img.shape[0]
    top_border = img[0:margin, :]
    bottom_border = img[height-margin:height, :]
    left_border = img[:, 0:margin]
    right_border = img[:, width-margin:width]
    # check edges presence in the borders
    top_canny = cv2.Canny(top_border, 100,  150)
    bottom_canny = cv2.Canny(bottom_border, 100,  150)
    left_canny = cv2.Canny(left_border, 100,  150)
    right_canny = cv2.Canny(right_border, 100,  150)
    
    concatenated = np.concatenate(
        [top_canny, bottom_canny, np.transpose(left_canny), np.transpose(right_canny)], axis=1)
    # edges pixels are represented as 255 in canny, while the non-edges are 0
    return concatenated.max() == 0

def has_uniform_bg(img_path, threshold=10, margin=1):
    # read img
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # extract stripes
    width = img.shape[1]
    height = img.shape[0]
    top_border = img[0:margin, :]
    bottom_border = img[height-margin:height, :]
    left_border = img[:, 0:margin]
    right_border = img[:, width-margin:width]
    # concatenate stripes
    concatenated = np.concatenate([top_border, bottom_border, np.transpose(left_border), np.transpose(right_border)], axis=1)
    # compute std
    std = concatenated.std()
    print('std', std)
    std_ok = std < threshold
    # check edge presence in the borders of the image without using filters
    no_edges = has_clear_margins(img_path,margin,bilateral=False)
    return std_ok and no_edges