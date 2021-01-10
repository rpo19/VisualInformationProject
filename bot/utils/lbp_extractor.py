from skimage import feature
import numpy as np
import cv2

class LBPFeaturesExtractor():

    def __init__(self, n_points, radius, center_region_size):
        # number of points
        self.n_points = n_points
        # radius
        self.radius = radius
        self.center_region_size = center_region_size

    def extract(self, img, center_only=False):
        # convert image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # get size of image
        (w, h) = img.shape
        # compute center coordinates
        (centerX, centerY) = (int(w/2), int(h/2))
        
        if center_only:
            # get mask for center region
            rect_mask = self.__getCenterRect(centerX, centerY, w, h)
            # extract features from center region
            features = self.__LBPhist(img, self.n_points, self.radius, rect_mask)
        else:
            # extract features from the whole image
            features = self.__LBPhist(img, n_points=self.n_points, radius=self.radius)
        
        return features
    
    def __getCenterRect(self, centerX, centerY, w, h):
        # create mask of zeros
        mask = np.zeros((w, h), dtype = 'uint8')
        # compute top left corner of rect and bottom right corner of rect
        top_left = (centerX - int(centerX * self.center_region_size), centerY + int(centerY * self.center_region_size))
        bottom_right = (centerX + int(centerX * self.center_region_size),  centerY - int(centerY * self.center_region_size)) 

        # set center rectangle to white
        mask[top_left[0]: bottom_right[0], bottom_right[1]:top_left[1]] = 255
        return mask
    
    def __LBPhist(self, img, n_points, radius, mask=None):
        # if a mask is given filter out only the needed part
        if mask is not None:
            img = cv2.multiply(img, mask)
        # compute LBP 
        lbp = feature.local_binary_pattern(img, n_points, radius, method='uniform')
        # build histogram of LBP representation
        bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins))
        # normalize histogram
        hist = hist.astype("float")
        hist = np.divide(hist, hist.sum())
        return hist


if __name__ == "__main__":
    img = cv2.imread('../data/train/1.jpg')
    # istantiate class to extract LBP features with the 
    # number of points and radius of the LBP computation

    features_extractor = LBPFeaturesExtractor(24, 7, 0.6)
    features = features_extractor.extract(img, center_only=True)
    print('Number of features extracted: ', len(features))

 
