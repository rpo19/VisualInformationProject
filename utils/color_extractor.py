import numpy as np
import cv2

class ColorFeaturesExtractor():

    def __init__(self, bins, center_region_size):
		# number of bins of the histogram
        # the more bins you have the less is the generalization.
        self.bins = bins
        self.center_region_size = center_region_size

    def extract(self, img, center_only=False):
        # convert color space to HSV (RGB doesn't handle well shadows)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # get size of image
        (w, h, _) = img.shape
        # compute center coordinates
        (centerX, centerY) = (int(w/2), int(h/2))

        # divide image in regions
        corners = [(0,0), (w,0), (w,h), (0,h)]
        
        # get mask for center region
        rect_mask = self.__getCenterRect(centerX, centerY, w, h)

        features = []
        # extract features from center region
        features_region = self.__colorHist(img, rect_mask, self.bins)
        features.extend(features_region)

        if not center_only:
            # divide image in corner regions and extract features from each one of them
            for corner in corners:
                corner_mask = self.__getCornerRect(corner, centerX, centerY, w, h, rect_mask)
                # extract features from corner region
                features_region = self.__colorHist(img, corner_mask, self.bins)
                # append array of local features to all features
                features.extend(features_region)
        
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
    
    def __getCornerRect(self, corner, centerX, centerY, w, h, centerMask):
        # create mask of zeros
        mask = np.zeros((w, h), dtype = 'uint8')
        # fill corner region
        cv2.rectangle(mask, corner, (centerX, centerY), 255, -1)
        mask = cv2.subtract(mask, centerMask)
        return mask
    
    def __colorHist(self, img, mask, bins):
        # calc 3D hist 
        hist = cv2.calcHist([img], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
        # normalize values in each bin
        hist = cv2.normalize(hist, hist)
        # flatten hist to a 1D vector
        hist = hist.flatten()
        return hist


if __name__ == "__main__":
    img = cv2.imread('../data/train/128.jpg')
    # istantiate class to extract color features with the number of bins
    # Bins are specified for each channel of the color space (HSV in this case)
    # We shouldn't specify an high number of bins for the brightness channel because
    # it can introduce noise cause by shadows.
    features_extractor = ColorFeaturesExtractor((24, 26, 3), 0.6)
    features = features_extractor.extract(img, center_only=True)
    print('Number of features extracted: ', len(features))
    
 