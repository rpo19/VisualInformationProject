from skimage.transform import resize
from skimage.feature import hog

class HogFeaturesExtractor():
    
    def extract(self, img, f_resize=True, dims_resize=(128,64)):
        if f_resize:
            img=self.__resize_img(img, dims_resize)

        features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
        return features

    def __resize_img(self, img, dims_resize):
        img = resize(img, dims_resize) 
        
        return img