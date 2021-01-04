import cv2
import os
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans

class BOVWFeaturesExtractor():

    def __init__(self, path_to_dict=None):
        self.sift = cv2.SIFT_create()
        self.path_to_dict = path_to_dict
        self.kmeans = None
        # load BOVW dictionary if provided
        if path_to_dict is not None:
            with open(os.path.join(path_to_dict, 'BOVW_dict.pckl'), 'rb') as handle:
                self.kmeans = pickle.load(handle)
                self.kmeans.verbose = 0
            
    
    # build BOVW dioctionary
    def build_dict(self, path_to_images, n_of_words, save_path, path_to_descriptors=None):
        
        if path_to_descriptors is None:
            descriptors = self.__extract_sift_descriptors_from_directory(path_to_images)
            # save descriptors to file
            with open('SIFT_descriptors.pckl', 'wb') as handle:
                pickle.dump(descriptors, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        else:
            # load descriptors
            with open('SIFT_descriptors.pckl', 'rb') as handle:
                descriptors = pickle.load(handle)
        
        # instantiate k-mean with n_of_words clusters
        self.kmeans = MiniBatchKMeans(n_clusters = n_of_words, batch_size=350, verbose=1, n_init=6, max_no_improvement=20)
        # fit kmeans on all descriptors
        print('Fitting kmeans')
        self.kmeans.fit(descriptors)

        # save dictionary to file
        with open('BOVW_dict.pckl', 'wb') as handle:
            pickle.dump(self.kmeans, handle, protocol=pickle.HIGHEST_PROTOCOL)    


    # extract features vector
    def extract(self, img):

        # extract SIFT descriptors from image
        descriptors = self.__extract_sift_descriptors(img)
        descriptors = np.array(descriptors, dtype=np.float)

        # initilize feature vector (histogram)
        hist = np.zeros(self.kmeans.cluster_centers_.shape[0])
        # get closest centroid
        clusters = self.kmeans.predict(descriptors)

        (words, counts) = np.unique(clusters, return_counts=True)

        # compute histogram with word frequencies
        for i, word in enumerate(words):
            hist[word] = counts[i]
        # normalize hist
        hist /= sum(hist)
        return list(hist)

    
    def __extract_sift_descriptors(self, img):
        # convert image to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (w, h) = img.shape

        if w > 500 and h > 500: 
            # calculate new dimensions
            width = int(img.shape[1] * 0.6)
            height = int(img.shape[0] * 0.6)

            # resize image by mantaining aspect ratio
            resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

            

        # extract SIFT keypoints and descriptors from the whole image
        _, desc = self.sift.detectAndCompute(img, mask=None)
        return desc
    
    def __extract_sift_descriptors_from_directory(self, path_to_images):
        directories = os.listdir(path_to_images)

        descriptors = []

        for directory in directories:
                print(directory)
                files = os.listdir(path_to_images + '/' + directory + '/')

                # extract SIFT descriptor from each image
                for i, f_name in enumerate(files):
                    print(f'\r{i+1}/{len(files)}', end='')
                    img = cv2.imread(path_to_images + '/' + directory + '/' + f_name)
                    desc = self.__extract_sift_descriptors(img)
                    descriptors.extend(desc)
                print('\n')
        return descriptors



if __name__ == "__main__":
    img = cv2.imread('../data/train/1.jpg')
    #features_extractor = BOVWFeaturesExtractor()
    features_extractor = BOVWFeaturesExtractor('./')
    features = features_extractor.extract(img)
    print(len(features))
    # print(features)
    # features_extractor.build_dict('./data/train/', 800, 
    #                                 save_path='./', 
    #                                 path_to_descriptors='./SIFT_descriptors.pckl')
