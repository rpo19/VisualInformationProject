import cv2
import os
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

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
    def build_dict(self, path_to_images, n_of_words, save_path, path_to_descriptors=None, descriptors=None):
        if descriptors is None:
            if path_to_descriptors is None:
                descriptors = self.__extract_sift_descriptors_from_directory(path_to_images)
                # save descriptors to file
                with open(path_to_descriptors + 'SIFT_descriptors.pckl', 'wb') as handle:
                    pickle.dump(descriptors, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            else:
                # load descriptors
                with open(path_to_descriptors + 'SIFT_descriptors.pckl', 'rb') as handle:
                    descriptors = pickle.load(handle)
                
                print(len(descriptors))
        
        # instantiate k-mean with n_of_words clusters
        self.kmeans = MiniBatchKMeans(n_clusters = n_of_words, batch_size=300, verbose=1)
        # fit kmeans on all descriptors
        print('Fitting kmeans')
        self.kmeans.fit(descriptors)
        self.kmeans.verbose = 0
        # save dictionary to file
        with open(save_path, 'wb') as handle:
            pickle.dump(self.kmeans, handle, protocol=pickle.HIGHEST_PROTOCOL)    


    # extract features vector
    def extract(self, img):

        # extract SIFT descriptors from image
        descriptors = self.extract_sift_descriptors(img)
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

    
    def extract_sift_descriptors(self, img):
        # convert image to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # (w, h) = img.shape

        # if w > 500 and h > 500: 
        #     # calculate new dimensions
        #     width = int(img.shape[1] * 0.6)
        #     height = int(img.shape[0] * 0.6)

        #     # resize image by mantaining aspect ratio
        #     resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

            

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
                    desc = self.extract_sift_descriptors(img)
                    descriptors.extend(desc)
                print('\n')
        return descriptors


def train_test_splitting(path, test_size = 0.2):
    directories = os.listdir(path)

    train_ranges = []
    test_ranges = []
    for directory in directories:

        files = os.listdir(path + directory)
        
        n_files = len(files)
        n_test = int(len(files) * test_size)
        train_ranges.append((0, n_files - n_test))
        test_ranges.append((n_files - n_test + 1, n_files))
    return train_ranges, test_ranges

def extract_descriptors(path, ranges, extractor, save_path=None):
    directories = os.listdir(path)

    descriptors = []
    labels = []
    for i, directory in enumerate(directories):
        files = os.listdir(path + directory)

        for j in range(ranges[i][0], ranges[i][1]):
            print(f'\r{j+1}/{ranges[i][1]}', end='')
            img = cv2.imread(path + '/' + directory + '/' + files[j])
            desc = extractor.extract_sift_descriptors(img)
            descriptors.extend(desc)
            labels.append(i)
        print('\n')
    
    if save_path is not None:
        with open(save_path + '_descriptors.pckl', 'wb') as handle:
            pickle.dump(descriptors, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        with open(save_path + '_labels.pckl', 'wb') as handle:
            pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    return descriptors, labels

def extract_histograms(path, ranges, extractor):
    directories = os.listdir(path)

    histograms = []
    for i, directory in enumerate(directories):
        files = os.listdir(path + directory)

        for j in range(ranges[i][0], ranges[i][1]):
            print(f'\r{j+1}/{ranges[i][1]}', end='')
            img = cv2.imread(path + '/' + directory + '/' + files[j])
            hist = extractor.extract(img)
            histograms.append(hist)

    return histograms
        
            


if __name__ == "__main__":

    extractor = BOVWFeaturesExtractor()
    # split image in train and test
    train_ranges, test_ranges = train_test_splitting('./data/train_filtered/')

    print('Extract train:')
    if os.path.isfile('./data/train_descriptors.pckl'):
        with open('./data/train_descriptors.pckl', 'rb') as handle:
             train_desc = pickle.load(handle)
        with open('./data/train_labels.pckl', 'rb') as handle:
             train_labels = pickle.load(handle)
    else:
        # extract descriptors and labels from train images
        train_desc, train_labels = extract_descriptors('./data/train_filtered/', train_ranges, extractor, save_path='./data/train')

    print('Extract test:')
    if os.path.isfile('./data/test_descriptors.pckl'):
        with open('./data/test_descriptors.pckl', 'rb') as handle:
             test_desc = pickle.load(handle)
        with open('./data/test_labels.pckl', 'rb') as handle:
             test_labels = pickle.load(handle)
    else:
        # extract descriptors and labels from test images
        test_desc, test_labels = extract_descriptors('./data/train_filtered/', test_ranges, extractor, save_path='./data/test')

    del test_desc

    print('Building dictionary:')
    # build dictionary with train descriptors
    extractor.build_dict('./data/train/', 100, save_path='./data/BOVW_dict_400.pckl', descriptors=train_desc)
    del train_desc

    print('Building training histograms:')
    # build training histograms
    train_histograms = extract_histograms('./data/train_filtered/', train_ranges, extractor)
    

    print('Number of train histograms: ', len(train_histograms))
    print('Number of train labels: ', len(train_labels))

    print('Building test histograms:')
    # build test histograms
    test_histograms = extract_histograms('./data/train_filtered/', test_ranges, extractor)

    print('Number of test histograms: ', len(test_histograms))
    print('Number of test labels: ', len(test_labels))

    # standardize features
    scaler=StandardScaler().fit(train_histograms)
    train_histograms = scaler.transform(train_histograms)
    test_histograms = scaler.transform(test_histograms)

    # train svm
    svm = LinearSVC(max_iter=80000)

    svm.fit(train_histograms, np.array(train_labels))

    # prediction on test set
    predictions = svm.predict(test_histograms)

    # evaluate accuracy
    accuracy = accuracy_score(np.array(test_labels),predictions)

    print(accuracy)


    #img = cv2.imread('../data/train/1.jpg')
    #features_extractor = BOVWFeaturesExtractor()
    # features_extractor = BOVWFeaturesExtractor('./')
    # features = features_extractor.extract(img)
    # print(len(features))
    # print(features)
    # features_extractor.build_dict('./data/train/', 400, 
    #                                 save_path='./data/', 
    #                                 path_to_descriptors='./data/')
