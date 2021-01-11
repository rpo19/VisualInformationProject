#!/usr/bin/env python
import pandas as pd
import sys
import os
from sklearn.decomposition import PCA
import pickle
import numpy as np

sys.path.insert(0, '../')

basedir = '..'

from bot.utils.retriever import Retriever

# Create indexes

def create_index(features_path, index_dir_path, retrieval_mode, metric):
    # read features
    df_features = pd.read_csv(features_path, sep=',', header=None)
    # instantiate retriever
    retriever = Retriever(index_dir_path)
    # create index
    retriever.create_index(df_features, retrieval_mode=retrieval_mode, metric = metric)

if not os.path.isdir(os.path.join(basedir, 'indexes')):
    os.mkdir(os.path.join(basedir, 'indexes'))

## neural network features efficientnet b3 fine tuned last layer

# read nn features
nn_features = pd.read_csv(os.path.join(basedir, 'data/nn_features.csv'), sep=',', header=None)

# create pca
pca_nn = PCA(100)
# fit pca
pca_nn.fit(nn_features)

# save pca model
with open(os.path.join(basedir, 'data/pca_nn.pckl'), 'wb') as handle:
                    pickle.dump(pca_nn, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
# transform features
nn_features_pca = pca_nn.transform(nn_features)
print('nn pca dimensions', nn_features_pca.shape)

# save new features
np.savetxt(os.path.join(basedir, 'data/nn_features_pca.csv'), nn_features_pca, delimiter=',')

print('creating neural network index...')
create_index(os.path.join(basedir, 'data/nn_features_pca.csv'),
    os.path.join(basedir, 'indexes/'), 'neural_network_pca', 'angular')

## color features
print('creating color features index...')
create_index(os.path.join(basedir, 'data/color_features_center_subregions.csv'),
    os.path.join(basedir, 'indexes/'), 'color_center_subregions', 'euclidean')

## HOG features

# read hog features
hog_features = pd.read_csv(os.path.join(basedir, 'data/hog_features.csv'), sep=',', header=None)

# create pca
pca_hog = PCA(100)
# fit pca
pca_hog.fit(hog_features)

# save pca model
with open(os.path.join(basedir, 'data/pca_hog.pckl'), 'wb') as handle:
                    pickle.dump(pca_hog, handle, protocol=pickle.HIGHEST_PROTOCOL)

# transform features
hog_features_pca = pca_hog.transform(hog_features)
print('hog pca dimension', hog_features_pca.shape)

# save new features
np.savetxt(os.path.join(basedir, 'data/hog_features_pca.csv'), hog_features_pca, delimiter=',')

print('creating hog features index...')
create_index(os.path.join(basedir, 'data/hog_features_pca.csv'),
    os.path.join(basedir, 'indexes/'), 'hog_pca', 'euclidean')

## resnet features

# read nn features
nn_features = pd.read_csv(os.path.join(basedir, 'data/nn_resnet_features.csv')
    , sep=',', header=None)

# create pca
pca_nn = PCA(100)
# fit pca
pca_nn.fit(nn_features)

# save pca model
with open(os.path.join(basedir, 'data/pca_nn_resnet.pckl'), 'wb') as handle:
                    pickle.dump(pca_nn, handle, protocol=pickle.HIGHEST_PROTOCOL)

# transform features
nn_features_pca = pca_nn.transform(nn_features)
print('dimension', nn_features_pca.shape)

# save new features
np.savetxt(os.path.join(basedir, 'data/nn_resnet_features_pca.csv'),
    nn_features_pca, delimiter=',')

print('creating resnet index')
create_index(os.path.join(basedir, 'data/nn_resnet_features_pca.csv'),
    os.path.join(basedir, 'indexes/'), 'neural_network_resnet_pca', 'angular')
