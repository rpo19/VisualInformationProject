#!/usr/bin/env python
import pandas as pd
import sys
import os


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


# neural network features
print('creating neural network index...')
create_index(os.path.join(basedir, 'data/nn_features.csv'),
    os.path.join(basedir, 'indexes/'), 'neural_network', 'euclidean')

# color features
print('creating color features index...')
create_index(os.path.join(basedir, 'data/nn_features.csv'),
    os.path.join(basedir, 'indexes/'), 'color', 'euclidean')

# HOG features
print('creating hog features index...')
create_index(os.path.join(basedir, 'data/nn_features.csv'),
    os.path.join(basedir, 'indexes/'), 'hog', 'euclidean')
