from bot.Updater import Updater
import os, sys, platform, subprocess
import tensorflow as tf
import random
import numpy as np
from annoy import AnnoyIndex
import cv2
from bot.utils.color_extractor import ColorFeaturesExtractor
from bot.utils.retriever import Retriever
from bot.utils.utils import get_names_from_indexes
import pandas as pd
import bot.secrets

unknown_threshold = 0.5

base_dir = '.'

index_dir = os.path.join(base_dir, 'indexes')
data_dir = os.path.join(base_dir, 'data')
img_dir = os.path.join(data_dir, 'train')

names_df_path = os.path.join(data_dir, 'train_filtered.csv')

model_path = os.path.join(data_dir, 'model.h5')

def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension


def imageHandler(bot, message, chat_id, local_filename):
    print(local_filename)
    # send message to user
    # bot.sendMessage(chat_id, "Welcome")
    # # set matlab command
    # if 'Linux' in platform.system():
    #     matlab_cmd = '/usr/local/bin/matlab'
    # else:
    #     matlab_cmd = '"C:\\Program Files\\MATLAB\\R2016a\\bin\\matlab.exe"'
    # # set command to start matlab script "edges.m"
    # cur_dir = os.path.dirname(os.path.realpath(__file__))
    # cmd = matlab_cmd + " -nodesktop -nosplash -nodisplay -wait -r \"addpath(\'" + cur_dir + "\'); edges(\'" + local_filename + "\'); quit\""
    # # lunch command
    # subprocess.call(cmd,shell=True)
    # # send back the manipulated image
    # dirName, fileBaseName, fileExtension = fileparts(local_filename)
    # new_fn = os.path.join(dirName, fileBaseName + '_ok' + fileExtension)
    #new_fn = '/home/rpo/Downloads/Telegram Desktop/photo_2020-12-28_17-24-20.jpg'

    X = loadimg(local_filename)
    pred = model.predict(X)

    detected_class, confidence_lvl = softmax2class(
        pred[0], classes, threshold=unknown_threshold)

    detected_class = detected_class.upper()

    bot.sendMessage(chat_id, f"""
I bet this is a **{detected_class}** with a confidence of {confidence_lvl}!
""")

    img = cv2.imread(local_filename)
    img_features = cfe.extract(img, True)
    indexes = retriever.retrieve(
        img_features, retrieval_mode='color', n_neighbours=5, include_distances=False)
    names = get_names_from_indexes(indexes, names_df)

    names = [os.path.join(img_dir, name) for name in names]

    bot.sendMediaGroup(chat_id, names, "Here some similar images!")

def loadimg(img_path):

    im = tf.keras.preprocessing.image.load_img(
        img_path,
        target_size=(300, 300, 3)
    )
    imarr = tf.keras.preprocessing.image.img_to_array(im)
    imarr = tf.keras.applications.efficientnet.preprocess_input(imarr)

    return np.array([imarr])

def softmax2class(softmax, classes, threshold=0.5, unknown='unknown'):
  max = softmax.max()
  if max >= threshold:
    argmax = softmax.argmax()
    return classes[argmax], max
  else:
    return unknown, max

if __name__ == "__main__":

    model = tf.keras.models.load_model(model_path)
    classes = [
		'trousers',
        'shoe',
        'shorts',
        'jacket',
        'sweatshirt',
        'elegant_jacket',
        'high_heels_shoe',
        't_shirt',
        'bag'
        ]

    cfe = ColorFeaturesExtractor((24,26,3), 0.6)

    retriever = Retriever(index_dir)

    names_df = pd.read_csv(names_df_path)

    bot_id = bot.secrets.bot_id
    updater = Updater(bot_id)
    updater.setPhotoHandler(imageHandler)
    updater.start()