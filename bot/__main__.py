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
import bot.utils.filter_input as filter_input
import bot.utils.PickleDBExtended

unknown_threshold = 0
blur_threshold = 0.3
dark_threshold = 0.3

base_dir = '.'

index_dir = os.path.join(base_dir, 'indexes')
data_dir = os.path.join(base_dir, 'data')
img_dir = os.path.join(data_dir, 'train')

names_df_path = os.path.join(data_dir, 'retrieval_base.csv')

model_path = os.path.join(data_dir, 'model.h5')

db_path = os.path.join(data_dir, 'state.db')

def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension

def textHandler(bot, message, chat_id, text):
    #print(f'{chat_id}: {message} text: {text}')

    state = db.get(chat_id) or 'tostart'

    print('state', state)

    if text == '/stop':
        db.set(chat_id, 'tostart')
        bot.sendMessage(chat_id, 'Hi, send /start to begin!', keyboard=[['/start']])

    elif state == 'tostart':
        if text == '/start':
            db.set(chat_id, 'wait4command')
            bot.sendMessage(chat_id, 'What do you want to do?',
                keyboard=[['send an image','nothing']])
        else:
            bot.sendMessage(chat_id, 'Hi, send /start to begin!', keyboard=[['/start']])

    elif state == 'wait4command':
        if text == 'send an image':
            db.set(chat_id, 'wait4image')
            bot.sendMessage(chat_id, 'Send an image then!')
        else:
            db.set(chat_id, 'tostart')
            bot.sendMessage(chat_id, 'Hi, send /start to begin!', keyboard=[['/start']])

    elif state == 'wait4image':
        bot.sendMessage(chat_id, "I'm waiting for your image...")       



def imageHandler(bot, message, chat_id, img_path):
    state = db.get(chat_id) or 'tostart'
    if state != 'wait4image':
        bot.sendMessage(chat_id, 'Hi, send /start to begin!', keyboard=[['/start']])
        return

    state = db.set(chat_id, 'tostart')

    print(img_path)

    # quality check
    quality_check = True
    # blur
    is_blurred, sharpness = filter_input.is_blurred(img_path, blur_threshold)
    if is_blurred:
        bot.sendMessage(chat_id, f"The image is blurred! Sharpness: {sharpness}")
        print("image is blurred!")
        quality_check = False

    is_dark, brightness = filter_input.is_dark(img_path, dark_threshold)
    if is_dark:
        bot.sendMessage(chat_id, f"The image is dark! Brightness: {brightness}")
        print("image is dark!")
        quality_check = False

    if not quality_check:
        bot.sendMessage(chat_id, "I'm sorry your image didn't pass the quality check!")
        print("quality check failed!")
        return        

    X = loadimg(img_path)
    pred = model.predict(X)

    detected_class, confidence_lvl = softmax2class(
        pred[0], classes, threshold=unknown_threshold)

    detected_class = detected_class.upper()

    bot.sendMessage(chat_id, f"""
I bet this is a **{detected_class}** with a confidence of {confidence_lvl}!
""")

    img = cv2.imread(img_path)
    img_features = cfe.extract(img, True)
    indexes = retriever.retrieve(
        img_features, retrieval_mode='color', n_neighbours=5, include_distances=False)
    names = get_names_from_indexes(indexes, names_df)

    names = [os.path.join(img_dir, name) for name in names]

    bot.sendMediaGroup(chat_id, names, "Here some similar images!")

    bot.sendMessage(chat_id, 'Hi, send /start to begin!', keyboard=[['/start']])

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
    classes = ['bag',
                'elegant_jacket',
                'high_heels_shoe',
                'jacket',
                'shoe',
                'shorts',
                'sweatshirt',
                't_shirt',
                'trousers',
                'unknown'
              ]

    cfe = ColorFeaturesExtractor((24,26,3), 0.6)

    retriever = Retriever(index_dir)

    names_df = pd.read_csv(names_df_path)

    db = bot.utils.PickleDBExtended.load(db_path, True)

    bot_id = bot.secrets.bot_id
    updater = Updater(bot_id)
    updater.setPhotoHandler(imageHandler)
    updater.setTextHandler(textHandler)
    updater.start()