from bot.Updater import Updater
import os
import sys
import platform
import subprocess
import tensorflow as tf
import random
import numpy as np
from annoy import AnnoyIndex
import cv2
from skimage.io import imread
from bot.utils.color_extractor import ColorFeaturesExtractor, Mode
from bot.utils.hog_extractor import HogFeaturesExtractor
from bot.utils.retriever import Retriever
from bot.utils.utils import get_names_from_indexes
import pandas as pd
import bot.secrets
import bot.utils.filter_input as filter_input
import bot.utils.PickleDBExtended
from bot.utils import style_transfer
# enums
from bot.enums.state import State
from bot.enums.message import Message
from bot.enums.button import Button
from bot.enums.key import Key


UNKNOWN_THRESHOLD = 0
BLUR_THRESHOLD = 0.3
DARK_THRESHOLD = 30

base_dir = '.'

index_dir = os.path.join(base_dir, 'indexes')
data_dir = os.path.join(base_dir, 'data')
img_dir = os.path.join(data_dir, 'train')

names_df_path = os.path.join(data_dir, 'retrieval_base.csv')

model_path = os.path.join(data_dir, 'model.h5')
blur_model_path = os.path.join(data_dir, 'blur_model_4k.h5')

db_path = os.path.join(data_dir, 'state.db')


def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension


def textHandler(bot, message, chat_id, text):
    #print(f'{chat_id}: {message} text: {text}')

    state = db.get(chat_id) or State.STATE_TOSTART

    print('state', state)

    if text == Button.BTN_STOP:
        db.set(chat_id, State.STATE_TOSTART)
        bot.sendMessage(chat_id, Message.MSG_START, keyboard=[[Button.BTN_START]])

    elif state == State.STATE_TOSTART:
        if text == Button.BTN_START:
            bot.sendMessage(chat_id, Message.MSG_PROMPT_ACTION,
                            keyboard=[
                                [Button.BTN_SEARCH_SIMILAR, Button.BTN_STYLE_TRANSFER],
                                [Button.BTN_APPLY_FILTER],
                                [Button.BTN_STOP]
                            ])
            db.set(chat_id, State.STATE_WAIT_ACTION)
        else:
            bot.sendMessage(chat_id, Message.MSG_START, keyboard=[[Button.BTN_START]])

    elif state == State.STATE_WAIT_ACTION:
        if text == Button.BTN_SEARCH_SIMILAR:
            bot.sendMessage(chat_id, Message.MSG_CHOOSE_SIMILARITY,
                            keyboard=[
                                [Button.BTN_SIMIL_COLOR, Button.BTN_SIMIL_SHAPE, Button.BTN_SIMIL_NEURAL],
                                [Button.BTN_STOP]
                            ])
            db.set(chat_id, State.STATE_CHOOSE_SIMILARITY)

        elif text == Button.BTN_STYLE_TRANSFER:
            bot.sendMessage(chat_id, Message.MSG_SEND_FOR_STYLE_BASE)
            db.set(chat_id, State.STATE_WAIT_STYLE_BASE)

        elif text == Button.BTN_APPLY_FILTER:
            bot.sendMessage(chat_id, Message.MSG_SEND_FOR_APPLY_FILTER_BASE,
                            keyboard=[
                                [Button.BTN_FILTER_MOSAIC, Button.BTN_FILTER_HANDDRAW],
                                [Button.BTN_STOP]
                            ])
            db.set(chat_id, State.STATE_CHOOSE_FILTER)

        else:
            db.set(chat_id, State.STATE_TOSTART)
            bot.sendMessage(chat_id, Message.MSG_START, keyboard=[[Button.BTN_START]])

    elif state == State.STATE_CHOOSE_SIMILARITY:
        if text not in [Button.BTN_SIMIL_COLOR, Button.BTN_SIMIL_SHAPE, Button.BTN_SIMIL_NEURAL]:
            # exception todo
            pass

        # note similarity for chat_id
        db.set(Key.KEY_SIMILARITY.format(chat_id), text)

        bot.sendMessage(chat_id, Message.MSG_SEND_FOR_SIMILAR)
        db.set(chat_id, State.STATE_WAIT_SIMILAR_IMAGE)
    
    elif state == State.STATE_CHOOSE_FILTER:
        if text not in [Button.BTN_FILTER_MOSAIC, Button.BTN_FILTER_HANDDRAW]:
            # exception todo
            pass

        db.set(Key.KEY_FILTER.format(chat_id), text)
        
        bot.sendMessage(chat_id, Message.MSG_SEND_FOR_APPLY_FILTER)
        db.set(chat_id, State.STATE_WAIT_FOR_APPLY_FILTER)

    elif state == State.STATE_QUALITY_ASK_CONTINUE:
        if text == Button.BTN_YES:
            prevstate = db.get(Key.KEY_QUALITY_CONTINUE_PREVSTATE.format(chat_id))

            if prevstate == State.STATE_WAIT_STYLE_BASE:
                bot.sendMessage(chat_id, Message.MSG_SEND_FOR_STYLE_STYLE)
                db.set(chat_id, State.STATE_WAIT_STYLE_STYLE)

            elif prevstate == State.STATE_WAIT_SIMILAR_IMAGE:
                img_path = db.get(Key.KEY_SIMILAR_IMAGE_IMG.format(chat_id)) or None

                if not img_path:
                    #exception
                    pass

                similarity = db.get(Key.KEY_SIMILARITY.format(chat_id)) or None
                print('similarity', similarity)

                if similarity not in [Button.BTN_SIMIL_COLOR, Button.BTN_SIMIL_SHAPE, Button.BTN_SIMIL_NEURAL]:
                    # exception todo
                    pass
                
                do_similarity(bot, chat_id, similarity, img_path)

                db.set(chat_id, State.STATE_TOSTART)

            else:
                # todo
                pass
        else:
            db.set(chat_id, State.STATE_TOSTART)
            bot.sendMessage(chat_id, Message.MSG_START, keyboard=[[Button.BTN_START]])
    # else: far capire all utente cosa si aspetta


def quality_control_blur_dark(img_path, blur_threshold, dark_threshold):
    is_blurred = filter_input.is_blurred(img_path, blur_model)
    if is_blurred:
        print(f"image is blurred!")

    is_dark, img_fixed = filter_input.fix_darkness(img_path, dark_threshold)
    if is_dark:
        cv2.imwrite(img_path, img_fixed)
        print(f"image is dark!")

    return is_blurred, is_dark

def do_style_transfer(bot, chat_id, img_style):
    # fare cose
    # ritornare immagini
    img_base = db.get(Key.KEY_STYLE_BASE_IMG.format(chat_id)) or None

    if not img_base:
        # exception
        pass
    
    # TODO: vedere se ok controllo sfondo uniforme (es: scarpa bianca su sfondo bianco fa casini con bilateral=True)
    use_bilateral = not filter_input.has_uniform_bg(img_base)
    print('use bilateral:', use_bilateral)
    # TODO: vedere se usare o no pesi di default
    img_new = style_transfer.style_transfer(img_base, img_style, maximize_color=True, 
                                             bilateral=use_bilateral, color_weight=0.6, details_weight=0.4,
                                             crop = False, white_bg = False)
    cv2.imwrite(img_base, img_new)
    bot.sendImage(chat_id, img_base, Message.MSG_STYLE_TRANSFER_DONE)
    return img_base

def do_similarity(bot, chat_id, similarity, img_path):
    # faccio cose
    # ritorno immagini

    bot.sendMessage(chat_id, f"""
chosen similarity: {similarity}
""")

    X = loadimg(img_path)
    pred = model.predict(X)

    detected_class, confidence_lvl = softmax2class(
        pred[0], classes, threshold=UNKNOWN_THRESHOLD)

    detected_class = detected_class.upper()

    bot.sendMessage(chat_id, f"""
I bet this is a **{detected_class}** with a confidence of {confidence_lvl}!
""")

    # retrieve similar with the selected modality
    if similarity == Button.BTN_SIMIL_COLOR:
        indexes = do_color_retrieval(img_path)
    elif similarity == Button.BTN_SIMIL_SHAPE:
        indexes = do_shape_retrieval(img_path)
    elif similarity == Button.BTN_SIMIL_NEURAL:
        indexes = do_neural_network_retrieval(img_path)

    names = get_names_from_indexes(indexes, names_df)

    names = [os.path.join(img_dir, name) for name in names]

    bot.sendMediaGroup(chat_id, names, Message.MSG_RETRIEVAL__DONE)

    bot.sendMessage(chat_id, Message.MSG_START, keyboard=[[Button.BTN_START]])

def do_color_retrieval(img_path):
    img = cv2.imread(img_path)
    img_features = cfe.extract(img, mode=Mode.CENTER_ONLY)
    indexes = retriever.retrieve(
        img_features, retrieval_mode='color', n_neighbours=5, include_distances=False)
    return indexes

def do_neural_network_retrieval(img_path):
    img = loadimg(img_path)
    img_features = model.predict(img)[1][0]
    indexes = retriever.retrieve(
        img_features, retrieval_mode='neural_network', n_neighbours=5, include_distances=False)
    return indexes

def do_shape_retrieval(img_path):
    img = imread(img_path)
    img_features = hfe.extract(img)
    indexes = retriever.retrieve(
        img_features, retrieval_mode='hog', n_neighbours=5, include_distances=False)
    return indexes

def do_filter(bot, chat_id, filter, img_path):
    bot.sendMessage(chat_id, f"""
Applying {filter} filter to image
""")

    img = cv2.imread(img_path)

    if filter == Button.BTN_FILTER_HANDDRAW:
        img = style_transfer.get_image_details(img)
    elif filter == Button.BTN_FILTER_MOSAIC:
        img = style_transfer.superpixelate(img)

    cv2.imwrite(img_path, img)
    
    bot.sendImage(chat_id, img_path, Message.MSG_FILTER_DONE)
    bot.sendMessage(chat_id, Message.MSG_START, keyboard=[[Button.BTN_START]])
    


def imageHandler(bot, message, chat_id, img_path):
    state = db.get(chat_id) or State.STATE_TOSTART
    print(state)

    if state == State.STATE_WAIT_STYLE_BASE:
        # note img path
        db.set(Key.KEY_STYLE_BASE_IMG.format(chat_id), img_path)

        # controllo input
        is_blur, is_dark = quality_control_blur_dark(
            img_path, BLUR_THRESHOLD, DARK_THRESHOLD)
        print('is dark:', is_dark)
        print('is blur:', is_blur)
        # check that clothe is bounded and with no disturbed background
        has_clear_margins = filter_input.has_clear_margins(img_path, margin=1)
        print('has clear margins:', has_clear_margins)
        if (is_blur) | (not has_clear_margins):

            if (is_blur) & (not has_clear_margins):
                cause = 'blurriness and margins not clear'
            elif is_blur:
                cause = 'blurriness'
            elif not has_clear_margins:
                cause = 'margins not clear'
        

            bot.sendMessage(chat_id, Message.MSG_QUALITY_CHECK_FAILED.format(cause),
                            keyboard=[
                [Button.BTN_YES, Button.BTN_NO],
                [Button.BTN_STOP]
            ])

            # annoto stato precedente
            db.set(Key.KEY_QUALITY_CONTINUE_PREVSTATE.format(chat_id), State.STATE_WAIT_STYLE_BASE)
            
            db.set(chat_id, State.STATE_QUALITY_ASK_CONTINUE)

            # todo: se non continua controllare che immagine annotata non dia problemi
        else:
            # note style base img
            db.set(Key.KEY_STYLE_BASE_IMG.format(chat_id), img_path)
            bot.sendMessage(chat_id, Message.MSG_SEND_FOR_STYLE_STYLE)
            db.set(chat_id, State.STATE_WAIT_STYLE_STYLE)

    elif state == State.STATE_WAIT_STYLE_STYLE:
        # apply new style
        img_new = do_style_transfer(bot, chat_id, img_path)
        # retrieve similar products
        indexes = do_neural_network_retrieval(img_new)
        names = get_names_from_indexes(indexes, names_df)
        names = [os.path.join(img_dir, name) for name in names]
        bot.sendMediaGroup(chat_id, names, Message.MSG_RETRIEVAL_NEW_STYLE_DONE)
        bot.sendMessage(chat_id, Message.MSG_START, keyboard=[[Button.BTN_START]])
        db.set(chat_id, State.STATE_TOSTART)

    elif state == State.STATE_WAIT_SIMILAR_IMAGE:
        # prendo la similarità scelta prima
        similarity = db.get(Key.KEY_SIMILARITY.format(chat_id)) or None
        print('similarity', similarity)

        if similarity not in [Button.BTN_SIMIL_COLOR, Button.BTN_SIMIL_SHAPE, Button.BTN_SIMIL_NEURAL]:
            # exception todo
            pass

        # controllo input
        is_blur, is_dark = quality_control_blur_dark(
            img_path, BLUR_THRESHOLD, DARK_THRESHOLD)

        # todo aggiungere controllo margini
        if is_blur:
            cause = 'blurriness'
            bot.sendMessage(chat_id, Message.MSG_QUALITY_CHECK_FAILED.format(cause),
                            keyboard=[
                [Button.BTN_YES, Button.BTN_NO],
                [Button.BTN_STOP]
            ])

            # annoto stato precedente
            db.set(Key.KEY_QUALITY_CONTINUE_PREVSTATE.format(chat_id), State.STATE_WAIT_SIMILAR_IMAGE)
            # annoto immagine
            db.set(Key.KEY_SIMILAR_IMAGE_IMG.format(chat_id), img_path)
            
            db.set(chat_id, State.STATE_QUALITY_ASK_CONTINUE)

            # todo: se non continua controllare che immagine annotata non dia
            # problemi
        else:
            do_similarity(bot, chat_id, similarity, img_path)

            db.set(chat_id, State.STATE_TOSTART)

    elif state == State.STATE_WAIT_FOR_APPLY_FILTER:

        selected_filter = db.get(Key.KEY_FILTER.format(chat_id)) or None
        
        if selected_filter not in [Button.BTN_FILTER_MOSAIC, Button.BTN_FILTER_HANDDRAW]:
            # exception todo
            pass
        
        do_filter(bot, chat_id, selected_filter, img_path)
        db.set(chat_id, State.STATE_TOSTART)


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
    blur_model = tf.keras.models.load_model(blur_model_path)
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

    cfe = ColorFeaturesExtractor((24, 26, 3), 0.6)
    
    hfe = HogFeaturesExtractor()

    retriever = Retriever(index_dir, load_all=True)

    names_df = pd.read_csv(names_df_path)

    db = bot.utils.PickleDBExtended.load(db_path, True)

    bot_id = bot.secrets.bot_id
    updater = Updater(bot_id)
    updater.setPhotoHandler(imageHandler)
    updater.setTextHandler(textHandler)
    updater.start()
