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
from bot.utils.color_extractor import ColorFeaturesExtractor
from bot.utils.retriever import Retriever
from bot.utils.utils import get_names_from_indexes
import pandas as pd
import bot.secrets
import bot.utils.filter_input as filter_input
import bot.utils.PickleDBExtended
from bot.utils import style_transfer

BTN_STOP = '/stop'
BTN_START = '/start'

KEY_SIMILARITY = 'KEY_SIMILARITY_{}'
KEY_STYLE_BASE_IMG = 'KEY_STYLE_BASE_{}'
KEY_FILTER = 'KEY_FILTER_{}'
# only needed when quality check failed if users choose to continue
KEY_SIMILAR_IMAGE_IMG = 'KEY_SIMILAR_IMAGE_{}'
KEY_QUALITY_CONTINUE_PREVSTATE = 'KEY_QUALITY_CONTINUE_PREVSTATE_{}'



STATE_TOSTART = 'STATE_TO_START'
STATE_WAIT_ACTION = 'STATE_WAIT_ACTION'
STATE_WAIT_SIMILAR_IMAGE = 'STATE_WAIT_SIMILAR'
STATE_WAIT_STYLE_BASE = 'STATE_WAIT_STYLE_BASE'
STATE_WAIT_STYLE_STYLE = 'STATE_WAIT_STYLE_STYLE'
STATE_CHOOSE_SIMILARITY = 'STATE_CHOOSE_SIMILARITY'
STATE_QUALITY_ASK_CONTINUE = 'STATE_QUALITY_ASK_CONTINUE'
STATE_CHOOSE_FILTER = 'STATE_CHOOSE_FILTER'
STATE_WAIT_FOR_APPLY_FILTER = 'STATE_WAIT_APPLYING_FILTER'

MSG_START = 'Hi, send /start to begin!'
MSG_PROMPT_ACTION = 'What do you want to do?'
MSG_SEND_FOR_SIMILAR = 'Send an image of a clothing'
MSG_SEND_FOR_STYLE_BASE = 'Send an image of a clothing to which you want to apply a new style'
MSG_SEND_FOR_STYLE_STYLE = 'Send a style texture'
MSG_CHOOSE_SIMILARITY = 'Choose which kind of similarity to look for'
MSG_QUALITY_CHECK_FAILED = "The image sent didn't pass the quality check due to {}. Continue anyway?"
MSG_SEND_FOR_APPLY_FILTER_BASE = 'Which kind of filter you want to apply?'
MSG_SEND_FOR_APPLY_FILTER = 'Send an image to apply the selected filter'
MSG_FILTER_DONE = 'Hope you like it!'

BTN_SEARCH_SIMILAR = 'Retrieve similar products'
BTN_STYLE_TRANSFER = 'Apply style to a product'
BTN_APPLY_FILTER = 'Apply filter to image'
BTN_FILTER_MOSAIC = 'Mosaic'
BTN_FILTER_HANDDRAW = 'Hand drawing'
BTN_SIMIL_COLOR = 'by Color'
BTN_SIMIL_SHAPE = 'by Shape'
BTN_SIMIL_NEURAL = 'by Neural Net'
BTN_YES = 'Yes'
BTN_NO = 'No'

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

    state = db.get(chat_id) or STATE_TOSTART

    print('state', state)

    if text == BTN_STOP:
        db.set(chat_id, STATE_TOSTART)
        bot.sendMessage(chat_id, MSG_START, keyboard=[[BTN_START]])

    elif state == STATE_TOSTART:
        if text == BTN_START:
            bot.sendMessage(chat_id, MSG_PROMPT_ACTION,
                            keyboard=[
                                [BTN_SEARCH_SIMILAR, BTN_STYLE_TRANSFER],
                                [BTN_APPLY_FILTER],
                                [BTN_STOP]
                            ])
            db.set(chat_id, STATE_WAIT_ACTION)
        else:
            bot.sendMessage(chat_id, MSG_START, keyboard=[[BTN_START]])

    elif state == STATE_WAIT_ACTION:
        if text == BTN_SEARCH_SIMILAR:
            bot.sendMessage(chat_id, MSG_CHOOSE_SIMILARITY,
                            keyboard=[
                                [BTN_SIMIL_COLOR, BTN_SIMIL_SHAPE, BTN_SIMIL_NEURAL],
                                [BTN_STOP]
                            ])
            db.set(chat_id, STATE_CHOOSE_SIMILARITY)

        elif text == BTN_STYLE_TRANSFER:
            bot.sendMessage(chat_id, MSG_SEND_FOR_STYLE_BASE)
            db.set(chat_id, STATE_WAIT_STYLE_BASE)

        elif text == BTN_APPLY_FILTER:
            bot.sendMessage(chat_id, MSG_SEND_FOR_APPLY_FILTER_BASE,
                            keyboard=[
                                [BTN_FILTER_MOSAIC, BTN_FILTER_HANDDRAW],
                                [BTN_STOP]
                            ])
            db.set(chat_id, STATE_CHOOSE_FILTER)

        else:
            db.set(chat_id, STATE_TOSTART)
            bot.sendMessage(chat_id, MSG_START, keyboard=[[BTN_START]])

    elif state == STATE_CHOOSE_SIMILARITY:
        if text not in [BTN_SIMIL_COLOR, BTN_SIMIL_SHAPE, BTN_SIMIL_NEURAL]:
            # exception todo
            pass

        # note similarity for chat_id
        db.set(KEY_SIMILARITY.format(chat_id), text)

        bot.sendMessage(chat_id, MSG_SEND_FOR_SIMILAR)
        db.set(chat_id, STATE_WAIT_SIMILAR_IMAGE)
    
    elif state == STATE_CHOOSE_FILTER:
        if text not in [BTN_FILTER_MOSAIC, BTN_FILTER_HANDDRAW]:
            # exception todo
            pass

        db.set(KEY_FILTER.format(chat_id), text)
        
        bot.sendMessage(chat_id, MSG_SEND_FOR_APPLY_FILTER)
        db.set(chat_id, STATE_WAIT_FOR_APPLY_FILTER)

    elif state == STATE_QUALITY_ASK_CONTINUE:
        if text == BTN_YES:
            prevstate = db.get(KEY_QUALITY_CONTINUE_PREVSTATE.format(chat_id))

            if prevstate == STATE_WAIT_STYLE_BASE:
                bot.sendMessage(chat_id, MSG_SEND_FOR_STYLE_STYLE)
                db.set(chat_id, STATE_WAIT_STYLE_STYLE)

            elif prevstate == STATE_WAIT_SIMILAR_IMAGE:
                img_path = db.get(KEY_SIMILAR_IMAGE_IMG.format(chat_id)) or None

                if not img_path:
                    #exception
                    pass

                similarity = db.get(KEY_SIMILARITY.format(chat_id)) or None
                print('similarity', similarity)

                if similarity not in [BTN_SIMIL_COLOR, BTN_SIMIL_SHAPE, BTN_SIMIL_NEURAL]:
                    # exception todo
                    pass
                
                do_similarity(bot, chat_id, similarity, img_path)

                db.set(chat_id, STATE_TOSTART)

            else:
                # todo
                pass
        else:
            db.set(chat_id, STATE_TOSTART)
            bot.sendMessage(chat_id, MSG_START, keyboard=[[BTN_START]])
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
    img_base = db.get(KEY_STYLE_BASE_IMG.format(chat_id)) or None

    if not img_base:
        # exception
        pass

    bot.sendMessage(chat_id, 'dai implementami!')

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

    img = cv2.imread(img_path)
    img_features = cfe.extract(img, True)
    indexes = retriever.retrieve(
        img_features, retrieval_mode='color', n_neighbours=5, include_distances=False)
    names = get_names_from_indexes(indexes, names_df)

    names = [os.path.join(img_dir, name) for name in names]

    bot.sendMediaGroup(chat_id, names, "Here some similar images!")

    bot.sendMessage(chat_id, MSG_START, keyboard=[[BTN_START]])
    
def do_filter(bot, chat_id, filter, img_path):
    bot.sendMessage(chat_id, f"""
Applying {filter} filter to image
""")

    img = cv2.imread(img_path)

    if filter == BTN_FILTER_HANDDRAW:
        img = style_transfer.get_image_details(img)
    elif filter == BTN_FILTER_MOSAIC:
        img = style_transfer.superpixelate(img)

    cv2.imwrite(img_path, img)
    
    bot.sendImage(chat_id, img_path, MSG_FILTER_DONE)
    bot.sendMessage(chat_id, MSG_START, keyboard=[[BTN_START]])
    


def imageHandler(bot, message, chat_id, img_path):
    state = db.get(chat_id) or STATE_TOSTART
    print(state)

    if state == STATE_WAIT_STYLE_BASE:
        # note img path
        db.set(KEY_STYLE_BASE_IMG.format(chat_id), img_path)

        # controllo input
        is_blur, is_dark = quality_control_blur_dark(
            img_path, BLUR_THRESHOLD, DARK_THRESHOLD)
        print(is_dark)
        # todo aggiungere controllo margini
        if is_blur:
            cause = 'blurriness'

            bot.sendMessage(chat_id, MSG_QUALITY_CHECK_FAILED.format(cause),
                            keyboard=[
                [BTN_YES, BTN_NO],
                [BTN_STOP]
            ])

            # annoto stato precedente
            db.set(KEY_QUALITY_CONTINUE_PREVSTATE.format(chat_id), STATE_WAIT_STYLE_BASE)
            
            db.set(chat_id, STATE_QUALITY_ASK_CONTINUE)

            # todo: se non continua controllare che immagine annotata non dia problemi
        else:

            bot.sendMessage(chat_id, MSG_SEND_FOR_STYLE_STYLE)
            db.set(chat_id, STATE_WAIT_STYLE_STYLE)

    elif state == STATE_WAIT_STYLE_STYLE:
        do_style_transfer(bot, chat_id, img_path)

        db.set(chat_id, STATE_TOSTART)

    elif state == STATE_WAIT_SIMILAR_IMAGE:
        # prendo la similarità scelta prima
        similarity = db.get(KEY_SIMILARITY.format(chat_id)) or None
        print('similarity', similarity)

        if similarity not in [BTN_SIMIL_COLOR, BTN_SIMIL_SHAPE, BTN_SIMIL_NEURAL]:
            # exception todo
            pass

        # controllo input
        is_blur, is_dark = quality_control_blur_dark(
            img_path, BLUR_THRESHOLD, DARK_THRESHOLD)

        # todo aggiungere controllo margini
        if is_blur:
            cause = 'blurriness'
            bot.sendMessage(chat_id, MSG_QUALITY_CHECK_FAILED.format(cause),
                            keyboard=[
                [BTN_YES, BTN_NO],
                [BTN_STOP]
            ])

            # annoto stato precedente
            db.set(KEY_QUALITY_CONTINUE_PREVSTATE.format(chat_id), STATE_WAIT_SIMILAR_IMAGE)
            # annoto immagine
            db.set(KEY_SIMILAR_IMAGE_IMG.format(chat_id), img_path)
            
            db.set(chat_id, STATE_QUALITY_ASK_CONTINUE)

            # todo: se non continua controllare che immagine annotata non dia
            # problemi
        else:
            do_similarity(bot, chat_id, similarity, img_path)
            db.set(chat_id, STATE_TOSTART)

    elif state == STATE_WAIT_FOR_APPLY_FILTER:

        selected_filter = db.get(KEY_FILTER.format(chat_id)) or None
        
        if selected_filter not in [BTN_FILTER_MOSAIC, BTN_FILTER_HANDDRAW]:
            # exception todo
            pass
        
        do_filter(bot, chat_id, selected_filter, img_path)
        db.set(chat_id, STATE_TOSTART)


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

    retriever = Retriever(index_dir, load_all=True)

    names_df = pd.read_csv(names_df_path)

    db = bot.utils.PickleDBExtended.load(db_path, True)

    bot_id = bot.secrets.bot_id
    updater = Updater(bot_id)
    updater.setPhotoHandler(imageHandler)
    updater.setTextHandler(textHandler)
    updater.start()
