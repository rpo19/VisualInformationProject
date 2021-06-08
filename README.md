# Overview
The developed systems allow the user:
- to retrieve products similar to a photo from the dataset
- to apply a style to an image of a clothing
- to apply effects to images
- to create a GIF from images

User interaction with the system is handled by a Telegram Bot.

For more information look at the presentation []().

# Dataset

Download dataset from https://products-10k.github.io/

Paper at https://arxiv.org/abs/2008.10545

Models and miscellaneous files from the archive `data.7z` you can find [here](https://drive.google.com/drive/folders/1hV-yZ98X1isiRYvwhKSkEviUUUZsPT0i?usp=sharing) and verify with `data.7z.md5`.

# Bot

## Directory structure

The bot expects this directory structure:
```
.
|__data
|  |__model.h5
|  |__blur_model_4k.h5
|  |__retrieval_base.csv
|  |__train
|     |__1.jpg
|     |  ...
|     |__2629.jpg
|
|__bot
|  |__secrets.py
|
|__indexes
   |__color
   |__hog
   |__neural_network
   |__retrieval_modes.ini
```

## Telegram token

Put the Telegram Bot token inside `bot/secrets.py`. Look at the sample
`bot/secrets_sample.py`

## Requirements

All the requirements needed by the bot are listed inside `requirements.txt`.

Supposing you prefer to create a virtualenv here are the steps to create a virtualenv and install the requirements:
```
python -m venv venv

# Linux
# you may need to install some packages; e.g.
# apt-get install ffmpeg libsm6 libxext6
source venv/bin/activate

# Windows
.\venv\Scripts\Activate

pip install -r requirements.txt
```

## Dataset images
Put dataset train images inside `./data/train`

## Retrieval indexes
Put the indexes inside `./indexes` as shown above otherwise create them starting
from feature files `./data/color_features.csv  ./data/hog_features.csv
./data/nn_features.csv`:

once features files are present:
```
cd scripts
python create_indexes.py
```


## Run
```
# Get inside the virtualenv if it exists
# Linux
source venv/bin/activate
# Windows
.\venv\Scripts\Activate

# Run the bot
python -m bot
```