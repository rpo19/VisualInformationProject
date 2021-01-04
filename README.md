# Bot

## Directory structure

The bot expects this directory structure:
```
.
|__data
|  |__model.h5
|  |__train_filtered.csv
|  |__train
|  |__1.jpg
|  |  ...
|  |__2629.jpg
|
|__bot
|  |__secrets.py
|
|__indexes
   |__color
```

## Telegram token

Put the Telegram Bot token inside `bot/secrets.py`. Look at the sample `bot/secrets_sample.py`

## Requirements

All the requirements needed by the bot are listed inside `requirements.txt`.

Supposing you prefer to create a virtualenv here are the steps to create a virtualenv and install the requirements:
```
python -m venv venv

# Linux
source venv/bin/activate

# Windows
.\venv\Scripts\Activate

pip install -r requirements.txt
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