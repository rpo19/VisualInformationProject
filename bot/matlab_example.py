from Updater import Updater
import os, sys, platform, subprocess

def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension


def imageHandler(bot, message, chat_id, local_filename):
	print(local_filename)
	# send message to user
	bot.sendMessage(chat_id, "Benvenuto ZanCoglione!")
	# bot.sendMessage(chat_id, "Hi, please wait until the image is ready")
	# # set matlab command
	# if 'Linux' in platform.system():
	# 	matlab_cmd = '/usr/local/bin/matlab'
	# else:
	# 	matlab_cmd = '"C:\\Program Files\\MATLAB\\R2016a\\bin\\matlab.exe"'
	# # set command to start matlab script "edges.m"
	# cur_dir = os.path.dirname(os.path.realpath(__file__))
	# cmd = matlab_cmd + " -nodesktop -nosplash -nodisplay -wait -r \"addpath(\'" + cur_dir + "\'); edges(\'" + local_filename + "\'); quit\""
	# # lunch command
	# subprocess.call(cmd,shell=True)
	# # send back the manipulated image
	# dirName, fileBaseName, fileExtension = fileparts(local_filename)
	# new_fn = os.path.join(dirName, fileBaseName + '_ok' + fileExtension)
	new_fn = '/home/rpo/Downloads/Telegram Desktop/photo_2020-12-28_17-24-20.jpg'
	bot.sendImage(chat_id, new_fn, "che bella ciabatta")



if __name__ == "__main__":
	bot_id = '1478693264:AAG-4qWeWjEIDl2hvV0khOuO4-zn2w2QCrQ'
	updater = Updater(bot_id)
	updater.setPhotoHandler(imageHandler)
	updater.start()