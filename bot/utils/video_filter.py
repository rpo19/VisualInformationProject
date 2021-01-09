import cv2
from style_transfer import superpixelate
import numpy as np
from skimage.segmentation import slic
from skimage import color
from skimage.measure import regionprops
from PIL import Image, ImageDraw
import moviepy.editor as mp
import random


class GifMaker():

	def to_mosaic_gif(self, img_path, n_segments = 150, segments_per_frame = 3):
		img = cv2.imread(img_path)
		# generate superpixels
		segments = slic(img, n_segments = n_segments, sigma = 5)
		# generate image with superpixels avg color
		superpixels_image = color.label2rgb(segments, img, kind='avg')
		superpixels_image = cv2.normalize(superpixels_image, None, alpha = 0, 
											beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)

		mask = np.zeros(img.shape[:2], dtype = "uint8")
		
		n_segments = len(np.unique(segments))
		frames = []

		for (i, segVal) in enumerate(np.unique(segments)):
			# construct a mask for the segment
			mask[segments == segVal] = 255

			a = cv2.bitwise_and(superpixels_image, superpixels_image, mask = mask)
			a = np.uint8(a)
			a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

			if i % segments_per_frame == 0:
				n_segments -= segments_per_frame
				frames.append(Image.fromarray(a))
		
		if n_segments > 0:
			frames.append(Image.fromarray(a))

		filename_with_extension = img_path.split('/')[-1]
		path = img_path.split('/')[0:-1]
		filename = filename_with_extension.split('.')[0]

		
		self.__save_gif(path, filename, frames)
		self.__to_mp4(path, filename)
	

	
	def __save_gif(self, path, filename, frames):
		save_path = '/'.join(path) + '/' + filename + '.gif'
		print(save_path)
		frames[0].save(save_path,
               save_all=True, format='GIF', append_images=frames[1:],
			   optimize=True, quality=20, duration=1, loop=0)
	
	def __to_mp4(self, path, filename):
		base_path = '/'.join(path)
		read_path = base_path + '/' + filename + '.gif'
		save_path = base_path + '/' + filename + '.mp4'
		clip = mp.VideoFileClip(read_path)
		clip.write_videofile('prova.mp4')


if __name__ == "__main__":
	img_path = './data/prova_gif.jpg'
	
	g = GifMaker()

	g.to_mosaic_gif(img_path)
