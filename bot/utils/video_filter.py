import cv2
from style_transfer import get_image_details, superpixelate
import numpy as np
from skimage.segmentation import slic
from skimage import color
from skimage.measure import regionprops
from PIL import Image, ImageDraw
import moviepy.editor as mp
import random

img = cv2.imread('./data/prova_gif.jpg')

#img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
segments = slic(img, n_segments = 150, sigma = 5)
superpixels_image = color.label2rgb(segments, img, kind='avg')
superpixels_image = cv2.normalize(superpixels_image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)

mask = np.zeros(img.shape[:2], dtype = "uint8")


frames = []

n_segments = len(np.unique(segments))

unique_segments = list(np.unique(segments))

already_fetched = []

for i, _ in enumerate(np.unique(segments)):

	if type(unique_segments) is not None:
		segVal = random.choice(unique_segments)
		unique_segments.remove(segVal)
		


		mask[segments == segVal] = 255

		a = cv2.bitwise_and(superpixels_image, superpixels_image, mask = mask)
		a = np.uint8(a)

		a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

		if i % 3 == 0:
			n_segments -= 3
			frames.append(Image.fromarray(a))


# for (i, segVal) in enumerate(np.unique(segments)):
# 	# construct a mask for the segment
# 	mask[segments == segVal] = 255
# 	#mask_complement = 255 - mask

# 	# show the masked region
# 	#cv2.imshow("Mask", mask)
# 	a = cv2.bitwise_and(superpixels_image, superpixels_image, mask = mask)
# 	#b = cv2.bitwise_and(img, img, mask=mask_complement)

# 	a = np.uint8(a)
# 	#b = np.uint8(b)
# 	#c = cv2.add(a, b)

# 	#c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)

# 	a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

# 	if i % 3 == 0:
# 		n_segments -= 3
# 		frames.append(Image.fromarray(a))


if n_segments > 0:
	frames.append(Image.fromarray(a))

frames[0].save('prova.gif',
               save_all=True, format='GIF', append_images=frames[1:], optimize=True, quality=20, duration=1, loop=0)

clip = mp.VideoFileClip("prova.gif")
clip.write_videofile("prova.mp4")


# cap = cv2.VideoCapture('./data/video.MOV')

# frame_width = int(cap.get(3)) 
# frame_height = int(cap.get(4)) 

# size = (frame_width, frame_height) 

# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter('prova.mp4', fourcc, 30.0, (frame_width,frame_height))

        

# while cap.isOpened():
#     ret, frame = cap.read()
#     frame = get_image_details(frame)
#     out.write(frame)
#     cv2.imshow('window-name', frame)
    
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cap.release()
# out.release() 
# cv2.destroyAllWindows() 