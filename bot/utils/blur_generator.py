import cv2
import os
import numpy as np
from random import randint

class BlurGenerator():

    def blur(self, img):
        method = randint(0,1)
        if method:
            size = randint(15, 19) 
            img = cv2.blur(img,(size,size))
        else:
            sigmaX = randint(3,7) 
            img = cv2.GaussianBlur(img, (45,45), sigmaX=sigmaX)
        return img
    
    def motion(self, img):
        size = randint(15, 33) 
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size

        img = cv2.filter2D(img, -1, kernel_motion_blur)
        return img


if __name__ == "__main__":
    files = os.listdir('./data/train/')
    generator = BlurGenerator()

    already_processed = []
    for i in range(0, 2000):
        print(f'\r{i+1}/{2000}', end='')
        random = randint(0, len(files))
        f_name = files[random]

        while f_name in already_processed:
            random = randint(0, len(files))
            f_name = files[random]
        
        already_processed.append(f_name)

        img = cv2.imread('./data/train/' + f_name)

        if i > 1000:  
            blurred = generator.blur(img)
        else:
            blurred = generator.motion(img)
        
        cv2.imwrite('./data/not_blurred/' + f_name, img)
        cv2.imwrite('./data/blurred/blurred_' + f_name, blurred)
        


