import cv2
import numpy as np
from skimage import io 
from skimage.transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt
import random
from skimage import img_as_ubyte
import os
from skimage.util import random_noise
from PIL import Image, ImageEnhance

def rotation(image):
        angle= random.randint(0,360)
        return rotate(image, angle)

def horizontal_flip(image):
    return  np.fliplr(image)

def vertical_flip(image):
    return np.flipud(image)

def add_noise(image):
    return random_noise(image)

def blur_image(image):
    return cv2.GaussianBlur(image, (9,9),0)


transformations = {
                      'rotate': rotation,
                      'horizontal flip': horizontal_flip, 
                      'vertical flip': vertical_flip,
                      'adding noise': add_noise,
                      'blur': blur_image,
                 }
class DataAugmentation():      
    
    #images è un array contente il percorso all'immagine
    def generate_images(self, images, n, augmented_path):
        
        for i in range(n):
            print(str(i)+"/"+str(n))
            image=random.choice(images)
            original_image = io.imread(image)
            augmented_images= original_image
            
            count = random.randint(2, len(transformations)) #choose random number of transformation to apply on the image
            print(count)
            t=[]
            j=0
            while j < count:
                key = random.choice(list(transformations)) #randomly choosing method 
                if key not in t:
                    print(key)
                    augmented_images = transformations[key](augmented_images)
                    j +=1
            
                t.append(key)
            new_image_path = "%s/augmented_%s.jpg" %(augmented_path, i)
            transformed_image = img_as_ubyte(augmented_images)  #Convert an image to unsigned byte format, with values in [0, 255].
            img_float32 = np.float32(transformed_image)
            augmented_images = cv2.cvtColor(img_float32, cv2.COLOR_BGR2RGB) #convert image to RGB 
            cv2.imwrite(new_image_path, augmented_images) # save transformed image to path
            
        return True
