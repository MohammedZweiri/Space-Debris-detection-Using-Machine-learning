# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 08:19:39 2021

@author: Mohammed Zweiri (2021) and Ravindu Senaratne (2020)

"""


from keras.preprocessing.image import ImageDataGenerator
from skimage import io

Aug_data = ImageDataGenerator(rotation_range = 90, width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             horizontal_flip = True,
                             fill_mode = 'reflect')
import numpy as np
import os
from PIL import Image

image_directory = r'Your directory'
Size_of_the_image = 224
dataset = []

Images_to_be_done = os.listdir(image_directory)
for n, image_name in enumerate(Images_to_be_done):
    if (image_name.split('.')[1] == 'jpg'):
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((Size_of_the_image,Size_of_the_image))
        dataset.append(np.array(image))
k = np.array(dataset)

n = 0
for batch in Aug_data.flow(k, batch_size = 16,
                          save_to_dir = 'Make a folder, name it then link it here',
                          save_prefix = 'aug',
                          save_format = 'png'):
    n = n+1
    if n > 870:       #Increase n gets you more augmented images
        break

