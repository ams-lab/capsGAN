#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 02:42:02 2018

@author: tsugaike3
"""

# save MNIST images
# put 300 images into each directory 

import keras
from keras.datasets import mnist

import numpy as np
from PIL import Image, ImageOps
import os

def save_image(filename, data_array):
    im = Image.fromarray(data_array.astype('uint8'))
    im_invert = ImageOps.invert(im)
    im_invert.save(filename)

# Load MNIST Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

class_names = np.unique(y_train)
DIR_BASENAME = "MNIST/"
for class_name in class_names:
    dirname = DIR_BASENAME + str(class_name)
    if os.path.exists(dirname) == False:
        os.mkdir(dirname)

# Save Images
for class_name in class_names:
    i = 0
    org_indices = np.where(y_train == class_name)[0]
    np.random.shuffle(org_indices)
    # pop 100 indices
    indices = org_indices[:300]
    for index in indices:
        filename = "{0}/{1:03d}.jpg".format(DIR_BASENAME+str(class_name), i)
        save_image(filename, x_train[index])
        i += 1
