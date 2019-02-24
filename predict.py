#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:52:53 2019

@author: harman
"""

import argparse
import models
import data_loader
from keras.models import load_model
import glob
import cv2
import numpy as np
import random
import os
from scipy.ndimage import zoom

# =============================================================================
# parser = argparse.ArgumentParser()
# parser.add_argument("--save_weights_path", type = str  )
# parser.add_argument("--epoch_number", type = int, default = 5 )
# parser.add_argument("--test_images", type = str , default = "")
# parser.add_argument("--output_path", type = str , default = "")
# parser.add_argument("--input_height", type=int , default = 224  )
# parser.add_argument("--input_width", type=int , default = 224 )
# parser.add_argument("--n_classes", type=int )
# 
# args = parser.parse_args()
# 
# n_classes = args.n_classes
# images_path = args.test_images
# input_width =  args.input_width
# input_height = args.input_height
# epoch_number = args.epoch_number
# =============================================================================
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
m = models.unet()
data= np.load('/home/hsuri/Datathon/fruit_fly_volumes.npz')


volume = np.expand_dims(data['volume'], axis=-1)
    
label = np.expand_dims(data['label'], axis=-1)
    

# =============================================================================
# m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
# m.load_weights(  args.save_weights_path + "." + str(  epoch_number )  )
# m.compile(loss='categorical_crossentropy',
#       optimizer= 'adadelta' ,
#       metrics=['accuracy'])
# =============================================================================

m.load_weights('/home/hsuri/Datathon/first_weights.h5')

print(volume[1].shape)
#x = np.reshape(volume[1],(384,384))

#x = cv2.resize(x,(1248,1248))
#x = np.reshape(x,(1248,1248,1))
x = volume[0]
pr = m.predict( np.array([x]))[0]
cv2.imwrite('/home/hsuri/Datathon/test3.png',pr)

#y = np.reshape(volume[2],(384,384))

#y = cv2.resize(y,(1248,1248))
#y = np.reshape(y,(1248,1248,1))
y = volume[2]
pr = m.predict( np.array([y]))[0]
cv2.imwrite('/home/hsuri/Datathon/test4.png',pr)

    


