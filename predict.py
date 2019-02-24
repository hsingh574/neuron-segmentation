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


'''
Note that you have to resize the image if it is not (1248,1248,1). Alternatively, 
you can remove the input layer of the model and load the convolutional weights 
and then your own pixelwise classifier on top of them
'''

#helper to resize the mouse images
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

#test image is path to test image
    
parser = argparse.ArgumentParser()
parser.add_argument("--test_image", type = str  )
parser.add_argument("--input_height", type=int , default = 1248  )
parser.add_argument("--input_width", type=int , default = 1248 )
parser.add_argument("--load_weights", type = str , default = "")
parser.add_argument("--output_file", type = str , default = "")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
test_image = args.test_image
input_height = args.input_height
input_width = args.input_width
weights = args.load_weights
output_file = args.output_file


m = models.NeuronSegNet(input_height, input_width)

m.load_weights('/home/hsuri/Datathon/first_weights.h5')
x = cv2.imread(test_image)
assert(test_image.shape == (1248,1248,1))
pr = m.predict( np.array([x]))[0]
cv2.imwrite(output_file,pr)


    


