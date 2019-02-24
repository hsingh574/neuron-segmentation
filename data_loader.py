#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:08:56 2019

@author: harman
"""

'''
ADD MORE PREPROCESSING AS NECESSARY

'''


import numpy as np
import random

import keras
import cv2

#returns the samples and the labels in two numpy arrays

#combine part of the mouse dataset with the fruit fly dataset to aid in 
#generalizability of the model,


def getSamples(path1, path2, input_height, input_width, mouse_height, 
               mouse_width, merge = 160):
    data = np.load(path1)
    data2 = np.load(path2)
    volume = np.expand_dims(data['volume'], axis=-1)
    volume2 = np.expand_dims(data2['volume'], axis=-1)
    
    blankvol2 = np.zeros((merge,input_height,input_width,1))
    for ind, row in enumerate(volume2):
        if ind >= merge:
            continue
        else:
            x = np.reshape(row,(mouse_height,mouse_width))
    
            x = cv2.resize(x,(input_height,input_width))
            x = np.reshape(x,(input_height,input_width,1))
            blankvol2[ind] = x
    

    
    label = np.expand_dims(data['label'], axis=-1)
    
    label2 = np.expand_dims(data2['label'], axis=-1)
    blanklab2 = np.zeros((merge,input_height,input_width,1))
    for ind, row in enumerate(label2):
        if ind >= merge:
            continue
        else:
            x = np.reshape(row,(mouse_height,mouse_width))
    
            x = cv2.resize(x,(input_height,input_width))
            x = np.reshape(x,(input_height,input_width,1))
            blanklab2[ind] = x
    
    
    
    return np.vstack((volume,blankvol2)), np.vstack((label, blanklab2))

#generator for samples to be passed in fit_generator
def generateSplit(path1, path2, input_height, input_width,mouse_height, 
                  mouse_width, merge=160, split= 0.2, validate=1):
    Xtrain = []
    Ytrain = []
    Xval, Yval = [],[]
    
    vol, seg = getSamples(path1, path2,input_height, input_width, mouse_height, 
                          mouse_width, merge)
    if validate:
        for i in range(len(vol)):
            if random.random() < split:
                Xval.append(vol[i])
                Yval.append(seg[i])
            else:
                Xtrain.append(vol[i])
                Ytrain.append(seg[i])
                
    return np.array(Xtrain), np.array(Ytrain), np.array(Xval), np.array(Yval)
    
    
class MovieGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    
    
    'Initialization'
    def __init__(self, batch_size, number_samples, samples, annotations):
        self.batch_size = batch_size
        self.number_samples = number_samples
        self.samples = samples
        self.annotations = annotations
        self.shuffle = True

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.number_samples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        a,b = self.samples[index], self.annotations[index]
        a = np.reshape(a,(self.batch_size, self.samples.shape[1],self.samples.shape[2],1))
        b = np.reshape(b, (self.batch_size, self.annotations.shape[1],self.annotations.shape[2],1))
        return a,b
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.number_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)




