#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:31:17 2019

@author: harman
"""
from keras.models import Sequential
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers import Conv2D, LeakyReLU,Concatenate
from keras.optimizers import Adam , SGD
from keras import backend as K
from keras.models import Model
from keras.layers import Input
import os
#X is a tuple of the shape 
def unet():
    inp = Input((1248,1248,1))
    x = Conv2D(32, (1,1), padding='same', kernel_initializer="glorot_normal")(inp) 
    x = BatchNormalization(axis = 3)(x)
    x = LeakyReLU()(x) 
    
    b_0 = x
    
    x = MaxPooling2D((2, 2), strides=2)(x)  
    b_1 = (x)
    
    x = Conv2D(64, (1,1), padding='same', kernel_initializer="glorot_normal")(x) 
    x = BatchNormalization(axis = 3)(x)
    x = LeakyReLU()(x)
    
    x = MaxPooling2D((2, 2), strides=2)(x) 
    b_2 = (x) 
    
    x = Conv2D(128, (1,1), padding='same', kernel_initializer="glorot_normal")(x) 
    x = BatchNormalization(axis = 3)(x)
    x = LeakyReLU()(x)
    
    x = MaxPooling2D((2, 2), strides=2)(x) 
    b_3 =(x) 
    
    x = Conv2D(256, (1,1), padding='same', kernel_initializer="glorot_normal")(x) 
    x = BatchNormalization(axis = 3)(x)
    x = LeakyReLU()(x)
    
    encoded = MaxPooling2D((2, 2))(x) 
    
    x = Conv2D(256, (1,1), padding='same', kernel_initializer="glorot_normal")(encoded) 
    x = BatchNormalization(axis = 3)(x)
    x = LeakyReLU()(x)
    
    x = UpSampling2D((2, 2))(x) 
    x = Concatenate(axis=3)([x, b_3])
    
    x = Conv2D(128, (1,1), padding='same', kernel_initializer="glorot_normal")(x) 
    x = BatchNormalization(axis = 3)(x)
    x = LeakyReLU()(x)
    
    x = UpSampling2D((2, 2))(x) 
    x = Concatenate(axis=3)([x, b_2])
    
    x = Conv2D(64, (1,1), padding='same', kernel_initializer="glorot_normal")(x) 
    x = BatchNormalization(axis = 3)(x)
    x = LeakyReLU()(x)
    
    x = UpSampling2D((2, 2))(x) 
    x = Concatenate(axis=3)([x, b_1])
    
    x = Conv2D(32, (1,1), padding='same', kernel_initializer="glorot_normal")(x) 
    x = BatchNormalization(axis = 3)(x)
    x = LeakyReLU()(x)
    
    x = UpSampling2D((2, 2))(x) 
    x = Concatenate(axis=3)([x, b_0])
    
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', kernel_initializer='glorot_normal')(x)
    
    model = Model(inp, decoded)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model



def segnet(nClasses , optimizer=None , input_height=1248, input_width=1248 ):

    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    model = Sequential()
    model.add(Layer(input_shape=(input_height , input_width,1 )))

    # encoder
    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Convolution2D(128, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Convolution2D(256, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Convolution2D(512, kernel, kernel, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    # decoder
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Convolution2D(512, kernel, kernel, border_mode='valid'))
    model.add( BatchNormalization())

    model.add( UpSampling2D(size=(pool_size,pool_size)))
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Convolution2D(256, kernel, kernel, border_mode='valid'))
    model.add( BatchNormalization())

    model.add( UpSampling2D(size=(pool_size,pool_size)))
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Convolution2D(128, kernel, kernel, border_mode='valid'))
    model.add( BatchNormalization())

    model.add( UpSampling2D(size=(pool_size,pool_size)))
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
    model.add( BatchNormalization())


    model.add(Convolution2D( nClasses , 1, 1, border_mode='valid',))
    print(model.output_shape)
    model.outputHeight = model.output_shape[1]
    model.outputWidth = model.output_shape[2]
    print(model.outputHeight)
    print(model.outputWidth)
        
    model.add(Reshape(( nClasses ,  model.output_shape[1]*model.output_shape[2]   ), input_shape=( nClasses , model.output_shape[1], model.output_shape[2]  )))
    
    model.add(Permute((2, 1)))

    model.add(Activation('softmax'))
        
    if not optimizer is None:
        model.compile(loss="categorical_crossentropy", optimizer= optimizer , metrics=['accuracy'] )
    model.summary()
    return model


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#segnet(2)
