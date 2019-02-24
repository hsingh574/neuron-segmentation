#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:03:13 2019

@author: harman
"""

import argparse
import data_loader
import models
import os
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint



#def focal_loss(gamma=2., alpha=.25):
#	def focal_loss_fixed(y_true, y_pred):
#		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#        	pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#        	return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
#	return focal_loss_fixed



parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--train_images", type = str  )
parser.add_argument("--train_images2", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 1248  )
parser.add_argument("--input_width", type=int , default = 1248 )
parser.add_argument("--mouse_height", type=int , default = 384 )
parser.add_argument("--mouse_width", type=int , default = 384  )

parser.add_argument("--epochs", type = int, default = 5 )
parser.add_argument("--batch_size", type = int, default = 1 )
parser.add_argument("--load_weights", type = str , default = "")

parser.add_argument("--optimizer_name", type = str , default = "adadelta")
parser.add_argument("--checkpoints", type = str , default = "/home/hsuri/Datathon/weights/seg_")


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_images_path1 = args.train_images
train_images_path2 = args.train_images2
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
mouse_height = args.mouse_height
mouse_width = args.mouse_width
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights
checkpoint = args.checkpoints

optimizer_name = args.optimizer_name

m = models.NeuronSegNet(input_height, input_width)

if len( load_weights ) > 0:
    m.load_weights(load_weights)
Xtrain, Ytrain, Xval, Yval  = data_loader.generateSplit(train_images_path1,
              train_images_path2,input_height, input_width, 
              mouse_width,merge=160, split= 0.2, validate=1)

tg = data_loader.MovieGenerator( train_batch_size, len(Xtrain), Xtrain,Ytrain)
vg = data_loader.MovieGenerator( train_batch_size, len(Xval), Xval, Yval)
        

filepath = checkpoint + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
#restore_ckpt_callback = RestoreCkptCallback(pretrian_model_path='./XXXX.ckpt') 
callbacks_list = [checkpoint]
m.fit_generator(generator=tg, validation_data=vg,steps_per_epoch=2500,
                epochs=epochs,validation_steps=100,callbacks = callbacks_list)

m.save_weights(save_weights_path)


