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



#def focal_loss(gamma=2., alpha=.25):
#	def focal_loss_fixed(y_true, y_pred):
#		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#        	pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#        	return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
#	return focal_loss_fixed



parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--train_images", type = str  )
parser.add_argument("--train_annotations", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 1248  )
parser.add_argument("--input_width", type=int , default = 1248 )

parser.add_argument('--validate',action='store_false')
parser.add_argument("--val_images", type = str , default = "")
parser.add_argument("--val_annotations", type = str , default = "")

parser.add_argument("--epochs", type = int, default = 5 )
parser.add_argument("--batch_size", type = int, default = 1 )
parser.add_argument("--val_batch_size", type = int, default = 1 )
parser.add_argument("--load_weights", type = str , default = "")

parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name
validate = True


#modelFN = models.segnet

#m = modelFN( n_classes , input_height=input_height, input_width=input_width)
#m.compile(loss='binary_crossentropy',
#      optimizer= optimizer_name ,
#      metrics=['accuracy'])
m = models.unet()


if len( load_weights ) > 0:
    m.load_weights(load_weights)


#print("Model output shape" ,  m.output_shape)

#output_height = m.outputHeight
#output_width = m.outputWidth
temp1 = '/home/hsuri/Datathon/fruit_fly_volumes.npz'
temp2 = '/home/hsuri/Datathon/mouse_volumes.npz'

Xtrain, Ytrain, Xval, Yval  = data_loader.generateSplit(temp1,temp2, 
                                                        split= 0.2, validate=1)
#for  index,row in enumerate(Ytrain):
#    newYtrain = np.zeros((Ytrain.shape[0],input_width*input_height,n_classes))

#    y = np.reshape(row, (input_width*input_height, n_classes))
#    newYtrain[index] = y
    
#for  index, row  in enumerate(Yval):
#    newYval = np.zeros((Yval.shape[0],input_width*input_height,n_classes))
#    y = np.reshape(row, (input_width*input_height, n_classes))
#    newYval[index] = y

                
        

tg = data_loader.MovieGenerator( train_batch_size, len(Xtrain), Xtrain,Ytrain)
vg = data_loader.MovieGenerator( train_batch_size, len(Xval), Xval, Yval)
        
        


#Xtrain, Ytrain, Xval, Yval = data_loader.generateSplit(train_images_path, 0.2, 
#                                                       validate)
#G = data_loader.generateTrainingImages(Xtrain, Ytrain,train_batch_size, 
#                                       n_classes, input_height,input_width)

#Gval = data_loader.generateValidationImages(Xval, Yval)



m.fit_generator(generator=tg, validation_data=vg,steps_per_epoch=2500,epochs=1,
                validation_steps=100)

m.save_weights('/home/hsuri/Datathon/first_weights.h5' )
print('yeet')
#m.save( save_weights_path + ".model." + str( 1 ) )

