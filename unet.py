import numpy as np
import random
import os
 
from keras.models import save_model, load_model, Model
from keras.layers import Input, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose
import matplotlib.pyplot as plt

def build_model(input_size_1=256,input_size_2=256,output_size=2):
    inpt = Input(shape=(input_size_1, input_size_2, 3))

    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1,1), padding='same'):
        x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x
 
    def Conv2dT_BN(x, filters, kernel_size, strides=(2,2), padding='same'):
        x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x

    conv1 = Conv2d_BN(inpt, 8, (3, 3))
    conv1 = Conv2d_BN(conv1, 8, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(conv1)
 
    conv2 = Conv2d_BN(pool1, 16, (3, 3))
    conv2 = Conv2d_BN(conv2, 16, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(conv2)
 
    conv3 = Conv2d_BN(pool2, 32, (3, 3))
    conv3 = Conv2d_BN(conv3, 32, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(conv3)
 
    conv4 = Conv2d_BN(pool3, 64, (3, 3))
    conv4 = Conv2d_BN(conv4, 64, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(conv4)
 
    conv5 = Conv2d_BN(pool4, 128, (3, 3))
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2d_BN(conv5, 128, (3, 3))
    conv5 = Dropout(0.5)(conv5)
 
    convt1 = Conv2dT_BN(conv5, 64, (3, 3))
    concat1 = concatenate([conv4, convt1], axis=3)
    concat1 = Dropout(0.5)(concat1)
    conv6 = Conv2d_BN(concat1, 64, (3, 3))
    conv6 = Conv2d_BN(conv6, 64, (3, 3))
 
    convt2 = Conv2dT_BN(conv6, 32, (3, 3))
    concat2 = concatenate([conv3, convt2], axis=3)
    concat2 = Dropout(0.5)(concat2)
    conv7 = Conv2d_BN(concat2, 32, (3, 3))
    conv7 = Conv2d_BN(conv7, 32, (3, 3))
 
    convt3 = Conv2dT_BN(conv7, 16, (3, 3))
    concat3 = concatenate([conv2, convt3], axis=3)
    concat3 = Dropout(0.5)(concat3)
    conv8 = Conv2d_BN(concat3, 16, (3, 3))
    conv8 = Conv2d_BN(conv8, 16, (3, 3))
 
    convt4 = Conv2dT_BN(conv8, 8, (3, 3))
    concat4 = concatenate([conv1, convt4], axis=3)
    concat4 = Dropout(0.5)(concat4)
    conv9 = Conv2d_BN(concat4, 8, (3, 3))
    conv9 = Conv2d_BN(conv9, 8, (3, 3))
    conv9 = Dropout(0.5)(conv9)
    outpt = Conv2D(filters=output_size, kernel_size=(1,1), strides=(1,1), padding='same', activation="sigmoid")(conv9)
 
    model = Model(inpt, outpt)
    return model
#model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['mse'])

#model.summary()
