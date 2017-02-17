import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D,ELU
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np

def create_model(input_shape=(100,220,3)):
    #Setting Keras image dimension sequence to tensorflow
    K.set_image_dim_ordering('tf')
    model = Sequential()
    #Using Nvidia's model
    
    #normalizing input image
    model.add(Lambda(lambda x: x / 255 - 0.5,input_shape=input_shape))

    #1st layer
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())    
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    #2nd layer
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    #3rd layer
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    #4th layer
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    #5th layer
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    #Flatten layers
    model.add(Flatten())

    #Fully connected layers
    model.add(Dense(1024))
    model.add(ELU())

    #Fully connected layers    
    model.add(Dense(100))
    model.add(ELU())    

    #Fully connected layers    
    model.add(Dense(50))
    model.add(ELU())
    #Fully connected layer with single output
    model.add(Dense(1))
    
    return model