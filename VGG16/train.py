"""Clasificador con VGG16 DIABETIC RETINOPLATY"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
from keras.preprocessing.image import ImageDataGenerator
import keras as K
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Input
import matplotlib as plt
from matplotlib.figure import Figure
import pandas as pandas
from keras.applications import vgg16
import numpy as np
import tensorflow as tf
from keras.models import Model

#Corregir el fallo de la CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["PATH"].append("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin/cudart64_100.dll")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

trainLabels = pandas.read_csv("./trainLabels_cropped.csv", dtype=str)

#Hay que añadir la extension a la lista de imagenes


def append_ext(fn):
    return fn+".jpeg"


trainLabels["image"]=trainLabels["image"].apply(append_ext)
#test_data["id_code"]=test_data["id_code"].apply(append_ext)
#Mejor resultado: 0.7490
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.5,
        zoom_range=0.2,
        samplewise_std_normalization=False,
        featurewise_std_normalization=False,
        #zca_epsilon=1e-6,
        #zca_whitening=True,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=45,
        channel_shift_range=0.10,
        validation_split=0.30)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=trainLabels,
        directory='resized_train_cropped/resized_train_cropped/',
        x_col="image",
        y_col="level",
        target_size=(224, 224),
        batch_size=5,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_dataframe(
        dataframe=trainLabels,
        directory='resized_train_cropped/resized_train_cropped/',
        x_col="image",
        y_col="level",
        target_size=(224, 224),
        batch_size=5,
        class_mode='categorical',
        subset='validation')
"""
vgg16 = vgg16.VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(224,224,3), pooling=None, classes=5)


optimizer=keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
vgg16.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['acc', 'mse'])

vgg16.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=100)
"""

def VGG16_Without_lastPool(include_top=False, input_tensor='imagenet', input_shape=(224,224,3), pooling=None, classes=5):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
        # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)  # to 16x16

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)  # to 8x8

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)  # to 4x4

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)  # to 2x2

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.2)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.2)(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Create model.
    model = Model(img_input, x, name='vgg16Bis')
    return model

def create_vgg16WithoutPool():
  model = VGG16_Without_lastPool(include_top=True, input_tensor=None, input_shape=(224,224,3), pooling=None, classes=5)

  return model

vgg16Bis_model = create_vgg16WithoutPool()
vgg16Bis_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc', 'mse'])

vgg16Bis_model.fit_generator(
    train_generator,
    steps_per_epoch=500,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=100)

