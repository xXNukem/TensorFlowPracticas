"""Clasificador con VGG16 DIABETIC RETINOPLATY"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorboard
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Input
import pandas as pandas
from keras.applications import vgg16
import tensorflow as tf
from keras.models import Model
import datetime
from keras.callbacks import TensorBoard
import pkg_resources
from keras import callbacks
from PIL import Image
import numpy as np
from pathlib import Path


for entry_point in pkg_resources.iter_entry_points('tensorboard_plugins'):
    print(entry_point.dist)
#-----------------------------------------------------------
#Corregir el fallo de la CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["PATH"].append("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin/cudart64_100.dll")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#------------------------------------------------------------
trainLabels = pandas.read_csv("./trainLabels_cropped.csv", dtype=str)

#Hay que añadir la extension a la lista de imagenes
def append_ext(fn):
    return fn+".jpeg"

trainLabels["image"]=trainLabels["image"].apply(append_ext)
#test_data["id_code"]=test_data["id_code"].apply(append_ext)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    featurewise_center=True,
    samplewise_center=True,
    #samplewise_std_normalization=True,
    #featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True,
    #rotation_range=15,  # randomly rotate pictures
    #width_shift_range=0.1,  # randomly translate pictures
    #height_shift_range=0.1,
    #shear_range=0.2,  # randomly apply shearing
    #zoom_range=0.2,  # random zoom range
    #horizontal_flip=True,
    validation_split=0.30)
#Lectura de los estadísticos
"""
statistics=[]
with open('statistics.txt', 'r') as reader:
    for line in reader:
        statistics.append(line)

train_datagen.mean=statistics[0]
train_datagen.std=statistics[1]
#---------------------
"""

train_datagen.mean=[[['106.23706131  74.19983673  53.1510033']]]
train_datagen.std=[[['59.24701176 41.45938749 29.7537066']]]

train_generator = train_datagen.flow_from_dataframe(
        dataframe=trainLabels,
        directory='resized_train_cropped/resized_train_cropped/',
        x_col="image",
        y_col="level",
        target_size=(224, 224),
        batch_size=10,
        class_mode='categorical',
        color_mode='rgb', #quitar o no quitar
        subset='training')

validation_generator = train_datagen.flow_from_dataframe(
        dataframe=trainLabels,
        directory='resized_train_cropped/resized_train_cropped/',
        x_col="image",
        y_col="level",
        target_size=(224, 224),
        batch_size=10,
        class_mode='categorical',
        color_mode='rgb',
        subset='validation')


model=Sequential()
model.add(vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None))
model.add(Flatten())
model.add(Dense(4096, activation='relu', name='fc1'))
model.add(Dropout(0.1))
model.add(Dense(2048, activation='relu', name='fc2'))
model.add(Dropout(0.1))
model.add(Dense(1024, activation='relu', name='fc3'))
model.add(Dropout(0.1))
model.add(Dense(512, activation='relu', name='fc4'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu', name='fc5'))
model.add(Dense(5, activation='softmax', name='predictions'))

model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['acc', 'mse'])

#Model Summary
model.summary()

log_dir="logs\\fit\\" +'Prueba'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

parada=callbacks.callbacks.EarlyStopping(monitor='acc',mode='max',verbose=1,restore_best_weights=True,patience=2)
learningRate=callbacks.callbacks.ReduceLROnPlateau(monitor='acc', factor=0.1, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0,patience=2)
#checkpoint=keras.callbacks.callbacks.ModelCheckpoint('\\pesos\\weights', monitor='acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

for i in range(5):

    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=30,
        validation_freq=1,
        callbacks=[parada])

