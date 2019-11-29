"""Clasificador con VGG16 DIABETIC RETINOPLATY"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorboard
from keras.preprocessing.image import ImageDataGenerator
import keras as K
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
#Mejor resultado: 0.7490
train_datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_std_normalization=False,
        samplewise_std_normalization=True,
        validation_split=0.30)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=trainLabels,
        directory='resized_train_cropped/resized_train_cropped/',
        x_col="image",
        y_col="level",
        target_size=(224, 224),
        batch_size=10,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_dataframe(
        dataframe=trainLabels,
        directory='resized_train_cropped/resized_train_cropped/',
        x_col="image",
        y_col="level",
        target_size=(224, 224),
        batch_size=10,
        class_mode='categorical',
        subset='validation')

vgg16 = vgg16.VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(224,224,3), pooling=None, classes=5)

vgg16.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['acc', 'mse'])


log_dir="logs\\fit\\" +'Condicion de parada (restaura pesos y paciencia)'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

parada=callbacks.callbacks.EarlyStopping(monitor='acc',mode='max',verbose=1,restore_best_weights=True,patience=3)

vgg16.fit_generator(
    train_generator,
    steps_per_epoch=500,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=100,
    callbacks=[tensorboard_callback,parada])






