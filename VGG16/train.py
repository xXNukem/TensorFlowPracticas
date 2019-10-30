"""Clasificador con VGG16 DIABETIC RETINOPLATY"""
from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pandas
from keras.applications import vgg16
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sn
#Corregir el fallo de la CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["PATH"].append("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin/cudart64_100.dll")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

trainLabels = pandas.read_csv("./trainLabels.csv", dtype=str)

#Hay que añadir la extension a la lista de imagenes


def append_ext(fn):
    return fn+".jpeg"


trainLabels["image"]=trainLabels["image"].apply(append_ext)
#test_data["id_code"]=test_data["id_code"].apply(append_ext)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=trainLabels,
        directory='resized_train/resized_train/',
        x_col="image",
        y_col="level",
        target_size=(224, 224),
        batch_size=5,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_dataframe(
        dataframe=trainLabels,
        directory='resized_train/resized_train/',
        x_col="image",
        y_col="level",
        target_size=(224, 224),
        batch_size=5,
        class_mode='categorical',
        subset='validation')

vgg16 = vgg16.VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(224,224,3), pooling=None, classes=5)
vgg16.compile(loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy'])
vgg16.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=300)

