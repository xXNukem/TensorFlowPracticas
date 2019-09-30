"""Practicando clasificadores"""
import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session() #Eliminamos posibles sesiones de Keras abiertas

data_train = './data/trainingSet'
data_test = './data/trainingSet'

"""Parametros de configuracion"""
nEpochs = 20  # Iteraciones sobre el set de datos
width, height = 28, 28  # Dimensiones de la imagen
batch_size = 32  # Imagenes a procesar en cada paso (32 para ahorrar rercursos)
steps_per_epochs = 1000  # Iteraciones por epoca
validation_steps = 300  # Tras cada epoca se comprueba 300 pasos con el set de test
convolution_filter_1 = 32  # Filtros de cada convolucion
convolution_filter_2 = 64
filter_size_1 = (3, 3)  # Tamaño de los filtros
filter_size_2 = (2, 2)
pool_size = (2, 2)  # Tamaño del filtro de max pooling
nClasses = 10  # Clases, 10 clases para 10 numeros
lr = 0.0010  # Learning rate, mide los ajustes para acercarse a la prediccion
# Tamaños de filtros y demas sacados de las recomendaciones en la documentacion y otros tutoriales

# Preprocesamiento de imagenes

train_DataGen = ImageDataGenerator(
    rescale=1. / 255,  # Reescalamos los valores de pixels de 1 a 255 (en este caso las img estan en blanco y negro)
    shear_range=0.2,  # Inclinamos las imagenes un poco para facilitar el aprendizaje
    zoom_range=0.2,  # HAcemos zoom para que el algoritmo aprenda de diferentes tamaños
    horizontal_flip=True)  # Movemos la direccion de las imagenes para que el algoritmo aprenda sobre ello

test_DataGen = ImageDataGenerator(
    rescale=1. / 255)  # Las imagenes de test no se tocan porque se quiere que esten como son realmenten

"""Generamos las imagenes con los parametros establecidos"""
train_img_generator = train_DataGen.flow_from_directory(
    data_train,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical')

test_img_generator = test_DataGen.flow_from_directory(
    data_test,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical')

# Creacion de la red neuronal
cnn = Sequential()  # La red es secuencial
cnn.add(Convolution2D(convolution_filter_1, filter_size_1, padding="same", input_shape=(width, height, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=pool_size))

cnn.add(Convolution2D(convolution_filter_2, filter_size_2, padding="same"))
cnn.add(MaxPooling2D(pool_size=pool_size))
"""Comenzamos a meter la red que clasifica"""
cnn.add(Flatten()) # Aplanamos la imagen
cnn.add(Dense(256, activation='relu')) # Mandamos la informacion aplanada a una capa normal de toda la vida
cnn.add(Dropout(0.5)) # Evitamos el sobreentrenamiento desactivando la mitad de las neuronas cada paso
cnn.add(Dense(nClasses, activation='softmax')) #Ultima capa softmax con tantas salidas somo clases

cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr), #Parametros de optimizacion del algoritmo
            metrics=['accuracy'])

cnn.fit_generator( # Funcion de entrenamiento del algoritmo, le pasamos los parametros establecidos
    train_img_generator,
    steps_per_epoch=steps_per_epochs,
    epochs=nEpochs,
    validation_data=test_img_generator,
    validation_steps=validation_steps)

# Guardamos los modelos
target_dir = './model/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
cnn.save('./model/ModeloRed1.h5')
cnn.save_weights('./model/ModeloPesos1.h5')
