import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import tensorflow as tf

longitud, altura = 28, 28
modelo = './model/ModeloRed1.h5'
pesos_modelo = './model/ModeloPesos1.h5'
cnn = tf.keras.models.load_model(modelo)

cnn.load_weights(pesos_modelo)


def predecir(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    print('Respuesta: ')
    print(array[0])

    if answer == 0:
        print("El numero es: 0")
    elif answer == 1:
        print("El numero es: 1")
    elif answer == 2:
        print("El numero es: 2")
    elif answer == 3:
        print("El numero es: 3")
    elif answer == 4:
        print("El numero es: 4")
    elif answer == 5:
        print("El numero es: 5")
    elif answer == 6:
        print("El numero es: 6")
    elif answer == 7:
        print("El numero es: 7")
    elif answer == 8:
        print("El numero es: 8")
    elif answer == 9:
        print("El numero es: 9")

    return answer


predecir('./data/testSet/img_215.jpg')
