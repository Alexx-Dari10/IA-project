import os

import tensorflow as tf
import matplotlib.pyplot as plt

generador_entrenamiento = tf.keras.preprocessing.image.ImageDataGenerator()
datos_entrenamiento = generador_entrenamiento.flow_from_directory(os.getcwd() + "/dataset/Animals")

generador_test = tf.keras.preprocessing.image.ImageDataGenerator()
datos_test = generador_test.flow_from_directory(os.getcwd() + "/dataset/test", class_mode=None)

lote_test = next(datos_test)

from matplotlib import pyplot as plt
plt.imshow(lote_test[0]/255.)
plt.axis('off')
plt.show()
plt.imshow(lote_test[1]/255.)
plt.axis('off')
plt.show()

inception = tf.keras.applications.InceptionV3(include_top=False, input_shape=(256, 256, 3))

predictor = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(128, activation="relu"), 
    tf.keras.layers.Dense(5, activation="softmax")
])
modelo = tf.keras.models.Sequential([inception, predictor])
modelo.compile(optimizer="adam", loss="categorical_crossentropy")

modelo.fit(datos_entrenamiento, epochs=20)



probs = modelo.predict(lote_test)

import numpy as np
clase = np.argmax(probs, -1)

mostrar_imagenes = 15

for i in range(mostrar_imagenes):
    plt.imshow(lote_test[i]/255.)
    plt.axis('off')
    plt.show()

    if clase[i] == 0: print('Cheeta')
    elif clase[i] == 1: print('Jaguar')
    elif clase[i] == 2: print('Leopard')
    elif clase[i] == 3: print('Lion')
    else: print('Tiger')

modelo.save("animals_model.h5")