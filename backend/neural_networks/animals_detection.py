import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import Dropout, BatchNormalization
from PIL import Image, ImageOps
from keras.models import load_model


class animals_detection:
    def __init__(self):
        pass

    def run(self, img, img_test):
        data = img.tf_process()
        val_data = img_test.tf_process(True)

        class_num = len(data.class_indices)
        types = list(data.class_indices.keys())

        if not os.path.exists(os.getcwd() + '/models/animal_model.h5'):
            inception = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(75, 75, 3))
            predictor = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(), 
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(class_num, activation="softmax")
            ])

            model = tf.keras.models.Sequential([inception, predictor])

            model.add(Dropout(0.2))
            model.add(BatchNormalization())

            model.compile(optimizer="adam", loss="categorical_crossentropy")

            model.fit(data, epochs=50, validation_data=val_data)
            model.save(os.getcwd() + '/models/animal_model.h5')
        
        if val_data:
            model = load_model(os.getcwd() + '/models/animal_model.h5')

            predicts = []
           
            for path in val_data.filepaths:
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                image = Image.open(path)
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.ANTIALIAS)

                image_array = np.asarray(image)
                image_array.resize((224,224,3), refcheck=False)
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array

                prediction = model.predict(data)
                print(prediction)
                predicts.append(np.argmax(prediction, -1)[0])

            _asserts = 0
            for i in range(len(predicts)):
                if predicts[i] == val_data.classes[i]:
                    _asserts += 1
            
            print(f'Accuracy: {(_asserts*100)/len(predicts)} %')