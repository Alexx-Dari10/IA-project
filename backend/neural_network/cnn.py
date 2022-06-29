import tensorflow as tf


class cnn:
    def __init__(self):
        pass

    def run(self, data):
        inception = tf.keras.applications.InceptionV3(include_top=False, input_shape=(256, 256, 3))
        predictor = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(128, activation="relu"), 
            tf.keras.layers.Dense(5, activation="softmax")
        ])

        model = tf.keras.models.Sequential([inception, predictor])
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        model.fit(data, epochs=50)

        model.save("animals_model.h5")


        # probs = modelo.predict(lote_test)

        # import numpy as np
        # clase = np.argmax(probs, -1)
        # mostrar_imagenes = 15

        # for i in range(mostrar_imagenes):
        #     plt.imshow(lote_test[i]/255.)
        #     plt.axis('off')
        #     plt.show()

        #     if clase[i] == 0: print('Cheeta')
        #     elif clase[i] == 1: print('Jaguar')
        #     elif clase[i] == 2: print('Leopard')
        #     elif clase[i] == 3: print('Lion')
        #     else: print('Tiger')

        # modelo.save("animals_model.h5")
        