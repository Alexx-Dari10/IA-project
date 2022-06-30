import os
import pickle
import numpy

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.models import load_model

class object_detection:
    def __init__(self):
        path = os.getcwd() + '/datasets/Objects/'

        with open(path + "XTrain.pickle", "rb") as f:
            self.X_train = pickle.load(f)

        with open(path + "yTrain.pickle", "rb") as f:
            self.y_train = pickle.load(f)

        with open(path + "Xtest.pickle", "rb") as f:
            self.X_test = pickle.load(f)

        with open(path + "yTest.pickle", "rb") as f:
            self.y_test = pickle.load(f)


    def run(self):
        X_train = self.X_train.astype('float32')
        X_test = self.X_test.astype('float32')

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        y_train = np_utils.to_categorical(self.y_train)
        y_test = np_utils.to_categorical(self.y_test)
        
        if not os.path.exists(os.getcwd() + '/models/object_model.h5'):
            seed = 21
                
            class_num = y_test.shape[1]

            model = Sequential()

            model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
            model.add(Activation('relu'))

            model.add(Dropout(0.2))
            model.add(BatchNormalization())

            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())

            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
                
            model.add(Conv2D(128, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())

            model.add(Flatten())
            model.add(Dropout(0.2))

            model.add(Dense(256, kernel_constraint=maxnorm(3)))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
                
            model.add(Dense(128, kernel_constraint=maxnorm(3)))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())

            model.add(Dense(class_num))
            model.add(Activation('softmax'))

            epochs = 25
            optimizer="adam"

            model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
            print(model.summary())

            numpy.random.seed(seed)
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

            model.save(os.getcwd() + '/models/object_model.h5')

        model = load_model(os.getcwd() + '/models/object_model.h5')

        # Model evaluation
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))