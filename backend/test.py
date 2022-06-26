import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split 
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model,Sequential
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras.layers import Dropout, BatchNormalization
from tensorflow.python.keras.layers import (
    Input, Dense, Conv2D, Flatten, Activation, 
    MaxPooling2D, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D, add
)

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.utils import plot_model

from keras.applications.vgg19 import VGG19
#from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3


from PIL import Image



filelist = []

for dirname, _, filenames in os.walk('./dataset/Animals/'):
    for filename in filenames:
        filelist.append(os.path.join(dirname, filename))

print(filelist)