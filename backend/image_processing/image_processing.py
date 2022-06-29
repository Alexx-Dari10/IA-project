import os
import re
import cv2
import imutils

import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class img_processing:

    def __init__(self, path, cv = True):
        self.path = path

        if cv:
            self.images      = []
            self.directories = []
            self.dircount    = []
            self.filePaths   = []
            self.rawImages   = []
            self.features    = []
            self.labels      = []
            self.types       = []

            self.cv_process(path)
    

    def image_to_feature_vector(self, image):
        scale_percent = 40
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)

        size = (50, 50)
        return cv2.resize(image, size).flatten()


    def extract_color_histogram(self, image, bins=(8, 8, 8)):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        if imutils.is_cv2():
            hist = cv2.normalize(hist)
        else:
            cv2.normalize(hist, hist)

        return hist.flatten()


    def cv_process(self, imgpath):
        prevRoot = ''
        cant = 0
        
        print("Reading images of ",imgpath)
        
        for root, _, filenames in os.walk(imgpath):
            for filename in filenames:
                if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                    cant += 1
                    filepath = os.path.join(root, filename)
                    image = plt.imread(filepath)

                    pixels = self.image_to_feature_vector(image)
                    hist = self.extract_color_histogram(image)
                    
                    self.rawImages.append(pixels)
                    self.features.append(hist)

                    self.filePaths.append(filepath)
                    self.images.append(image)

                    b = "Processing..." + str(cant)
                    print (b, end="\r")

            if prevRoot != root and root != imgpath:
                print(root, cant)
                prevRoot=root
                self.directories.append(root)
                self.dircount.append(cant)
                cant = 0
        
        print('Analysed directories:',len(self.directories))
        print("Images by directory", self.dircount)
        print('total imgs:',sum(self.dircount))



        indice = 0
        for cantidad in self.dircount:
            for _ in range(cantidad):
                self.labels.append(indice)
            indice = indice + 1 
        print("Total of labels created: ",len(self.labels))

        indice = 0
        for directorio in self.directories:
            name = directorio.split(os.sep)
            print(indice , name[len(name)-1])
            self.types.append(name[len(name)-1])
            indice = indice + 1
    

    def reduction(self, data):
        scaler = StandardScaler()
        r_data = scaler.fit_transform(data)

        pca = PCA(n_components=2)
        pca.fit(r_data)
        return pca.transform(r_data)

    def tf_process(self):
        images = tf.keras.preprocessing.image.ImageDataGenerator()
        return images.flow_from_directory(self.path)
