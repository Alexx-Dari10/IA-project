import cv2
import imutils
import os

class ImageProcessing:
    def __init__(self):
        # initialize the raw pixel intensities matrix, the features matrix,
        # and labels list
        self.rawImages = []
        self.features = []
        self.labels = []

    def image_to_feature_vector(self, image, size=(32, 32)):
        # resize the image to a fixed size, then flatten the image into
        # a list of raw pixel intensities
        return cv2.resize(image, size).flatten()


    def extract_color_histogram(self, image, bins=(8, 8, 8)):
        # extract a 3D color histogram from the HSV color space using
        # the supplied number of `bins` per channel
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
            [0, 180, 0, 256, 0, 256])
        # handle normalizing the histogram if we are using OpenCV 2.4.X
        if imutils.is_cv2():
            hist = cv2.normalize(hist)
        # otherwise, perform "in place" normalization in OpenCV 3 (I
        # personally hate the way this is done
        else:
            cv2.normalize(hist, hist)
        # return the flattened histogram as the feature vector
        return hist.flatten()

    def cv_process(self, path):
        imagePaths = list(path)

                # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label (assuming that our
            # path as the format: /path/to/dataset/{class}.{image_num}.jpg
            image = cv2.imread(imagePath)
            

            label = imagePath.split(os.path.sep)[1] ##################################################cambiar aqui si se le cambia la direccion del dataset
            # extract raw pixel intensity "features", followed by a color
            # histogram to characterize the color distribution of the pixels
            # in the image

            
            pixels = self.image_to_feature_vector(image)
            hist = self.extract_color_histogram(image)
            # update the raw images, features, and labels matricies,
            # respectively
            self.rawImages.append(pixels)
            self.features.append(hist)
            self.labels.append(label)

            # show an update every 1,000 images
            if i > 0 and i % 500 == 0:
                print("[INFO] processed {}/{}".format(i, len(imagePaths)))

            