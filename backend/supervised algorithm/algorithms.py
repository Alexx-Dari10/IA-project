
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

import warnings

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os


warnings.filterwarnings("ignore")

from dotenv import dotenv_values

config = dotenv_values("../.env")

dataset = config['DATASET_ALG']



def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
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


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="number of dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())


print("[INFO] describing images...")
imagePaths = list(paths.list_images(dataset + args["dataset"]))

animalsLabels = os.listdir(dataset + args["dataset"])

# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []



# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	
	print(imagePaths)

	label = imagePath.split(os.path.sep)[1] ##################################################cambiar aqui si se le cambia la direccion del dataset
	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels
	# in the image

	
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	# update the raw images, features, and labels matricies,
	# respectively
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)
	# show an update every 1,000 images
	if i > 0 and i % 500 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))



# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
(X_train, X_test, y_train, y_test) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)



supervised_algorithms = [("GaussianNB", GaussianNB()),
                            ("KNeighborsClassifier", KNeighborsClassifier()),
                            ("DecisionTreeClassifier", DecisionTreeClassifier(criterion = "entropy")),
                            ("RandomForestClassifier", RandomForestClassifier(criterion = "entropy"))]


# Supervised algorithms


for name, algorithm in supervised_algorithms:
    
	algorithm.fit(X_train, y_train)
	
	if len(animalsLabels) <= 5:

		disp = plot_confusion_matrix(algorithm, X_test, y_test,
										display_labels=animalsLabels,
										cmap=plt.cm.Blues,
										normalize='true')
		disp.ax_.set_title("Algorithm: " + name + "\n" + "Score: " + str(algorithm.score(X_test, y_test)))
				
		plt.show()

	else:	
		print(name + " has score: " + str(algorithm.score(X_test, y_test)))
    	

# for execute use 
# pyhon algorithms --dataset <number of dataset>

   

    

