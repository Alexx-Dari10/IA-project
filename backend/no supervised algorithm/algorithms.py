from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from imutils import paths
import cv2
import imutils

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf


from dotenv import dotenv_values
config = dotenv_values("../.env")



dataset_no_supervised = config["DATASET_NO_SUPERVISED"]
#dataset_no_supervised = "../datasets/No_supervised"

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

outliers_fraction = 0.15

non_supervised_algorithms = [
        ("One Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel='rbf', gamma=0.1)),
        ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
        ("Isolation Forest", IsolationForest(contamination=outliers_fraction, random_state=42))
    ]

X = []
imagePaths = list(paths.list_images(dataset_no_supervised))



for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	
	#print(imagePaths)

	
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	# update the raw images, features, and labels matricies,
	# respectively
	X.append(pixels)

y = [1]*3103
y[5] = y[6] = y[7] = y[12] = y[15] = -1

scaler = StandardScaler()
X = scaler.fit_transform(X)

pca = PCA(n_components=2)
pca.fit(X)

X = pca.transform(X)


for name,algorithm in non_supervised_algorithms:
    print(name + '\n')

    y_predict = algorithm.fit(X).predict(X)
    #print("y_predict \n" + str(y_predict))


    # Calcular la homogeneidad y la integridad de los clusters.
    
    homogeneity = metrics.homogeneity_score(y, y_predict)
    
    completeness = metrics.completeness_score(y, y_predict)
    
    # Calcular el coeficiente de coeficiente de Silhouette para cada muestra.

    s = metrics.silhouette_samples(X, y_predict)
    
    # Calcule el coeficiente de Silhouette medio de todos los puntos de datos.

    s_mean = metrics.silhouette_score(X, y_predict)

    
    # Para la configuración de los graficos -----------------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    
    # Configura el gráfico.
    plt.suptitle('Silhouette analysis ' + name + ' : {}'.format(2),
                fontsize=14, fontweight='bold')
    
    # Configura el 1er subgrafico.
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([0, len(X) + (7) * 10])
    
    # Configura el 2do subgrafico.
    plt.suptitle('Silhouette analysis ' + name + ' : ' + '\n Homogeneity: {}, Completeness: {}, Mean Silhouette score: {}'.format(homogeneity,
                                                                                        completeness,
                                                                                        s_mean))
    
    
    # Para el 1er subgráfico ------------------------------------------------------------------------------------------
    
    # Grafica el coeficiente de Silhouette para cada muestra.
    cmap = cm.get_cmap("Spectral")
    y_lower = 10
    for i in range(2):
        ith_s = s[y_predict == i]
        ith_s.sort()
        size_i = ith_s.shape[0]
        y_upper = y_lower + size_i
        color = cmap(float(i) / 2)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_s,
                        facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10
        
    # Trazar el coeficiente de silueta medio utilizando la línea discontinua vertical roja.
    ax1.axvline(x=s_mean, color="red", linestyle="--")
    
    # Para el 2do subgráfico 
    
    # Grafica las predicciones
    colors = cmap(y_predict.astype(float) / 2)
    ax2.scatter(X[:,0], X[:,1], c=colors)

    print(y_predict)
    
    plt.show()
