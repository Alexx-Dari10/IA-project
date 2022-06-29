import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import metrics
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest


class unsup_algoritms:

    def __init__(self):
        outliers_fraction = 0.15

        self.algorithms = [
                ("One Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel='rbf', gamma=0.1)),
                ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
                ("Isolation Forest", IsolationForest(contamination=outliers_fraction, random_state=42))]
    
    def run(self, X, y):
        for name,algorithm in self.algorithms:
            print(name + '\n')

            y_predict = algorithm.fit(X).predict(X)


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