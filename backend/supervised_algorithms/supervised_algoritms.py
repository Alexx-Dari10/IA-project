import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix



class sup_algoritms:

    def __init__(self):
        self.algorithms = [("GaussianNB", GaussianNB()),
                            ("KNeighborsClassifier", KNeighborsClassifier()),
                            ("DecisionTreeClassifier", DecisionTreeClassifier(criterion = "entropy")),
                            ("RandomForestClassifier", RandomForestClassifier(criterion = "entropy"))]
    
    def run(self, data, labels, types):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

        for name, algorithm in self.algorithms:
    
            algorithm.fit(X_train, y_train)
            
            if len(types) <= 5:

                disp = plot_confusion_matrix(algorithm, X_test, y_test,
                                                display_labels = types,
                                                cmap=plt.cm.Blues,
                                                normalize='true')

                disp.ax_.set_title("Algorithm: " + name + "\n" + "Score: " + str(algorithm.score(X_test, y_test)))
                        
                plt.show()

            else:	
                print(name + " has score: " + str(algorithm.score(X_test, y_test)))
