import os

from image_processing.image_processing import img_processing
from neural_networks.animals_detection import animals_detection
from neural_networks.object_detection import object_detection
from supervised_algorithms.supervised_algoritms import sup_algoritms
from unsupervised_algorithms.unsupervised_algorithms import unsup_algoritms


path_dataset_Animals = os.getcwd() + '\datasets\Animals'
path_dataset_Animals_test = os.getcwd() + '\datasets\Animals_test'
path_dataset_Animals_5 = os.getcwd() + '\datasets\Animals_5'
path_dataset_Animals_with_anomalies = os.getcwd() + '\datasets\Animals_with_anomalies'


# =================================================================================
# supervisado ----------------------------------------------------------------------
img = img_processing(path_dataset_Animals_5)

run_supervised = sup_algoritms()
run_supervised.run(img.reduction(img.rawImages), img.labels, img.types)


# redes neuronales ------------------------------------------------------------------
img = img_processing(path_dataset_Animals, False)
img_test = img_processing(path_dataset_Animals_test, False)

animal_neural_net = animals_detection()
animal_neural_net.run(img, img_test)

object_neural_net = object_detection()
object_neural_net.run()
# ===================================================================================

# no supervisado ---------------------------------------------------------------------
img = img_processing(path_dataset_Animals_with_anomalies)

run_unsupervised = unsup_algoritms()
run_unsupervised.run(img.reduction(img.rawImages), [1 if i < 3000 else -1 for i in range(3087)])