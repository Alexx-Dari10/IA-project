import os

from image_processing.image_processing import img_processing
from neural_network.cnn import cnn
from supervised_algorithms.supervised_algoritms import sup_algoritms
from unsupervised_algorithms.unsupervised_algorithms import unsup_algoritms


path_dataset_Animals = os.getcwd() + '\datasets\Animals'
path_dataset_Animals_with_anomalies = os.getcwd() + '\datasets\Animals_with_anomalies'


# # supervisado -------------------------------------------------------
# img = img_processing(path_dataset_Animals)

# run_supervised = sup_algoritms()
# run_supervised.run(img.rawImages, img.labels, img.types)


# # no supervisado -----------------------------------------------------
# img = img_processing(path_dataset_Animals_with_anomalies)

# run_unsupervised = unsup_algoritms()
# run_unsupervised.run(img.reduction(img.rawImages), [1 if i < 3000 else -1 for i in range(3087)])

# CNN
img = img_processing(path_dataset_Animals, False)
neural_net = cnn()
neural_net.run(img.tf_process())
