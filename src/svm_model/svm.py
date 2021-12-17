import gzip
import pickle
import time

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from tensorflow.keras.datasets import mnist


def load_mnist_dataset():
    """
    Function to load and normalizing the data
    """
    # load the data from tensorflow
    (trainImage, trainLabel), (testImage, testLabel) = mnist.load_data()
    trainImage = trainImage.reshape(trainImage.shape[0], 28 * 28)
    testImage = testImage.reshape(testImage.shape[0], 28 * 28)
    return trainImage, trainLabel, testImage, testLabel


def svm_model():  # 88.6%
    # load the data
    trainImage, trainLabel, testImage, testLabel = load_mnist_dataset()
    print(trainImage.shape, trainLabel.shape, testImage.shape, testLabel.shape)
    print('Training the Model')
    trainImage, trainLabel = shuffle(trainImage, trainLabel, random_state=0)
    classifier = LinearSVC()
    start = time.time()
    classifier = classifier.fit(trainImage, trainLabel)
    end = time.time()
    print(end - start, "seconds") # 169.29505586624146 seconds
    print("Fitted")
    start = time.time()
    y_pred = classifier.predict(testImage)
    end = time.time()
    print(end - start, "seconds")
    print(accuracy_score(testLabel, y_pred))
    plot_confusion_matrix(classifier, testImage, testLabel)
    plt.show()
    modelName = "svm_model.gz"
    with gzip.open(modelName, 'wb') as file:
        pickle.dump(classifier, file)


svm_model()
