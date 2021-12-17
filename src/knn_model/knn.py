import gzip
import pickle
import time

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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


def knn_model():  # 97.05 96.94
    # load the data
    trainImage, trainLabel, testImage, testLabel = load_mnist_dataset()
    print(trainImage.shape, trainLabel.shape, testImage.shape, testLabel.shape)
    print('Training the Model')

    # shuffle the data
    trainImage, trainLabel = shuffle(trainImage, trainLabel, random_state=0)

    # choose the KNN Classifier with K = 3
    classifier = KNeighborsClassifier(n_neighbors=3)

    # train it
    start = time.time()
    classifier = classifier.fit(trainImage, trainLabel)
    end = time.time()
    print(end - start, "seconds") #0.004902362823486328 seconds (K =3)
    print("Fitted")

    # get the score
    start = time.time()
    y_pred = classifier.predict(testImage)
    end = time.time()
    print(end - start, "seconds") # 32.95009398460388 seconds
    print(accuracy_score(testLabel, y_pred))
    plot_confusion_matrix(classifier, testImage, testLabel)
    plt.title("Confusion matrix of KNN with K = 7")
    plt.show()
    # export the model
    modelName = "knn_model.gz"
    with gzip.open(modelName, 'wb') as file:
        pickle.dump(classifier, file)


knn_model()
