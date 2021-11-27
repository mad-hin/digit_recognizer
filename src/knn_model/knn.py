import gzip
import pickle
from sklearn.metrics import accuracy_score
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


def knn_model(): # 96.94
    # load the data
    trainImage, trainLabel, testImage, testLabel = load_mnist_dataset()
    print(trainImage.shape, trainLabel.shape, testImage.shape, testLabel.shape)
    print('Training the Model')
    trainImage, trainLabel = shuffle(trainImage, trainLabel, random_state=0)
    classifier = KNeighborsClassifier(n_neighbors=7)
    classifier = classifier.fit(trainImage, trainLabel)
    y_pred = classifier.predict(testImage)
    print(accuracy_score(testLabel, y_pred))
    modelName = "knn_model.gz"
    with gzip.open(modelName, 'wb') as file:
        pickle.dump(classifier, file)


knn_model()
