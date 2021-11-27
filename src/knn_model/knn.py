import pickle
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
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


def knn_model():
    # load the data
    trainImage, trainLabel, testImage, testLabel = load_mnist_dataset()
    print('Training the Model')
    classifier = KNeighborsClassifier(n_neighbors=7)
    classifier = classifier.fit(trainImage, trainLabel)
    y_pred = classifier.predict(testImage)
    print(accuracy_score(testLabel, y_pred))
    modelName = "knn_model.pkl"
    with open(modelName, 'wb') as file:
        pickle.dump(classifier, file)


knn_model()
