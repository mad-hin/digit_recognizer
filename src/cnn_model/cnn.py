import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # removing noise

    _, x_train_th = cv2.threshold(x_train, 127, 255, cv2.THRESH_BINARY)
    _, x_test_th = cv2.threshold(x_test, 127, 255, cv2.THRESH_BINARY)

    # Reshaping

    x_train = x_train_th.reshape(-1, 28, 28, 1)
    x_test = x_test_th.reshape(-1, 28, 28, 1)

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    return x_train, y_train, x_test, y_test


def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def export_cnn_model():
    x_train, y_train, x_test, y_test = get_mnist()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    model = create_cnn_model()
    model.fit(x_train, y_train, epochs=10, shuffle=True, batch_size=32, validation_data=(x_test, y_test))
    _, acc = model.evaluate(x_test, y_test, verbose=1)
    model.save("cnn_model2.h5")


export_cnn_model()
