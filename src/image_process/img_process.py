import math

import cv2
from numpy import argmax, pad
from keras.models import load_model

model = load_model('cnn_model/cnn_model.h5')


def predict_digit(img):
    test_image = img.reshape(-1, 28, 28, 1)
    print(test_image.shape)
    return argmax(model.predict(test_image))


def img_preprocess(path):
    img = cv2.imread(path, 2)
    img_org = cv2.imread(path)
    # remove the noise
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for j, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        if (hierarchy[0][j][3] != -1 and w > 10 and h > 10):
            # putting boundary on each digit
            cv2.rectangle(img_org, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # cropping each image and process
            roi = img[y:y + h, x:x + w]
            roi = cv2.bitwise_not(roi)
            roi = image_refiner(roi)

            # getting prediction of cropped image
            pred = predict_digit(roi)
            if pred is None:
                print("no")
                return predict_digit(img)
            else:
                print("yes")
                print(pred)
                return pred


# refining each digit
def image_refiner(gray):
    org_size = 22
    img_size = 28
    rows, cols = gray.shape

    if rows > cols:
        factor = org_size / rows
        rows = org_size
        cols = int(round(cols * factor))
    else:
        factor = org_size / cols
        cols = org_size
        rows = int(round(rows * factor))
    gray = cv2.resize(gray, (cols, rows))

    # get padding
    colsPadding = (int(math.ceil((img_size - cols) / 2.0)), int(math.floor((img_size - cols) / 2.0)))
    rowsPadding = (int(math.ceil((img_size - rows) / 2.0)), int(math.floor((img_size - rows) / 2.0)))

    # apply apdding
    gray = pad(gray, (rowsPadding, colsPadding), 'constant')
    return gray
