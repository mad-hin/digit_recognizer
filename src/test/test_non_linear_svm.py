import gzip
import pickle

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.python.ops.confusion_matrix import confusion_matrix

from src.image_process.img_process import img_preprocess


def predict_digit(img):
    with gzip.open("../svm_model/non_linear_svm_model.gz", 'rb') as file:
        model = pickle.load(file)
        test_image = img.reshape(-1, 28, 28, 1)
        test_image = test_image.reshape(test_image.shape[0], 28 * 28)
        pred = model.predict(test_image)
        print(int(pred))
        return int(pred)


lable = [3, 4, 9, 7, 6, 5, 4, 3, 2, 1, 0, 6, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 7, 9, 8, 5, 4, 2, 1, 7, 6, 9, 6, 7,
         2, 0, 1, 3, 4, 5, 8, 8, 7, 7, 2, 4, 5, 9, 9, 0, 8, 1, 0, 3, 4, 5, 6, 7, 8, 9, 0, 2, 2, 1, 3, 9, 9, 9, 0, 3, 2,
         3, 5, 6, 5, 7, 8, 2, 7, 0, 8, 0, 1, 1, 2, 8, 6, 5, 6, 4, 1, 3, 5, 4, 6, 3, 4]
test_predict = []
for cnt in range(100):
    s = "../images/hw_test_" + str(cnt) + ".png"
    img = img_preprocess(s)
    predict = predict_digit(img)
    test_predict.append(predict)

matrix = confusion_matrix(test_predict, lable)
import seaborn as sns

fig, ax = plt.subplots(figsize=(15, 10))
sns.set(font_scale=1.4)
ax = sns.heatmap(matrix, linewidths=1, annot=True, ax=ax, cmap='Blues', fmt='g')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.set_title("Confusion Matrix of the Non-linear SVM model (User Handwriting)")
plt.show()
print(classification_report(test_predict, lable))
"""
              precision    recall  f1-score   support

           0       1.00      0.91      0.95        11
           1       0.90      0.90      0.90        10
           2       1.00      0.71      0.83        14
           3       0.90      0.47      0.62        19
           4       0.70      0.78      0.74         9
           5       1.00      0.53      0.69        19
           6       0.30      0.75      0.43         4
           7       0.00      0.00      0.00         2
           8       0.80      0.73      0.76        11
           9       0.10      1.00      0.18         1

    accuracy                           0.67       100
   macro avg       0.67      0.68      0.61       100
weighted avg       0.86      0.67      0.73       100
"""
