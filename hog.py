import os
import numpy as np
import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from prepare_images import prepare_hog
from joblib import dump, load
import seaborn as sb
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))


nbins = 9
cell_size = (16, 16)
block_size = (4, 4)
image_shape = (128, 128)
features = []

hog = cv2.HOGDescriptor(_winSize=(image_shape[1] // cell_size[1] * cell_size[1],
                                  image_shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

train_images, train_labels, test_images, test_labels = prepare_hog(root_directory='Dataset')

for image in train_images:
    hog_comp = hog.compute(image)
    print('hog: ', hog_comp)
    features.append(hog_comp)

x_train = np.array(features)


x_train = reshape_data(x_train)
print(x_train.shape)
clf_svm = SVC(kernel='linear', probability=False, verbose=True)
clf_svm.fit(x_train, train_labels.ravel())
dump(clf_svm, 'svm_16_4_128_rc.joblib')

# clf_svm = load('svm_model.joblib')
# print("Loaded")

features_test = []
for image in test_images:
    hog_comp = hog.compute(image)
    features_test.append(hog_comp)

x_test = np.array(features_test)
x_test = reshape_data(x_test)

predictions = clf_svm.predict(x_test)

cm = confusion_matrix(test_labels, predictions)
sb.heatmap(cm, annot=True, fmt='', cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Labels')
plt.show()

number_labels = list(range(0, 33))
report = classification_report(test_labels, predictions, target_names=number_labels, output_dict=True)
sb.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
plt.show()









