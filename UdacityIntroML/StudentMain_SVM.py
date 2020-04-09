import sys
from prep_terrain_data import makeTerrainData
from class_viz import prettyPicture

import matplotlib.pyplot as plt

import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
### create classifier (linear | poly | rbf | sigmoid, ...)
#clf = SVC(kernel="linear", gamma=1)
clf = SVC(kernel="rbf", C=1000)

### fit the classifier on the training features and labels
clf.fit(features_train, labels_train)

#### store your predictions in a list named pred
pred = clf.predict(features_test)


from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print "accuracy =", acc

### draw the decision boundary with the text points overlaid

prettyPicture(clf, features_test, labels_test)
#plt.switch_backend('Agg')
plt.ion()
plt.show()


def submitAccuracy():
    return acc