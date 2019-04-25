# Simple/Base SVM on MNIST Data

from __future__ import print_function, division
from builtins import range
from sklearn.svm import SVC
from util import getKaggleMNIST
from datetime import datetime

X_train, y_train, X_test, y_test = getKaggleMNIST()

model = SVC()

t0 = datetime.now()
model.fit(X_train, y_train)
print("Train duration: ", datetime.now() - t0)
t0 = datetime.now()
print("Train score: ", model.score(X_train, y_train), "Duration: ", datetime.now() - t0)
t0 = datetime.now()
print("Test score: ", model.score(X_test, y_test), "Duration: ", datetime.now() - t0)