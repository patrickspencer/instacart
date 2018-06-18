# -*- coding: utf-8 -*-

"""
model.py
~~~~~~~~

A class for the model. Data is expected to be preprocessed before going in the
model.
"""
import pandas as pd
from sklearn.metrics import classification_report


class Model:

    def __init__(self, clf, X_train, y_train, X_test):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.clf_fit = clf.fit(X_train, y_train)
        self.y_pred = clf.predict(X_train)
        self.classification_report = classification_report(y_train, self.y_pred)
