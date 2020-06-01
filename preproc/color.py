
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path

import importlib_resources as pkg_resources  # backport of core 3.7 library
import pickle as pk
import numpy as np
import os

class ColorGroup:
    """
    Color grouping functions

    Source:
        https://blog.algolia.com/how-we-handled-color-identification/
        https://blog.xkcd.com/2010/05/03/color-survey-results/
    """
    # KNearestNeighbor with K=10 and uniform weights
    MODEL_PATH = pkg_resources.open_binary("preproc.data", "knn_10_uniform.pk")
    LABELS_PATH = pkg_resources.open_binary("preproc.data", "color_names.npz")

    def __init__(self):
        """Loads the saved model"""
        self.model = pk.load(ColorGroup.MODEL_PATH)

    def predict(self, color):
        """Predicts a new color from saved models
        Parameters
        ----------
            color: numpy array of shape (1,3)
                A RGB color
        
        Returns
        -------
            str, predicted color class
        """
        if color.shape != (1,3):
            raise ValueError(f"{repr(color)} must be of shape (1,3)")
        return self.model.predict(color)[0]

    @staticmethod
    def simple_predict(color):
        """A simple predictor using Euclidean Distance
        Parameters
        ----------
            color: numpy array of shape (1,3)
                A RGB color
        
        Returns
        -------
            str, predicted color class
        """
        data = np.load(ColorGroup.LABELS_PATH)
        samples = data.get("samples")
        labels = data.get("labels")

        euclid_dist = np.sqrt(np.sum((samples - test_color) ** 2, axis=1))
        idx = np.argmin(euclid_dist)
        return labels[idx]

    @staticmethod
    def kneighbors_predict(color, n_neighbors=10, weights="uniform"):
        """Builds a new KNearestNeighbor classifier
        Parameters
        ----------
            color: numpy array of shape (1,3)
                A RGB color
        
        Returns
        -------
            str, predicted color class
        """
        data = np.load(ColorGroup.LABELS_PATH)
        samples = data.get("samples")
        labels = data.get("labels")

        model = KNeighborsClassifier(n_neighbors, weights=weights)
        model.fit(samples, labels)
        return model.predict(color)[0]


        