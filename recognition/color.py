from preproc import Quantize
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import importlib_resources as pkg_resources
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
import cv2

class HighlightDetector:
    NUM_CLASSES = 2
    TARGET_SIZE = (150,150, 3)
    MODEL_PATH = pkg_resources.files("recognition.data").joinpath("highlight_model_30e.h5")
    # remember to add __init__.py to data/ folder

    def __init__(self, target_size=None):
        self.input_size = target_size or HighlightDetector.TARGET_SIZE

        self.model = KerasClassifier(
            build_fn=HighlightDetector._build_model,
            input_size=self.input_size,
            epochs=30,
            batch_size=128,
            verbose=1
        )

    @staticmethod    
    def minmax_scaler(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    @staticmethod
    def preprocess(image):
        """
        Parameters
        ----------
            image: ndarray image in RGB mode
        """
        image = Quantize.reduce_palette(image, num_colors=4)
        return HighlightDetector.minmax_scaler(image)

    @staticmethod
    def _build_model(input_size):
        model = Sequential()

        model.add(Conv2D(32, (3,3), activation="relu", input_shape=input_size, name="conv1"))
        model.add(MaxPooling2D((2,2), name="pool1"))
        
        model.add(Conv2D(64, (3,3), activation="relu", name="conv2"))
        model.add(MaxPooling2D((2,2), name="pool2"))
        
        model.add(Flatten())
        model.add(Dense(64, activation='relu', name="dense1"))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid', name="dense2"))
        
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def plot_results(training_results):
        _, axes = plt.subplots(1,2, figsize=(10,5))
        axes = axes.flatten()

        axes[0].plot(training_results.history["accuracy"])
        axes[0].set_title("Training Accuracy")

        axes[1].plot(training_results.history["loss"])
        axes[1].set_title("Training Loss")

        plt.show()

    def train(self, X_train, Y_train, epochs=30, plot=False):
        training_results = self.model.fit(X_train, Y_train, verbose=1)
        if plot:
            HighlightDetector.plot_results(training_results)

    def predict(self, X_test):
        """"""
        return self.model.predict(X_test).flatten()

    def cross_validate(self, X, Y, k=10, epochs=30, batch_sz=128):
        kfold = StratifiedKFold(n_splits=k, shuffle=True)
        results = cross_val_score(self.model, X, Y, cv=kfold)
        print(f"Baseline: {(results.mean() * 100):.2f}% ({(results.std()*100):.2f}%)")

    def load_model(self, fpath=None):
        fpath = fpath or str(HighlightDetector.MODEL_PATH)
        self.model = keras.models.load_model(fpath)

# ------

def image_loader(fpath, source_dir, target_size=(150,150)):
    image = cv2.cvtColor(cv2.imread(f"{source_dir}/{fpath}.png"), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size, cv2.INTER_CUBIC)
    return HighlightDetector.preprocess(image)

if __name__ == "__main__":
    # data is a dataframe in style of IAM dataset with row
    # row["id"] holds the image filename (with .png extension)
    # row["highlighted"] has "highlighted" value for highlighted words, "non-highlighted" otherwise
    # source_dir is the dir where all filenames are located

    X = np.zeros((len(data), *HighlightDetector.TARGET_SIZE))
    for idx, row in data.iterrows():
        X[idx, :, :, :] = image_loader(row["id"], SOURCE_DIR)

    Y = data["highlighted"]
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, encoded_Y, test_size=0.2, shuffle=True)

    net = HighlightDetector()
    # net.cross_validate(X, encoded_Y)
    net.train(X_train, Y_train, plot=False)
    