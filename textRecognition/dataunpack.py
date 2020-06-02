from tensorflow.keras.preprocessing.sequence import pad_sequences
from preproc import RobustBinarize
from unicodedata import normalize
from numpy import ndarray

import pandas as pd
import numpy as np
import h5py as h5
import cv2
import os

class DataUnpack:
    """A set of functions to load Image datasets in HDF5 format into ImageDataGenerator Keras objects.
    It works along with the TinyData class.
    
    Source:
        https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/data/preproc.py
        https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/data/generator.py
        https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/data/evaluation.py
        https://machinelearnings.co/deep-spelling-9ffef96a24f6#.2c9pu8nlm
    """
    @staticmethod
    def imwrite(fpath, image, color=False):
        if color:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(fpath, image)
        else:
            cv2.imwrite(fpath, image)

    @staticmethod
    def resize(name, image: ndarray, target_size, aspect_ratio: float=None):
        """Resizes a grayscale image

        image: numpy ndarray with shape (height, width) and 0 channels

        Source:
            https://stackoverflow.com/questions/58248121/opencv-python-how-to-overlay-an-image-into-the-centre-of-another-image
            https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
        """
        h, w = image.shape
        image_ratio = np.round(w / h, 2)
        new_image = image.copy()

        if aspect_ratio:
            ratio_diff = abs(np.round(aspect_ratio, 2) - image_ratio)
            if ratio_diff > 0.1:
                mult_factor = aspect_ratio / image_ratio
                resized_w = int(mult_factor * w)
                resized_h = h
                
                if resized_w < w: # as we're padding the image, we don't want to crop it
                    # so instead of resizing width, it resizes height
                    mult_factor = image_ratio / aspect_ratio
                    resized_w = w
                    resized_h = int(mult_factor * h)

                new_image = np.ones((resized_h, resized_w), np.uint8) * 255
                hback, wback = new_image.shape
                
                yoff, xoff = ((hback - h) // 2, (wback - w) // 2)
                new_image[yoff:(yoff + h), xoff:(xoff + w)] = image

        mean = np.sum(new_image.shape[:2]) // 2
        maxi = np.max(new_image.shape[:2])
        interpolation = cv2.INTER_AREA if maxi > mean else cv2.INTER_CUBIC
        return cv2.resize(new_image, target_size, interpolation)

        

    @staticmethod
    def unpack_set(set_name, dest_dir, file, color, target_size=None, aspect_ratio=None, max_word_length=34):
        """Performs the unpacking step for each dataset split (train/test/validation)"""
        result = pd.DataFrame(columns=["name", "label"])
        vectorizer = StringVectorizer(max_word_length=34)

        names, labels, vector_labels = [], [], [f"X{i}" for i in range(max_word_length)]

        os.makedirs(f"{dest_dir}/{set_name}", exist_ok=True)

        for sample_name in file[set_name]:
            data = file[f"{set_name}/{sample_name}"]
            image = data[:]
            label = data.attrs["label"].decode("utf-8")

            names.append(sample_name)
            labels.append(label)
            
            if target_size and aspect_ratio: # are not None
                image = DataUnpack.resize(sample_name, image, target_size, aspect_ratio)

            # Remove cursive style (WORKS BADLY FOR 'train_spa/w02_bw_22' and takes so much time)
            # image = RobustBinarize.remove_cursive_style(image)

            # at the time of writing, it should follow the folder structure for RIMES and IAM
            DataUnpack.imwrite(f"{dest_dir}/{set_name}/{sample_name}.png", image, color=color)

        result["name"] = names
        result["fname"] = result["name"].apply(lambda x: x + ".png")  # add the filename
        result["label"] = labels
        
        # create a column for each encoded character label
        for lb in vector_labels:
            result[lb] = int()
        # fill all encoded character labels
        for idx, row in result[["label"]].iterrows():
            result.loc[idx, vector_labels] = vectorizer.vectorize(row.label)

        return result

    @staticmethod
    def unpack(input_path, dest_dir, color=False, save_df=False, target_size=None, aspect_ratio:float=None, max_word_length=34):
        """Deminifies / Unpacks the compressed file into train/test/validation DataFrames.
        Images are saved in .png format.
        """
        with h5.File(input_path, "r") as f:
            train = DataUnpack.unpack_set("train", dest_dir, f, color, target_size, aspect_ratio, max_word_length)
            test = DataUnpack.unpack_set("test", dest_dir, f, color, target_size, aspect_ratio, max_word_length)

            # if there is a valid split stored, then extract it as well
            if f.get("valid", None):
                valid = DataUnpack.unpack_set("valid", dest_dir, f, color, target_size, aspect_ratio, max_word_length)
            else:
                valid = None

        if save_df:
            train.to_csv(f"{dest_dir}/train/train.csv", sep=",", index=False)
            test.to_csv(f"{dest_dir}/test/test.csv", sep=",", index=False)
            valid.to_csv(f"{dest_dir}/valid/valid.csv", sep=",", index=False)

        return train, test, valid



class StringVectorizer:
    """A set of functions to encode strings into integer arrays given a character set.
    
    Parameters
    ----------
        max_word_length: int, Default 34
            Set by the most common longest word in French, English and Spanish
            Sources:
                https://en.wikipedia.org/wiki/Longest_word_in_Spanish
                https://en.wikipedia.org/wiki/Longest_word_in_English
                https://en.wikipedia.org/wiki/Longest_word_in_French
    """
    # CHARSET = WordHTRFlor.LATIN_CHAR
    CHARSET = " !\"#$%&'()*+,-.0123456789:;<>@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáÁéÉíÍóÓúÚëËïÏüÜñÑçÇâÂêÊîÎôÔûÛàÀèÈùÙ"
    
    def __init__(self, max_word_length=34):
        self.PAD_TK, self.UNK_TK = "¶", "¤"
        self.chars = (self.PAD_TK + self.UNK_TK + StringVectorizer.CHARSET)

        self.PAD = self.chars.find(self.PAD_TK)
        self.UNK = self.chars.find(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_word_length
    
    def vectorize(self, word, as_onehot=False):
        """Encode word to integer vector (or binary vector if 'as_onehot' set to True)
        Spanish characters work with NFKC, not NFKD (https://docs.python.org/3/library/unicodedata.html)

        Source:
            https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/data/generator.py#L115
        """
        word = normalize("NFKC", word).encode("UTF-8", "ignore").decode("UTF-8")
        
        vector = (self.chars.find(word[i]) for i in range(len(word)))
        vector = [num if num != -1 else self.UNK for num in vector]
                
        padded_vector = pad_sequences([vector], maxlen=self.maxlen, padding="post")
        return padded_vector[0]
        
    def decode(self, vector, as_onehot=False):
        """Inverses the vectorization"""
        chars = "".join([self.chars[pos] for pos in vector])
        return chars.replace(self.PAD_TK, "").replace(self.UNK_TK, "")
    