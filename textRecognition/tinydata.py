from pandas import DataFrame as DF
from tqdm import tqdm

import pandas as pd
import numpy as np
import h5py as h5
import cv2
import os

class TinyData:
    """A set of functions to minify HTR word image datasets into HDF5 files.
    It saves the text labels as one-hot vectors.
    """

    @staticmethod
    def imread(fpath:str, color=False):
        if color:
            return cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
        else:
            return cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        
    @staticmethod
    def minify(train, test, valid, source_dir, target_path, name_label, x_label="fpath", y_label="word", color=False):
        """Minifies the training/test/validation splits in the target_path file.
        It needs the filepath column label and the Y class column label.
        """

        with h5.File(target_path, "w") as hf:
            
            for _, row in tqdm(train.iterrows()):
                fpath = f"{source_dir}/{row[x_label]}"
                image = TinyData.imread(fpath, color=color)
                label = row[y_label].encode("utf-8")  # turns Unicode into ASCII bytes
                nm = row[name_label]
                
                if image is None:
                    raise ValueError(f"(Train Set) Image {fpath} has size 0. Please remove it from the dataset.")
                    
                new_data = hf.create_dataset(f"train/{nm}", data=image, dtype=image.dtype, compression="gzip", compression_opts=9)
                new_data.attrs["label"] = label

            for _, row in tqdm(test.iterrows()):
                fpath = f"{source_dir}/{row[x_label]}"
                image = TinyData.imread(fpath, color=color)
                label = row[y_label].encode("utf-8")
                nm = row[name_label]
                
                if image is None:
                    raise ValueError(f"(Test Set) Image {fpath} has size 0. Please remove it from the dataset.")
                    
                new_data = hf.create_dataset(f"test/{nm}", data=image, dtype=image.dtype, compression="gzip", compression_opts=9)
                new_data.attrs["label"] = label
                

            if valid is not None:
                valid_set = hf.create_group("valid")
                for _, row in tqdm(valid.iterrows()):
                    fpath = f"{source_dir}/{row[x_label]}"
                    image = TinyData.imread(fpath, color=color)
                    label = row[y_label].encode("utf-8")
                    nm = row[name_label]
                    
                    if image is None:
                        raise ValueError(f"(Validation Set) Image {fpath} has size 0. Please remove it from the dataset.")

                    new_data = hf.create_dataset(f"valid/{nm}", data=image, dtype=image.dtype, compression="gzip", compression_opts=9)
                    new_data.attrs["label"] = label

        print("Done")

