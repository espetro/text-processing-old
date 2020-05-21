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
    def imwrite(fpath, image, color=False):
        if color:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imsave(fpath, image)
        else:
            cv2.imsave(fpath, image)
        
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

                new_data = hf.create_dataset(f"train/{nm}", data=image, compression="gzip", compression_opts=9)
                new_data.attrs["label"] = label

            for _, row in tqdm(test.iterrows()):
                fpath = f"{source_dir}/{row[x_label]}"
                image = TinyData.imread(fpath, color=color)
                label = row[y_label].encode("utf-8")
                nm = row[name_label]

                new_data = hf.create_dataset(f"test/{nm}", data=image, compression="gzip", compression_opts=9)
                new_data.attrs["label"] = label
                

            if valid is not None:
                valid_set = hf.create_group("valid")
                for _, row in tqdm(valid.iterrows()):
                    fpath = f"{source_dir}/{row[x_label]}"
                    image = TinyData.imread(fpath, color=color)
                    label = row[y_label].encode("utf-8")
                    nm = row[name_label]

                    hf.create_dataset(f"valid/{nm}", data=image, compression="gzip", compression_opts=9)
                    new_data.attrs["label"] = label

        print("Done")
    
    @staticmethod
    def unpack(input_path, dest_dir, color=False):
        """Deminifies / Unpacks the compressed file into train/test/validation DataFrames.
        Images are saved in .png format.
        """
        train, test, valid = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        os.mkdirs(f"{dest_dir}/train", exist_ok=True)
        os.mkdirs(f"{dest_dir}/test", exist_ok=True)
        os.mkdirs(f"{dest_dir}/valid", exist_ok=True)

        with h5.File(input_path, "r") as hf:
            names, labels = []
            for data in hf["train"]:
                name = data.name.split("/")[-1]
                image = data[:]
                label = data.attrs["label"][:].item(0).decode("utf-8")

                names.append(name)
                labels.append(label)
                TinyData.imwrite(f"{source_dir}/train/{name}.png", image, color=color)

            train["name"] = names
            train["label"] = labels

            names, labels = []
            for data in hf["test"]:
                name = data.name.split("/")[-1]
                image = data[:]
                label = data.attrs["label"][:].item(0).decode("utf-8")

                names.append(name)
                labels.append(label)
                TinyData.imwrite(f"{source_dir}/test/{name}.png", image, color=color)

            test["name"] = names
            test["label"] = labels

            have_valid = hf.get("valid", None) # returns None if it doesn't find valid dataset
            if have_valid:
                names, labels = []
                for data in hf["valid"]:
                    name = data.name.split("/")[-1]
                    image = data[:]
                    label = data.attrs["label"][:].item(0).decode("utf-8")

                    names.append(name)
                    labels.append(label)
                    TinyData.imwrite(f"{source_dir}/valid/{name}.png", image, color=color)

                valid["name"] = names
                valid["label"] = labels


        train.to_csv(f"{dest_dir}/train.csv", sep=",", index=False)
        test.to_csv(f"{dest_dir}/test.csv", sep=",", index=False)
        valid.to_csv(f"{dest_dir}/valid.csv", sep=",", index=False)
        
        return (train, test, valid)
        


