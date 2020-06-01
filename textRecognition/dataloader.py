
import pandas as pd
import h5py as h5
import os

class DataLoader:
    """A set of functions to load Image datasets in HDF5 format into ImageDataGenerator Keras objects.
    It works along with the TinyData class.
    
    Source:
        https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/data/preproc.py
        https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/data/generator.py
        https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/data/evaluation.py
        https://machinelearnings.co/deep-spelling-9ffef96a24f6#.2c9pu8nlm
    """
    def __init__(self):
        pass

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