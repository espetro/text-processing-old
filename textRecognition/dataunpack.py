
import pandas as pd
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
    def unpack_set(set_name, dest_dir, file, color):
        """Performs the unpacking step for each dataset split (train/test/validation)"""
        result = pd.DataFrame(columns=["name", "label"])
        names, labels = [], []

        os.makedirs(f"{dest_dir}/{set_name}", exist_ok=True)

        for sample_name in file[set_name]:
            data = file[f"{set_name}/{sample_name}"]
            image = data[:]
            label = data.attrs["label"].decode("utf-8")

            names.append(sample_name)
            labels.append(label)
            # at the time of writing, it should follow the folder structure for RIMES and IAM
            DataUnpack.imwrite(f"{dest_dir}/{set_name}/{sample_name}.png", image, color=color)

        result["name"] = names
        result["label"] = labels
        return result

    @staticmethod
    def unpack(input_path, dest_dir, color=False, save_df=False):
        """Deminifies / Unpacks the compressed file into train/test/validation DataFrames.
        Images are saved in .png format.
        """
        with h5.File(input_path, "r") as f:
            train = DataUnpack.unpack_set("train", dest_dir, f, color)
            test = DataUnpack.unpack_set("test", dest_dir, f, color)

            # if there is a valid split stored, then extract it as well
            if f.get("valid", None):
                valid = DataUnpack.unpack_set("valid", dest_dir, f, color)
            else:
                valid = None

        if save_df:
            train.to_csv(f"{dest_dir}/train/train.csv", sep=",", index=False)
            test.to_csv(f"{dest_dir}/test/test.csv", sep=",", index=False)
            valid.to_csv(f"{dest_dir}/valid/valid.csv", sep=",", index=False)

        return train, test, valid
