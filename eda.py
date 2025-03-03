"""
EDA for Mineral Classifier

Author: Ivan Klevanski

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import pickle
import sklearn.decomposition as skl_d

abs_path = os.path.dirname(os.path.abspath(__file__))

dataset_path = "D:\\Users\\ibarn\\Documents\\Dataset Repository\\image\\mineralimage5k"
images_path = os.path.join(dataset_path, "mineral_images")


def main():
    df = pd.read_csv(os.path.join(dataset_path, "minerals_full.csv"))

    n_classes = df["en_name"].value_counts()

    for idx, val in n_classes.items():
        if val > 100:
            print(f"{idx}: {val}")

    pass


if __name__ == "__main__":

    main()