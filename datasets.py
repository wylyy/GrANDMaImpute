import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def data_process():

    path="./MIMIC_data/"

    y_train = pd.read_csv(str(path) + "MIMIC_y_train.csv", index_col=0)
    y_test = pd.read_csv(str(path) + "MIMIC_y_test.csv", index_col=0)


    x_train_z = pd.read_csv(str(path) + "MIMIC_x_train.csv",
                          index_col=0)

    x_test_z = pd.read_csv(str(path) + "MIMIC_x_test.csv",
                           index_col=0)
    train_mask_ID = mask_ID(x_train_z)
    test_mask_ID = mask_ID(x_test_z)

    train_dataset=pd.concat([x_train_z,y_train],axis=1)

    test_dataset = pd.concat([x_test_z, y_test], axis=1)


    return train_dataset,test_dataset,train_mask_ID ,test_mask_ID
