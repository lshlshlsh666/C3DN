import numpy as np
import pandas as pd
import argparse
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser("Interface for CGFTD data preprocessing")
parser.add_argument(
    "--data", type=str, help="Dataset name (eg. reddit)", default="wikipedia"
)

args = parser.parse_args()


def preprocess(dataname):
    if dataname == "reddit":
        df = pd.read_csv(
            f"Data/{dataname}/{dataname}.csv", names=np.arange(176), low_memory=False
        )
        df = df[1:]
        user = df[[0]].values.astype(int)
        item = df[[1]].values.astype(int)
        time = df[2].astype(float)
        time_stamp = pd.cut(time, bins=(len(time) // 10000), labels=False).values.reshape(-1, 1)
        attributor = df.iloc[:, 4:].values.astype(float)
        X = np.hstack([user, attributor, item, time_stamp])
        return X
    elif dataname == "wikipedia":
        df = pd.read_csv(
            f"Data/{dataname}/{dataname}.csv", names=np.arange(176), low_memory=False
        )
        df = df[1:]
        user = df[[0]].values.astype(int)
        item = df[[1]].values.astype(int)
        time = df[2].astype(float)
        time_stamp = pd.cut(time, bins=(len(time) // 3000), labels=False).values.reshape(-1, 1)
        attributor = df.iloc[:, 4:].values.astype(float)
        X = np.hstack([user, attributor, item, time_stamp])
        return X


def run(dataname):
    path = f"Data/{dataname}/pre_processed_{dataname}.npy"
    print(path)
    np.save(path, preprocess(dataname))


if __name__ == "__main__":
    run(args.data)
