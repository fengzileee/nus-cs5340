"""A script to generate the train-test data for different datasets.

Integer suffix correspondence:
0: eth-univ
1: eth-hotel
2: ucy-zara1
3: ucy-zara2
4: ucy-univ3
"""
from typing import Tuple
import os
import numpy as np
import pickle
from pathlib import Path

from uncertainty_motion_prediction.dataloader import Dataloader
from toolkit.core.trajdataset import TrajDataset, merge_datasets


def split_train_test(
    dataset: TrajDataset, ratio: float = 0.7
) -> Tuple[TrajDataset, TrajDataset]:
    """Split a dataset into train and test.

    Args:
        dataset: A TrajDataset object.
        ratio: Ratio of the training data.
    """
    df = dataset.data.copy(True)
    df["unique"] = df["agent_id"].astype(str) + df["scene_id"]
    test = df.copy(True)

    label = df["unique"].unique()
    mask = np.random.binomial(1, ratio, len(label))
    train_label = []
    test_label = []
    for i in range(len(label)):
        if mask[i] == 1:
            train_label.append(label[i])
        else:
            test_label.append(label[i])

    df = df[df.unique.isin(train_label)]
    df = df.drop(columns=["unique"])

    dataset_train = TrajDataset()
    dataset_train.data = df
    dataset_train.title = "train" + dataset.title

    test = test[test.unique.isin(test_label)]
    test = test.drop(columns=["unique"])

    dataset_test = TrajDataset()
    dataset_test.data = test
    dataset_test.title = "test" + dataset.title
    return dataset_train, dataset_test


def main(data_dir: str = "./data"):
    data_dir = Path(data_dir).resolve().expanduser()
    if data_dir.exists():
        raise RuntimeError("Data dir exists. You might have already generated data.")
    data_dir.mkdir()
    dataloader = Dataloader()
    train_datasets = []
    test_datasets = []
    for i in range(5):
        dataset = dataloader.load(i)
        train, test = split_train_test(dataset)
        train_datasets.append(train)
        test_datasets.append(test)
        with open(data_dir / f"train_{i}.pickle", "wb") as _file:
            pickle.dump(train, _file)
        with open(data_dir / f"test_{i}.pickle", "wb") as _file:
            pickle.dump(test, _file)
    train_all = merge_datasets(train_datasets)
    test_all = merge_datasets(test_datasets)
    with open(data_dir / "train_all.pickle", "wb") as _file:
        pickle.dump(train_all, _file)
    with open(data_dir / "test_all.pickle", "wb") as _file:
        pickle.dump(test_all, _file)


if __name__ == "__main__":
    main()
