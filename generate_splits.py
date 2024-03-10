import json

import numpy as np
from sklearn.model_selection import KFold

from config import K_FOLD
from load_data import DATA_SPLITS_DIR, DatasetName, GraphsDataset
from utils import NpEncoder


def generate(dataset_name: DatasetName):
    seed = 12
    path = f"data/{DATA_SPLITS_DIR}/{dataset_name.value}.json"
    size = GraphsDataset(dataset_name).size
    kfold = KFold(n_splits=K_FOLD, shuffle=True, random_state=seed)
    indexes = np.arange(size)

    np.random.seed(seed)
    folds = []
    for train, test in kfold.split(indexes):
        np.random.shuffle(train)
        np.random.shuffle(test)
        folds.append({"train": train, "test": test})
    dumped = json.dumps(folds, cls=NpEncoder)
    with open(path, "w") as f:
        f.write(dumped)


if __name__ == "__main__":
    for dataset_name in DatasetName:
        generate(dataset_name)
