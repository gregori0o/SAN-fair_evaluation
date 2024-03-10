import json
import os
import time

import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from config import K_FOLD, R_EVALUATION
from load_data import DatasetName, GraphsDataset, load_indexes
from utils import NpEncoder

experiment_name = time.strftime("%Y_%m_%d_%Hh%Mm%Ss")


def train_graph_transformer(dataset, train_config):
    targets = np.random.randint(0, 2, size=100)

    # trial = train_config.get("trial")

    # train and evaluate model

    # at the end of each epoch:
    # if trial is not None:
    #     trial.report(epoch_val_score, epoch)
    #     if trial.should_prune():
    #         raise optuna.TrialPruned()

    predictions = np.random.randint(0, 2, size=100)
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average="macro")
    recall = recall_score(targets, predictions, average="macro")
    f1 = f1_score(targets, predictions, average="macro")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def find_best_params(train_config, loaded_dataset, fold):
    def optuna_objective(trial):
        ### Definition of the search space ###
        train_config["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-6, 1e-3, log=True
        )
        train_config["dropout"] = trial.suggest_float("dropout", 0.0, 1.0)
        train_config["layers_number"] = trial.suggest_int("layers_number", 5, 20)
        ### End ###
        train_config["trial"] = trial

        dataset = loaded_dataset

        acc = train_graph_transformer(dataset, train_config)["accuracy"]
        return acc

    train_idx, val_idx = train_test_split(fold["train"], test_size=0.1)
    loaded_dataset.upload_indexes(train_idx, val_idx, val_idx)

    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=3, timeout=None)
    train_config["trial"] = None

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    best_params = {}
    for key, value in trial.params.items():
        best_params[key] = value

    return best_params, trial.value


def perform_experiment(dataset_name):
    config = {
        "tune_hyperparameters": True,
        "train_config_path": "train_config.json",
    }

    with open(config["train_config_path"], "r") as f:
        train_config = json.load(f)
    train_config["out_dir"] = f"out/{dataset_name.value}/"

    # load indexes
    indexes = load_indexes(dataset_name)
    assert len(indexes) == K_FOLD, "Re-generate splits for new K_FOLD."

    dataset = GraphsDataset(dataset_name)
    # add to config info about dataset
    train_config["num_classes"] = dataset.num_classes
    train_config["num_node_type"] = dataset.num_node_type
    train_config["num_edge_type"] = dataset.num_edge_type

    # loop over splits
    scores = {
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }
    tuning_result = {}
    for i, fold in enumerate(indexes):
        print(f"FOLD {i}")

        ## get best model for train data
        if config.get("tune_hyperparameters"):
            best_params, best_acc = find_best_params(train_config, dataset, fold)

            train_config.update(best_params)
            tuning_result[i] = {
                "params": best_params,
                "accuracy": best_acc,
            }

        # evaluate model R times
        scores_r = {
            "accuracy": 0,
            "f1": 0,
            "precision": 0,
            "recall": 0,
        }
        test_idx = fold["test"]
        for _ in range(R_EVALUATION):
            train_idx, val_idx = train_test_split(fold["train"], test_size=0.1)
            dataset.upload_indexes(train_idx, val_idx, test_idx)
            scores_class = train_graph_transformer(dataset, train_config)
            for key in scores_r.keys():
                scores_r[key] += scores_class[key]
        for key in scores_r.keys():
            scores_r[key] /= R_EVALUATION
        print(f"MEAN SCORES = {scores_r} in FOLD {i}")
        for key in scores_r.keys():
            scores[key].append(scores_r[key])

    del dataset
    # evaluate model
    summ = {}
    for key in scores.keys():
        summ[key] = {}
        summ[key]["mean"] = np.mean(scores[key])
        summ[key]["std"] = np.std(scores[key])

    # scores are acc, precision, recall and F1
    print(f"Evaluation of model on {dataset_name}")
    print(f"Scores: {scores}")
    print(f"Summary: {summ}")

    train_config["dataset_name"] = dataset_name.value
    train_config["run_config"] = config
    train_config["tune_hyperparameters"] = tuning_result
    train_config["summary_scores"] = summ
    train_config["scores"] = scores
    train_config["r_evaluation"] = R_EVALUATION
    train_config["k_fold"] = K_FOLD
    dumped = json.dumps(train_config, cls=NpEncoder)
    os.makedirs(f"results/{experiment_name}", exist_ok=True)
    with open(
        f"results/{experiment_name}/result_GT_{dataset_name.value}_{time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')}.json",
        "w",
    ) as f:
        f.write(dumped)


if __name__ == "__main__":
    for dataset_name in DatasetName:
        perform_experiment(dataset_name)
