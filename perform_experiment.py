import json
import os
import time

import numpy as np
# import optuna
# from optuna.trial import TrialState
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from config import K_FOLD, R_EVALUATION
from load_data import DatasetName, GraphsDataset, load_indexes
from utils import NpEncoder
from main_graph_classification import main
import argparse
from time_measure import time_measure
from torch.utils.data import DataLoader
import torch



experiment_name = time.strftime("%Y_%m_%d_%Hh%Mm%Ss")

special_params = {
    DatasetName.DD: {
        "params": {
            "epochs": 100,
            "batch_size": 64,
        },
        "net_params": {
            "full_graph": False,
        },
    },
    DatasetName.NCI1: {
        "params": {
            "epochs": 100,
            "batch_size": 64,
        },
        "net_params": {
            "full_graph": False,
        },
    },
    DatasetName.ENZYMES: {
        "params": {
            "epochs": 100,
            "batch_size": 64,
        },
        "net_params": {
            "full_graph": False,
        },
    },
    DatasetName.PROTEINS: {
        "params": {
            "epochs": 100,
            "batch_size": 64,
        },
        "net_params": {
            "full_graph": False,
        },
    },
    DatasetName.IMDB_BINARY: {
        "params": {
            "epochs": 100,
            "batch_size": 64,
        },
        "net_params": {
            "full_graph": False,
        },
    },
    DatasetName.IMDB_MULTI: {
        "params": {
            "epochs": 100,
            "batch_size": 64,
        },
        "net_params": {
            "full_graph": False,
        },
    },
    DatasetName.REDDIT_BINARY: {
        "params": {
            "epochs": 100,
            "batch_size": 32,
        },
        "net_params": {
            "full_graph": False,
        },
    },
    DatasetName.REDDIT_MULTI: {
        "params": {
            "epochs": 100,
            "batch_size": 32,
        },
        "net_params": {
            "full_graph": False,
        },
    },
    DatasetName.COLLAB: {
        "params": {
            "epochs": 100,
            "batch_size": 32,
        },
        "net_params": {
            "full_graph": False,
        },
    },
    DatasetName.MOLHIV: {
        "params": {
            "epochs": 100,
            "batch_size": 128,
        },
        "net_params": {
            "full_graph": False,
        },
    },
    DatasetName.WEB: {
        "params": {
            "epochs": 100,
            "batch_size": 128,
        },
        "net_params": {
            "full_graph": False,
        },
    },
    DatasetName.MUTAGEN: {
        "params": {
            "epochs": 100,
            "batch_size": 128,
        },
        "net_params": {
            "full_graph": False,
        },
    },

}


def get_prediction(model, device, data_loader):
    model.eval()
    list_predictions = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata["feat"].to(device)
            batch_e = batch_graphs.edata["feat"].to(device)
            batch_targets = batch_targets.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata["lap_pos_enc"].to(device)
            except:
                batch_lap_pos_enc = None

            try:
                batch_wl_pos_enc = batch_graphs.ndata["wl_pos_enc"].to(device)
            except:
                batch_wl_pos_enc = None

            batch_scores = model.forward(
                batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc
            )
            list_predictions.append(batch_scores.detach().argmax(dim=1).cpu().numpy())
    predictions = np.concatenate(list_predictions)
    return predictions


def train_graph_transformer(dataset, train_config):
    return main(dataset, train_config)

def find_best_params(train_config, loaded_dataset, fold):
    return None

# def find_best_params(train_config, loaded_dataset, fold):
#     def optuna_objective(trial):
#         ### Definition of the search space ###
#         train_config["learning_rate"] = trial.suggest_float(
#             "learning_rate", 1e-6, 1e-3, log=True
#         )
#         train_config["dropout"] = trial.suggest_float("dropout", 0.0, 1.0)
#         train_config["layers_number"] = trial.suggest_int("layers_number", 5, 20)
#         ### End ###
#         train_config["trial"] = trial

#         dataset = loaded_dataset

#         acc = train_graph_transformer(dataset, train_config)["accuracy"]
#         return acc

#     train_idx, val_idx = train_test_split(fold["train"], test_size=0.1)
#     loaded_dataset.upload_indexes(train_idx, val_idx, val_idx)

#     study = optuna.create_study(direction="maximize")
#     study.optimize(optuna_objective, n_trials=3, timeout=None)
#     train_config["trial"] = None

#     pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
#     complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

#     print("Study statistics: ")
#     print("  Number of finished trials: ", len(study.trials))
#     print("  Number of pruned trials: ", len(pruned_trials))
#     print("  Number of complete trials: ", len(complete_trials))

#     print("Best trial:")
#     trial = study.best_trial

#     print("  Value: ", trial.value)

#     best_params = {}
#     for key, value in trial.params.items():
#         best_params[key] = value

#     return best_params, trial.value

def prepare_dataset(dataset, train_config):
    if train_config["net_params"]['LPE'] in ['edge', 'node']:
        # st = time.time()
        # print("[!] Computing Laplace Decompositions..")
        dataset._laplace_decomp(train_config["net_params"]['m'])
        # print('Time LapPE:',time.time()-st)

    if train_config["net_params"]['full_graph']:
        # st = time.time()
        # print("[!] Adding full graph connectivity..")
        dataset._make_full_graph()
        # print('Time taken to convert to full graphs:',time.time()-st)

    if train_config["net_params"]['LPE'] == 'edge':
        # st = time.time()
        # print("[!] Computing edge Laplace features..")
        dataset._add_edge_laplace_feats()
        # print('Time taken to compute edge Laplace features: ',time.time()-st)


def perform_experiment(dataset_name):
    config = {
        "tune_hyperparameters": False,
        "train_config_path": "train_config.json",
    }

    with open(config["train_config_path"], "r") as f:
        train_config = json.load(f)
    train_config["params"].update(special_params[dataset_name]["params"])
    train_config["net_params"].update(special_params[dataset_name]["net_params"])
    train_config["out_dir"] = f"out/{dataset_name.value}/"

    # load indexes
    indexes = load_indexes(dataset_name)
    assert len(indexes) == K_FOLD, "Re-generate splits for new K_FOLD."

    dataset = GraphsDataset(dataset_name)
    # add to config info about dataset
    train_config["net_params"]["num_classes"] = dataset.num_classes
    train_config["net_params"]["num_node_type"] = dataset.num_node_type
    train_config["net_params"]["num_edge_type"] = dataset.num_edge_type

    # prepare_dataset(dataset, train_config)
    time_measure(prepare_dataset, "san", dataset_name, "preparation")(
        dataset, train_config
    )

    for i, fold in enumerate(indexes):
        print(f"FOLD {i}")

        test_idx = fold["test"]
        train_idx, val_idx = train_test_split(fold["train"], test_size=0.1)
        dataset.upload_indexes(train_idx, val_idx, test_idx)
        model, device = time_measure(
            train_graph_transformer, "san", dataset_name, "training"
        )(dataset, train_config)
        
        eval_idx = list(range(128))
        dataset.upload_indexes(eval_idx, eval_idx, eval_idx)
        eval_loader = DataLoader(
            dataset.test,
            batch_size=train_config["params"]["batch_size"],
            shuffle=False,
            collate_fn=dataset.collate,
        )

        predictions = time_measure(get_prediction, "san", dataset_name, "evaluation")(
            model, device, eval_loader
        )

        break
        
        
    del dataset
    # evaluate model

if __name__ == "__main__":
    for dataset_name in DatasetName:
        if dataset_name == DatasetName.WEB:
            continue
        print(f"Performing experiment for {dataset_name}")
        perform_experiment(dataset_name)
