import json
import os
import time
from enum import Enum
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import torch
from dgl.data import TUDataset
from ogb.graphproppred import DglGraphPropPredDataset
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

DATASETS_DIR = Path("datasets")
DATA_SPLITS_DIR = Path("data_splits")


class DatasetName(Enum):
    DD = "DD"
    NCI1 = "NCI1"
    PROTEINS = "PROTEINS_full"
    ENZYMES = "ENZYMES"
    IMDB_BINARY = "IMDB-BINARY"
    IMDB_MULTI = "IMDB-MULTI"
    REDDIT_BINARY = "REDDIT-BINARY"
    REDDIT_MULTI = "REDDIT-MULTI-5K"
    COLLAB = "COLLAB"
    MOLHIV = "ogbg-molhiv"


def load_indexes(dataset_name: DatasetName):
    path = f"data/{DATA_SPLITS_DIR}/{dataset_name.value}.json"
    if not os.path.exists(path):
        from generate_splits import generate

        generate(dataset_name)
    with open(path, "r") as f:
        indexes = json.load(f)
    return indexes


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, split, graphs, labels):
        self.split = split

        self.graph_lists = list(graphs)
        self.graph_labels = torch.tensor(list(labels)).float()
        self.n_samples = len(graphs)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get the idx^th sample.
        Parameters
        ---------
        idx : int
            The sample index.
        Returns
        -------
        (dgl.DGLGraph, int)
            DGLGraph with node feature stored in `feat` field
            And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class GraphsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name):
        self.name = dataset_name.value
        start = time.time()
        print("Loading dataset %s..." % (self.name))
        prefix = "/net/tscratch/people/plgglegeza"
        prefix = "."
        data_dir = f"{prefix}/data/{DATASETS_DIR}/{self.name}/"
        if self.name.startswith("ogbg-"):
            self.dgl_dataset = DglGraphPropPredDataset(name=self.name, root=data_dir)
            self.num_classes = int(self.dgl_dataset.num_classes)
            self.size = len(self.dgl_dataset)
            self.graphs = self.dgl_dataset.graphs
            self.labels = [int(label) for label in self.dgl_dataset.labels]
            self.max_num_node = max([g.num_nodes() for g in self.graphs])
            self.num_node_type = get_atom_feature_dims()
            self.num_edge_type = get_bond_feature_dims()
        else:
            self.dgl_dataset = TUDataset(self.name, raw_dir=data_dir)
            self.num_classes = self.dgl_dataset.num_labels
            self.size = len(self.dgl_dataset)

            # updated in _load_graphs
            self.max_num_node = 0
            self.num_edge_type = 1
            self.num_node_type = 1

            self.graphs, self.labels = self._load_graphs()
            self.num_edge_type = int(self.num_edge_type)
            self.num_node_type = int(self.num_node_type)

        self.train = None
        self.val = None
        self.test = None

        print("Dataset size: ", len(self.graphs))
        print("Finished loading.")
        print("Data load time: {:.4f}s".format(time.time() - start))

    def _load_graphs(self):
        graphs = []
        labels = []
        for idx in range(self.size):
            g, lab = self.dgl_dataset[idx]
            self.max_num_node = max(self.max_num_node, g.num_nodes())
            node_labels = g.ndata.get("node_labels")
            g.ndata["feat"] = (
                torch.zeros(g.num_nodes(), dtype=torch.long)
                if node_labels is None
                else node_labels.reshape(-1).long()
            )
            self.num_node_type = max(
                self.num_node_type, max(g.ndata["feat"].numpy()) + 1
            )
            edge_labels = g.edata.get("edge_labels")
            g.edata["feat"] = (
                torch.zeros(g.num_edges(), dtype=torch.long)
                if edge_labels is None
                else edge_labels.reshape(-1).long()
            )
            self.num_edge_type = max(
                self.num_edge_type, max(g.edata["feat"].numpy()) + 1
            )
            graphs.append(g)
            labels.append(int(lab))
        return graphs, labels

    def upload_indexes(self, train_idx, val_idx, test_idx):
        train_graphs = [self.graphs[ix] for ix in train_idx]
        train_labels = [self.labels[ix] for ix in train_idx]
        self.train = SplitDataset("train", train_graphs, train_labels)

        val_graphs = [self.graphs[ix] for ix in val_idx]
        val_labels = [self.labels[ix] for ix in val_idx]
        self.val = SplitDataset("val", val_graphs, val_labels)

        test_graphs = [self.graphs[ix] for ix in test_idx]
        test_labels = [self.labels[ix] for ix in test_idx]
        self.test = SplitDataset("test", test_graphs, test_labels)

        print("Loaded indexes of the dataset")
        print(
            "train, test, val sizes :", len(self.train), len(self.test), len(self.val)
        )

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels
