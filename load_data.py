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
from scipy import sparse as sp
import torch.nn.functional as F
# from ogb.graphproppred import DglGraphPropPredDataset
# from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

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
    # MOLHIV = "ogbg-molhiv"


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
        # prefix = "."
        data_dir = f"{prefix}/data/{DATASETS_DIR}/{self.name}/"
        if self.name.startswith("ogbg-"):
            pass
            # self.dgl_dataset = DglGraphPropPredDataset(name=self.name, root=data_dir)
            # self.num_classes = int(self.dgl_dataset.num_classes)
            # self.size = len(self.dgl_dataset)
            # self.graphs = self.dgl_dataset.graphs
            # self.labels = [int(label) for label in self.dgl_dataset.labels]
            # self.max_num_node = max([g.num_nodes() for g in self.graphs])
            # self.num_node_type = get_atom_feature_dims()
            # self.num_edge_type = get_bond_feature_dims()
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

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels = torch.stack(labels)

        return batched_graph, labels

    def _laplace_decomp(self, max_freqs):
        self.graphs = [laplace_decomp(graph, max_freqs) for graph in self.graphs]

    def _make_full_graph(self):
        self.graphs = [make_full_graph(graph) for graph in self.graphs]

    def _add_edge_laplace_feats(self):
        self.graphs = [add_edge_laplace_feats(graph) for graph in self.graphs]


def laplace_decomp(g, max_freqs):
    # Laplacian
    n = g.number_of_nodes()
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    sparse_matrix = g.adjacency_matrix()
    A = sp.csr_matrix(
        (sparse_matrix.val, sparse_matrix.csr()[1], sparse_matrix.csr()[0]),
        shape=sparse_matrix.shape,
    ).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L.toarray())
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)

    if n < max_freqs:
        g.ndata['EigVecs'] = F.pad(EigVecs, (0, max_freqs - n), value=float('nan'))
    else:
        g.ndata['EigVecs'] = EigVecs

    # Save eigenvalues and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(
        EigVals))))  # Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative

    if n < max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs - n), value=float('nan')).unsqueeze(0)
    else:
        EigVals = EigVals.unsqueeze(0)

    # Save EigVals node features
    g.ndata['EigVals'] = EigVals.repeat(g.number_of_nodes(), 1).unsqueeze(2)

    return g


def make_full_graph(g):
    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    # Copy over the node feature data and laplace eigvals/eigvecs
    full_g.ndata['feat'] = g.ndata['feat']

    try:
        full_g.ndata['EigVecs'] = g.ndata['EigVecs']
        full_g.ndata['EigVals'] = g.ndata['EigVals']
    except:
        pass

    # Initalize fake edge features w/ 0s
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    full_g.edata['real'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)

    # Copy real edge data over, and identify real edges!
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(
        g.edata['feat'].shape[0], dtype=torch.long)  # This indicates real edges

    return full_g


def add_edge_laplace_feats(g):
    EigVals = g.ndata['EigVals'][0].flatten()

    source, dest = g.find_edges(g.edges(form='eid'))

    # Compute diffusion differences and Green function
    g.edata['diff'] = torch.abs(g.nodes[source].data['EigVecs'] - g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['product'] = torch.mul(g.nodes[source].data['EigVecs'], g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['EigVals'] = EigVals.repeat(g.number_of_edges(), 1).unsqueeze(2)

    # No longer need EigVecs and EigVals stored as node features
    del g.ndata['EigVecs']
    del g.ndata['EigVals']

    return g