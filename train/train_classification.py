"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def evaluate(scores, targets):
    predictions = scores.argmax(dim=1)
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average="micro")
    recall = recall_score(targets, predictions, average="micro")
    f1 = f1_score(targets, predictions, average="micro")
    macro_f1 = f1_score(targets, predictions, average="macro")
    probs = F.softmax(scores, dim=1)
    # if scores.shape[1] == 2:
    #     probs = probs[:, 1]
    # roc = roc_auc_score(targets, probs, average="macro", multi_class="ovr")
    unique_values = np.unique(targets)
    if len(unique_values) == scores.shape[1]:
        if scores.shape[1] == 2:
            probs = probs[:, 1]
        roc = roc_auc_score(targets, probs, average="macro", multi_class="ovr")
    else:
        roc = 0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro f1": macro_f1,
        "roc": roc,
    }


def train_epoch(model, optimizer, device, data_loader, epoch, LPE, batch_accumulation):
    model.train()

    epoch_loss = 0

    targets=torch.tensor([])
    scores=torch.tensor([])

    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        # print(iter, torch.cuda.memory_allocated(0))
        batch_graphs = batch_graphs.to(device=device)
        batch_x = batch_graphs.ndata['feat']
        batch_e = batch_graphs.edata['feat']

        batch_targets = batch_targets.to(device)

        if LPE == 'node':

            batch_EigVecs = batch_graphs.ndata['EigVecs']
            #random sign flipping
            sign_flip = torch.rand(batch_EigVecs.size(1), device=device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_EigVecs = batch_EigVecs * sign_flip.unsqueeze(0)

            batch_EigVals = batch_graphs.ndata['EigVals']
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_EigVecs, batch_EigVals)

        elif LPE == 'edge':

            batch_diff = batch_graphs.edata['diff']
            batch_prod = batch_graphs.edata['product']
            batch_EigVals = batch_graphs.edata['EigVals']
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_diff, batch_prod, batch_EigVals)

        else:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)

        loss = model.loss(batch_scores, batch_targets)
        loss = loss / batch_accumulation
        loss.backward()

        # weights update
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(data_loader)):
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.detach().item()

        targets = torch.cat((targets, batch_targets.detach().cpu()), 0)
        scores = torch.cat((scores, batch_scores.detach().cpu()), 0)

    epoch_train_score = evaluate(scores, targets)["accuracy"]

    epoch_loss /= (iter + 1)

    return epoch_loss, epoch_train_score, optimizer

def evaluate_network(model, device, data_loader, epoch, LPE, all_metrics=False):
    model.eval()

    epoch_test_loss = 0

    targets=torch.tensor([])
    scores=torch.tensor([])

    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device=device)
            batch_x = batch_graphs.ndata['feat']
            batch_e = batch_graphs.edata['feat']
            batch_targets = batch_targets.to(device=device)

            if LPE == 'node':
                batch_EigVecs = batch_graphs.ndata['EigVecs']
                batch_EigVals = batch_graphs.ndata['EigVals']
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_EigVecs, batch_EigVals)

            elif LPE == 'edge':
                batch_diff = batch_graphs.edata['diff']
                batch_prod = batch_graphs.edata['product']
                batch_EigVals = batch_graphs.edata['EigVals']
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_diff, batch_prod, batch_EigVals)

            else:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)

            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()

            targets = torch.cat((targets, batch_targets.detach().cpu()), 0)
            scores = torch.cat((scores, batch_scores.detach().cpu()), 0)

    evaluation = evaluate(scores, targets)
    epoch_test_score = evaluation["accuracy"]

    epoch_test_loss /= (iter + 1)
    
    if all_metrics:
        return evaluation

    return epoch_test_loss, epoch_test_score
