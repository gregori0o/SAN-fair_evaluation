"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm


# import optuna

"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.classification.load_net import gnn_model

torch.set_default_dtype(torch.float32)


"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device



"""
    VIEWING ENCODING TYPE AND NUM PARAMS
"""
def view_model_param(LPE, net_params):
    model = gnn_model(LPE, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))

    if LPE == 'edge':
        print('Encoding Type/Total parameters:', 'Edge Laplace Encoding/', total_param)
    elif LPE == 'node':
        print('Encoding Type/Total parameters:', 'Node Laplace Encoding', total_param)
    else:
        print('Encoding Type/Total parameters:', 'None', total_param)
    return total_param


"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, trial):
    t0 = time.time()
    per_epoch_time = []

    DATASET_NAME = dataset.name

    net_params['total_param'] = view_model_param(net_params['LPE'], net_params)

    trainset, valset, testset = dataset.train, dataset.val, dataset.test


    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    model = gnn_model(net_params['LPE'], net_params)
    model = model.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_SCOREs, epoch_val_SCOREs, epoch_test_SCOREs = [], [], []


    from train.train_classification import train_epoch, evaluate_network


    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=params['batch_size'],  shuffle=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'],  shuffle=False, collate_fn=dataset.collate)

    epoch_test_scores = None
    best_val_score = 0

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, epoch_train_score, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, net_params['LPE'], params["batch_accumulation"])

                epoch_val_loss, epoch_val_score = evaluate_network(model, device, val_loader, epoch, net_params['LPE'])
                _, epoch_test_score = evaluate_network(model, device, test_loader, epoch, net_params['LPE'])

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_SCOREs.append(epoch_train_score)
                epoch_val_SCOREs.append(epoch_val_score)

                epoch_test_SCOREs.append(epoch_test_score)

                if epoch_val_score > best_val_score:
                    best_val_score = epoch_val_score
                    epoch_test_scores = evaluate_network(model, device, test_loader, epoch, net_params['LPE'], all_metrics=True)
                    epoch_test_scores["epoch"] = epoch

                if trial is not None:
                    trial.report(epoch_val_score, epoch)
                    # if trial.should_prune():
                    #     raise optuna.TrialPruned()

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_score', epoch_train_score, epoch)
                writer.add_scalar('val/_score', epoch_val_score, epoch)
                writer.add_scalar('test/_score', epoch_test_score, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)


                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_SCORE=epoch_train_score, val_SCORE=epoch_val_score,
                              test_SCORE=epoch_test_score)


                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                # ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                # if not os.path.exists(ckpt_dir):
                #     os.makedirs(ckpt_dir)
                # torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                # files = glob.glob(ckpt_dir + '/*.pkl')
                # for file in files:
                #     epoch_nb = file.split('_')[-1]
                #     epoch_nb = int(epoch_nb.split('.')[0])
                #     if epoch_nb < epoch-1:
                #         os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break

    # except Exception as e: # Sometimes there's out of memory error after many epochs
    #     print('-' * 89)
    #     print(f'Exiting from training early Exception: {e}')

    except KeyboardInterrupt:
        print('-' * 89)
        print(f'Exiting from training keyboard interrupt')


    #Return test and train metrics at best val metric
    index = epoch_val_SCOREs.index(max(epoch_val_SCOREs))

    test_score = epoch_test_SCOREs[index]
    val_score = epoch_val_SCOREs[index]
    train_score = epoch_train_SCOREs[index]

    scores = evaluate_network(model, device, test_loader, epoch, net_params['LPE'], all_metrics=True)

    print("Test SCORE: {:.4f}".format(test_score))
    print("Val SCORE: {:.4f}".format(val_score))
    print("Train SCORE: {:.4f}".format(train_score))
    print("Best epoch index: {:.4f}".format(index))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST SCORE: {:.4f}\nTRAIN SCORE: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  test_score, train_score, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
        
    return model, device
    # return scores, epoch_test_scores


def main(dataset, config):
    # device
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    
    # model, dataset, out_dir
    MODEL_NAME = "GraphTransformer"
    out_dir = config["out_dir"]

    # parameters
    params = config['params']
    # model parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']

    if net_params['LPE'] not in ['node', 'edge', 'none']:
        print('[!] User did not provide a valid input argument for \'LPE\'. Valid inputs are \'node\', \'edge\', and \'none\'.')
        exit()

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + dataset.name + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + dataset.name + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + dataset.name + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + dataset.name + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    return train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, config.get("trial"))
