#  MIT License
#
#  Copyright (c) 2019 Geom-GCN Authors
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import argparse
import json
import os
import time

import dgl.init
import numpy as np
import torch as th
import torch.nn.functional as F

import utils_data
from utils_layers import GeomGCNNet
from init_layers import init_layers

from ray import tune
import wandb

def pipe(config:dict):
    exp = config['exp']
    name = config['dataset']+'/'+config['init']
    wandb.init(project=f"exp{exp}", config=config, dir='/mnt/jiahanli/wandb', name=name)

    par_path = '/mnt/jiahanli/datasets/geom-gcn'
    config['dataset_split'] = f"{par_path}/splits/{config['dataset']}_split_0.6_0.2_{config['dataset_split']}.npz"
    args = argparse.Namespace(**config)
    vars(args)['model'] = 'GeomGCN_TwoLayers'

    t1 = time.time()
    print("--- load datasets ---")
    if args.dataset_split == 'jknet':
        g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels = utils_data.load_data(
            args.dataset, None, 0.6, 0.2, 'GeomGCN', args.dataset_embedding)
    else:
        g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels = utils_data.load_data(
            args.dataset, args.dataset_split, None, None, 'GeomGCN', args.dataset_embedding)
    print(f"period of loading dataset {time.time() - t1}")

    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    g = g.to('cuda')

    print("--- load models ---")
    net = GeomGCNNet(g=g, num_input_features=num_features, num_output_classes=num_labels, num_hidden=args.num_hidden,
                     num_divisions=9, dropout_rate=args.dropout_rate,
                     num_heads_layer_one=args.num_heads_layer_one, num_heads_layer_two=args.num_heads_layer_two,
                     layer_one_ggcn_merge=args.layer_one_ggcn_merge,
                     layer_one_channel_merge=args.layer_one_channel_merge,
                     layer_two_ggcn_merge=args.layer_two_ggcn_merge,
                     layer_two_channel_merge=args.layer_two_channel_merge)

    optimizer = th.optim.Adam([{'params': net.geomgcn1.parameters(), 'weight_decay': args.weight_decay_layer_one},
                               {'params': net.geomgcn2.parameters(), 'weight_decay': args.weight_decay_layer_two}],
                              lr=args.learning_rate)
    learning_rate_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                      factor=args.learning_rate_decay_factor,
                                                                      patience=args.learning_rate_decay_patience)
    
    net.cuda()
    features = features.cuda()
    labels = labels.cuda()
    train_mask = train_mask.cuda()
    val_mask = val_mask.cuda()
    test_mask = test_mask.cuda()

    # initialize layer with nim
    print('--- initialize layers ---')
    a_list = init_layers(g, features, net, args.init)

    # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
    patience = args.num_epochs_patience
    vlss_mn = np.inf
    vacc_mx = 0.0
    state_dict_early_model = None
    curr_step = 0

    # Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
    dur = []
    train_curve, valid_curve = [], []
    print("--- start training ---")
    for epoch in range(args.num_epochs_max):
        t0 = time.time()

        net.train()
        train_logits = net(features)
        train_logp = F.log_softmax(train_logits, 1)
        train_loss = F.nll_loss(train_logp[train_mask], labels[train_mask])
        train_pred = train_logp.argmax(dim=1)
        train_acc = th.eq(train_pred[train_mask], labels[train_mask]).float().mean().item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        net.eval()
        with th.no_grad():
            val_logits = net(features)
            val_logp = F.log_softmax(val_logits, 1)
            val_loss = F.nll_loss(val_logp[val_mask], labels[val_mask]).item()
            val_pred = val_logp.argmax(dim=1)
            val_acc = th.eq(val_pred[val_mask], labels[val_mask]).float().mean().item()

        train_curve.append(train_acc)
        valid_curve.append(val_acc)

        wandb.log({'train_curve': train_acc})
        wandb.log({'valid_curve': val_acc})

        learning_rate_scheduler.step(val_loss)

        dur.append(time.time() - t0)

        print(
            "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f} | Patience {}/{}".format(
                epoch, train_loss.item(), train_acc, val_loss, val_acc, sum(dur) / len(dur), curr_step, patience))


        # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
        if val_acc >= vacc_mx or val_loss <= vlss_mn:
            if val_acc >= vacc_mx and val_loss <= vlss_mn:
                state_dict_early_model = net.state_dict()
            vacc_mx = np.max((val_acc, vacc_mx))
            vlss_mn = np.min((val_loss, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= patience:
                break

    net.load_state_dict(state_dict_early_model)
    net.eval()
    with th.no_grad():
        test_logits = net(features)
        test_logp = F.log_softmax(test_logits, 1)
        test_loss = F.nll_loss(test_logp[test_mask], labels[test_mask]).item()
        test_pred = test_logp.argmax(dim=1)
        test_acc = th.eq(test_pred[test_mask], labels[test_mask]).float().mean().item()

    results_dict = vars(args)
    results_dict['test_loss'] = test_loss
    results_dict['test_acc'] = test_acc
    results_dict['actual_epochs'] = 1 + epoch
    results_dict['val_acc_max'] = vacc_mx
    results_dict['val_loss_min'] = vlss_mn
    results_dict['total_time'] = sum(dur)

    print(f"Test acc: {test_acc}")
    wandb.log({'test_acc': test_acc})
    wandb.finish()
    return train_curve, valid_curve, test_acc

def tune_pipe(config):
    train_curve, valid_curve, test_acc = pipe(config)
    tune.report(train_curve=train_curve, valid_curve=valid_curve, test_acc=test_acc)

def run_ray():
    exp = 68
    num_samples = 1
    searchSpace = {
        'dataset': tune.grid_search(['wisconsin', 'flim']),
        'dataset_embedding': 'poincare',
        'num_hidden': tune.grid_search([48, 128]),
        'num_heads_layer_one': 1,
        'num_heads_layer_two': 1,
        'layer_one_ggcn_merge': 'cat',
        'layer_two_ggcn_merge': 'mean',
        'layer_one_channel_merge': 'cat',
        'layer_two_channel_merge': 'mean',
        'dropout_rate': tune.grid_search([0.0, 0.5]),
        'learning_rate': tune.grid_search([5e-2, 1e-2]),
        'weight_decay_layer_one': tune.grid_search([5e-6, 1e-6]),
        'weight_decay_layer_two': tune.grid_search([5e-6, 1e-6]),
        'num_epochs_patience': 100,
        'num_epochs_max': 5000,
        'dataset_split': tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        'learning_rate_decay_patience': 50,
        'learning_rate_decay_factor': 0.8,
        'init': tune.grid_search(['nimfor', 'nimback']),
        'exp': exp
    }
    
    print(searchSpace)

    analysis=tune.run(tune_pipe, config=searchSpace, name=f"{exp}", num_samples=num_samples, \
        resources_per_trial={'cpu': 12, 'gpu':1}, log_to_file=f"out.log", \
        local_dir="/mnt/jiahanli/nim_output", max_failures=1)

def run_test():
    searchSpace = {
        'dataset': 'cornell',
        'dataset_embedding': 'poincare',
        'num_hidden': 48,
        'num_heads_layer_one': 1,
        'num_heads_layer_two': 1,
        'layer_one_ggcn_merge': 'cat',
        'layer_two_ggcn_merge': 'mean',
        'layer_one_channel_merge': 'cat',
        'layer_two_channel_merge': 'mean',
        'dropout_rate': 5e-1,
        'learning_rate': 5e-2,
        'weight_decay_layer_one': 5e-6,
        'weight_decay_layer_two': 5e-6,
        'num_epochs_patience': 100,
        'num_epochs_max': 5000,
        'dataset_split': 5,
        'learning_rate_decay_patience': 50,
        'learning_rate_decay_factor': 0.8,
        'init': 'nimfor',
        'exp': 500
    }
    print(searchSpace)
    pipe(searchSpace)

if __name__ == "__main__":
    run='ray'
    
    if run == 'test':
        run_test()
    elif run == 'ray':
        run_ray()
    

    print(1)