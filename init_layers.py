import torch as th
import torch.nn as nn
import dgl
import numpy as np

from dgl.nn.pytorch import GraphConv
from torch.nn.init import uniform_

from copy import deepcopy

class Aggr(nn.Module):
    def __init__(self, l):
        super(Aggr, self).__init__()
        self.l = l
        self.conv_layers = nn.ModuleList()
        for _ in range(self.l):
            self.conv_layers.append(GraphConv(1, 1, weight=False, bias=False))

    def forward(self, g: dgl.DGLGraph, h: th.Tensor):
        c5_list = []
        l = self.l
        for i in range(l):
            h = self.conv_layers[i](g, h)
            c5_list.append((h ** 2).sum().item())
        return c5_list

def to_uni_a(var_list):
    a_list = []
    for var in var_list:
        a_list.append(np.sqrt(3 * var))
    return a_list

def to_var(a_list):
    var_list = []
    for a in a_list:
        var_list.append(a**2 / 3)
    return var_list

def gen_c2_for(g:dgl.DGLGraph, feat, m_list):
    '''
    generate c2_for for all layers
    m_list == [in_dim, hid_dim_1, hid_dim_2, ..., out_dim]

    TODO: for now we initialize the 0th layer with kaiming_for for a fair comparison
    '''
    l = len(m_list)
    device = g.device
    c2_list = []
    c2_list.append(2 / m_list[0][0]) # kaiming for at the 0th layer
    aggr = Aggr(l)
    aggr = aggr.to(device)
    aggr_list = aggr(g, feat.mean(dim=1))
    for i in range(1, l):
        c5L_1, c5L = aggr_list[i-1], aggr_list[i]
        c2 = 2 / m_list[i][0] * c5L_1 / c5L
        c2_list.append(c2) # the c2 of the 1, 2, 3, ... layers
    return c2_list

def gen_c2_back(g:dgl.DGLGraph, m_list):
    l = len(m_list)
    m_pos_max = len(m_list)-1
    device = g.device
    c2_list = []
    aggr = Aggr(l)
    aggr = aggr.to(device)
    aggr_list = aggr(g, th.ones(g.num_nodes(), 1).to(g.device))
    c2_list.append(g.num_nodes() / ((m_list[-1][1]-1) * aggr_list[0]))
    for i in range(1, l):
        c5L_1, c5L = aggr_list[i-1], aggr_list[i]
        c2 = 2 / m_list[m_pos_max-i][1] * c5L_1 / c5L
        c2_list.append(c2)
    c2_list.reverse()
    return c2_list

def init_layers(g:dgl.DGLGraph, feat, model, init_name):
    assert init_name in ['nimfor', 'nimback', 'xav']
    m_list = [[model.geomgcn1.attention_heads[0].linear_for_each_division[0].in_features, 
            model.geomgcn1.attention_heads[0].linear_for_each_division[0].out_features],
            [model.geomgcn2.attention_heads[0].linear_for_each_division[0].in_features, 
            model.geomgcn2.attention_heads[0].linear_for_each_division[0].out_features]]

    if init_name == 'nimfor':
        a_list = to_uni_a(gen_c2_for(g, feat, m_list))
    elif init_name == 'nimback':
        a_list = to_uni_a(gen_c2_back(g, m_list))
    else: # the last case is xavier, the default initialization for GCN
        return
    
    # for geom-gcn, there're only two layers
    for layer in model.geomgcn1.attention_heads[0].linear_for_each_division:
        uniform_(layer.weight, a=-a_list[0], b=a_list[0])
    for layer in model.geomgcn2.attention_heads[0].linear_for_each_division:
        uniform_(layer.weight, a=-a_list[1], b=a_list[1])
    return a_list