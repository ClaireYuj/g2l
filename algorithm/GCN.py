# Attention
# Attention
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import functional as F
from utils.util import init
import torch_geometric.nn as geo_nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""@Dai Quoc Nguyen"""
"""Graph Transformer with Gated GNN"""
def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)
class GatedGT(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_classes,
                 num_self_att_layers, num_GNN_layers, nhead, dropout, act=torch.relu):
        super(GatedGT, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.emb_encode = nn.Linear(feature_dim_size, hidden_size)
        self.dropout_encode = nn.Dropout(dropout)
        self.gt_layers = torch.nn.ModuleList()
        for _layer in range(self.num_GNN_layers):
            encoder_layers = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size, dropout=0.5)  # Default batch_first=False (seq, batch, feature) for pytorch < 1.9.0
            self.gt_layers.append(TransformerEncoder(encoder_layers, num_self_att_layers))
        self.z0 = nn.Linear(hidden_size, hidden_size)
        self.z1 = nn.Linear(hidden_size, hidden_size)
        self.r0 = nn.Linear(hidden_size, hidden_size)
        self.r1 = nn.Linear(hidden_size, hidden_size)
        self.h0 = nn.Linear(hidden_size, hidden_size)
        self.h1 = nn.Linear(hidden_size, hidden_size)
        self.soft_att = nn.Linear(hidden_size, 1)
        self.ln = nn.Linear(hidden_size, hidden_size)
        self.act = act
        self.prediction = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def gatedGNN(self, x, adj):
        with torch.no_grad():
            adj = self.emb_encode(adj)
            adj = torch.transpose(adj, 0, 1)
            adj = self.emb_encode(adj)
        n_agent, seq, f = x.shape
        x = x.flatten(0, 1)
        a = torch.matmul(x, adj)
        x = x.view(-1, seq, f)
        a = a.view(-1, seq, f)
        # update gate
        z0 = self.z0(a)
        z1 = self.z1(x)
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a) + self.r1(x))
        # update embeddings
        h = self.act(self.h0(a) + self.h1(r * x))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj):
        x = inputs
        x = self.dropout_encode(x)
        x = self.emb_encode(x)
        # x = x * mask
        x = x
        for idx_layer in range(self.num_GNN_layers):
            x = torch.transpose(x, 0, 1)  # (seq, batch, feature) for pytorch transformer, pytorch < 1.9.0
            x = self.gt_layers[idx_layer](x)
            x = torch.transpose(x, 0, 1)  # (batch, seq, feature)
            x = x
            x = self.gatedGNN(x, adj)
            # x = x * mask
            # x = self.gatedGNN(x, adj) * mask
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x))
        x = self.act(self.ln(x))
        x = soft_att * x
        # x = soft_att * x * mask
        graph_embeddings = torch.clamp(x, min=-1e5, max=1e5)  # 将值限制在合理范围

        # graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)
        graph_embeddings = self.dropout(graph_embeddings)
        prediction_scores = self.prediction(graph_embeddings)

        return prediction_scores

"""Graph Transformer with GCN"""
class GraphTransformer(nn.Module):
    def __init__(self, feature_dim_size, num_classes, hidden_size,
                 num_self_att_layers=3, num_GNN_layers=2, nhead=8, dropout=0.1, act=torch.relu):
        super(GraphTransformer, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.emb_encode = nn.Linear(feature_dim_size, hidden_size)
        self.dropout_encode = nn.Dropout(dropout)
        self.gt_layers = torch.nn.ModuleList()
        self.gcn_layers = torch.nn.ModuleList()
        for _layer in range(self.num_GNN_layers):
            # self.gcn_layers.append(GraphConvolution(feature_dim_size, hidden_size, dropout))
            encoder_layers = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*4, dropout=0.5)  # Default batch_first=False (seq, batch, feature) for pytorch < 1.9.0

            self.gcn_layers.append(geo_nn.GCNConv(hidden_size, hidden_size))
            # self.gcn_layers.append(GraphConvolution(hidden_size, hidden_size, dropout))
            self.gt_layers.append(TransformerEncoder(encoder_layers, num_self_att_layers))

        self.soft_att = nn.Linear(hidden_size, 1)
        self.ln = nn.Linear(hidden_size, hidden_size)
        self.act = act
        self.prediction = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, adj): # inputs: (batch, seq, feature)
        x = inputs
        edge_index = adj.indices()
        n, seq, state = inputs.shape
        out = x.flatten(0, 1)
        x = self.emb_encode(out)
        res = x
        for idx_layer in range(self.num_GNN_layers):
            if idx_layer == 0: # the first layer
            # if idx_layer == self.num_GNN_layers - 1: # the first layer
                x = self.gt_layers[idx_layer](x)
                x = self.gcn_layers[idx_layer](x, edge_index)  # 两层--报错
                # x = x + res
                x = nn.functional.leaky_relu(x)
            # x = torch.transpose(x, 0, 1) # (batch, seq, feature)
            else:
                # x = self.gt_layers[idx_layer](x)
                x = x + self.gt_layers[idx_layer](x)
                x = self.gcn_layers[idx_layer](x, edge_index)  # 两层--报错
                x = nn.functional.leaky_relu(x)
            # x = x * mask
            # x = self.gcn_layers[idx_layer](x, adj) * mask
        # soft attention
        # soft_att = torch.sigmoid(self.soft_att(x))
        soft_att = torch.tanh(self.soft_att(x))
        ln_x = self.ln(x)
        # x = self.act(ln_x)
        # x = soft_att * x * mask
        x = nn.functional.leaky_relu(ln_x)
        x = soft_att * x

        # x = x.transpose(0,1)
        x = torch.clamp(x, min=-1e5, max=1e5)  # 将值限制在合理范围
        # graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)
        graph_embeddings = self.dropout(x)
        prediction_scores = self.prediction(graph_embeddings)
        prediction_scores = prediction_scores.view(n, seq, -1)
        # prediction_scores = prediction_scores.unsqueeze(1)
        return prediction_scores
        # return prediction_scores
        # out = prediction_scores.repeat(1, seq, 1)
        # return out

"""New advanced GCN using Residual Connection, following https://openreview.net/pdf?id=wVFkD13GpeX"""
class TextGCN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, num_classes, dropout, act=torch.relu):
        super(TextGCN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.gnnlayers = torch.nn.ModuleList()
        for layer in range(self.num_GNN_layers):
            if layer == 0:
                self.gnnlayers.append(GraphConvolution(feature_dim_size, hidden_size, dropout, act=act))
            else:
                self.gnnlayers.append(GraphConvolution(hidden_size, hidden_size, dropout, act=act))
        self.soft_att = nn.Linear(hidden_size, 1)
        self.ln = nn.Linear(hidden_size, hidden_size)
        self.act = act
        self.prediction = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, adj, mask):
        x = inputs
        for idx_layer in range(self.num_GNN_layers):
            if idx_layer == 0:
                x = self.gnnlayers[idx_layer](x, adj) * mask
            else: # Residual Connection
                x += self.gnnlayers[idx_layer](x, adj) * mask
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x))
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum and max pooling, following https://openreview.net/pdf?id=wVFkD13GpeX
        graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)
        graph_embeddings = self.dropout(graph_embeddings)
        prediction_scores = self.prediction(graph_embeddings)

        return prediction_scores


""" Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = self.dropout(input)
        if x.dim() == 3:
            n, seq, f = x.shape
            x = x.flatten(0, 1).permute(1, 0)
        else:
            x = x.permute(1, 0)
        support = torch.matmul(x, self.weight).permute(1, 0)

        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias
        if input.dim() == 3:
            output = output.view(n, seq, -1)

        return self.act(output)


import scipy.sparse as sp

def construct_e2e_matrix(adj_dim, e_dim):
    adj = torch.zeros((adj_dim, adj_dim), dtype=torch.float32).detach()
    # node with same id is connected, r[1] and priority[1] is connected, e and priority should be i the same dim:e_dim
    # all nodes are connected to the user' loc
    for i in range(e_dim):
        adj[i, i] = 1
        if i > 0:
            adj[i, i-1] = 1
            adj[i-1, i] = 1
        if i < e_dim -1 :
            adj[i, i+1] = 1
            adj[i+1, i] = 1
    # mobi
    # adj[:,-1:] = 1
    # adj[-1:,:] = 1
    edge_index_temp = sp.coo_matrix(adj)

    values = edge_index_temp.data  # 边上对应权重值weight
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
    edge_index_A = torch.LongTensor(indices)  # 我们真正需要的coo形式

    i = torch.LongTensor(indices)  # 转tensor
    v = torch.FloatTensor(values)  # 转tensor
    edge_index = torch.sparse_coo_tensor(i, v, edge_index_temp.shape)
    edge_index = edge_index.coalesce()


    return edge_index

def construct_u2e_matrix(e_dim, u_dim):
    adj = torch.full((u_dim, e_dim), 1, dtype=torch.float32).detach()
    edge_index_temp = sp.coo_matrix(adj)

    values = edge_index_temp.data  # 边上对应权重值weight
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
    edge_index_A = torch.LongTensor(indices)  # 我们真正需要的coo形式

    i = torch.LongTensor(indices)  # 转tensor
    v = torch.FloatTensor(values)  # 转tensor
    edge_index = torch.sparse_coo_tensor(i, v, edge_index_temp.shape)
    edge_index = edge_index.coalesce()
    return edge_index

def construct_follower_matrix(e_dim, u_dim):
    adj = torch.full((u_dim, e_dim), 1, dtype=torch.float32).detach()
    edge_index_temp = sp.coo_matrix(adj)

    values = edge_index_temp.data  # 边上对应权重值weight
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
    edge_index_A = torch.LongTensor(indices)  # 我们真正需要的coo形式

    i = torch.LongTensor(indices)  # 转tensor
    v = torch.FloatTensor(values)  # 转tensor
    edge_index = torch.sparse_coo_tensor(i, v, edge_index_temp.shape)
    edge_index = edge_index.coalesce()
    edge_index = edge_index.indices()
    return edge_index


def normalization_matrix(adj):
    degrees = torch.sum(adj, dim=1)  # calculate degree
    d_inv_sqrt = torch.pow(degrees, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, seq_size, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        attn_pdrop = 0.1
        resid_pdrop = 0.1
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(seq_size + 1, seq_size + 1))
                             .view(1, 1, seq_size + 1, seq_size + 1))

        self.att_bp = None

    def forward(self, x):
        B, L, D = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs) L:num_agents
        q = self.query(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # @==torch.matmul

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs) ## @==torch.matmul
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        # y = self.resid_drop(self.proj(y))
        y = self.proj(y)
        return y
