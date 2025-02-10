import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphModel(nn.Module):
    def __init__(self, sample_num=5, depth=2, dims=20, gcn=True, concat=True,
                 dims_first_agg=[20], dims_second_agg=[20], num_classes=20):
        super(GraphModel, self).__init__()
        self.sample_num = sample_num
        self.depth = depth
        self.dims = dims
        self.gcn = gcn
        self.concat = concat

        # 初始化一阶邻居聚合层
        self.dense_layers_first_agg = nn.ModuleList([
            nn.Linear(dims, dim) for dim in dims_first_agg
        ])

        # 如果 depth == 2，则初始化二阶邻居聚合层
        if depth == 2:
            self.dense_layers_second_agg = nn.ModuleList([
                nn.Linear(dims, dim) for dim in dims_second_agg
            ])

    def aggregator(self, node_features, neigh_features, dense_layers, concat=True):
        """
        邻居聚合函数
        """
        if concat:
            node_embed = node_features.unsqueeze(1)  # 增加维度
            to_feats = torch.cat([neigh_features, node_embed], dim=1)
        else:
            to_feats = neigh_features

        for layer in dense_layers:
            to_feats = F.relu(layer(to_feats))

        combined = torch.mean(to_feats, dim=1)  # 在邻居维度上取均值
        return combined

    def forward(self, features, batch_nodes, s1_neighs, s2_neighs, s1_weights, s2_weights):
        s1_weights = torch.tensor(s1_weights, dtype=torch.float32)
        s2_weights = torch.tensor(s2_weights, dtype=torch.float32)

        if self.depth == 1:
            node_fea = torch.index_select(features, 0, batch_nodes)
            neigh_1_fea = torch.index_select(features, 0, s1_neighs)
            weights_expanded = s1_weights.unsqueeze(-1)  # 扩展维度
            weighted_features = torch.mul(neigh_1_fea, weights_expanded)
            agg_result = self.aggregator(node_fea, weighted_features, self.dense_layers_first_agg, self.concat)

        else:
            node_fea = torch.index_select(features, 0, batch_nodes)
            neigh_1_fea = torch.index_select(features, 0, s1_neighs)
            weights_expanded = s1_weights.unsqueeze(-1)
            weighted_features_s1 = torch.mul(neigh_1_fea, weights_expanded)

            agg_node = self.aggregator(node_fea, weighted_features_s1, self.dense_layers_first_agg, self.concat)

            neigh_2_fea = torch.index_select(features, 0, s2_neighs)
            weights_expanded = s2_weights.unsqueeze(-1)
            weighted_features_s2 = torch.mul(neigh_2_fea, weights_expanded)

            agg_neigh1 = self.aggregator(weighted_features_s1, weighted_features_s2, self.dense_layers_first_agg,
                                         self.concat)
            agg_result = self.aggregator(agg_node, agg_neigh1, self.dense_layers_second_agg, self.concat)

        return agg_result

    def forward_without_weights(self, features, batch_nodes, s1_neighs, s2_neighs):
        if self.depth == 1:
            node_fea = torch.index_select(features, 0, batch_nodes)
            neigh_1_fea = torch.index_select(features, 0, s1_neighs)
            agg_result = self.aggregator(node_fea, neigh_1_fea, self.dense_layers_first_agg, self.concat)
        else:
            node_fea = torch.index_select(features, 0, batch_nodes)
            neigh_1_fea = torch.index_select(features, 0, s1_neighs)
            agg_node = self.aggregator(node_fea, neigh_1_fea, self.dense_layers_first_agg, self.concat)

            neigh_2_fea = torch.index_select(features, 0, s2_neighs)
            agg_neigh1 = self.aggregator(neigh_1_fea, neigh_2_fea, self.dense_layers_first_agg, self.concat)
            agg_result = self.aggregator(agg_node, agg_neigh1, self.dense_layers_second_agg, self.concat)

        return agg_result
