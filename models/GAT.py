import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm
from torch_geometric.nn import TopKPooling
import torch.nn as nn
from torch.nn import Linear


"""
Appling pyG lib
"""


class GATModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, nheads=1, k=10):
        super(GATModel, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.heads = nheads
        self.k = k

        # self.conv0 = GATConv(node_feature_dim, hidden_dim, heads=nheads)

        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=nheads)
        self.conv2 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)
        self.conv3 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads, concat=False)

        # self.norm0 = LayerNorm(nheads * hidden_dim)
        self.norm1 = LayerNorm(nheads * hidden_dim)
        self.norm2 = LayerNorm(nheads * hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)

        self.lin0 = Linear(k * hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

        self.topk_pool = TopKPooling(hidden_dim, ratio=k)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        # edge_index, _  = dropout_adj(edge_index, p = 0.2, training = self.training)

        # x = self.conv0(x, edge_index)
        # x = self.norm0(x, batch)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)

        # 2. Readout layer
        # x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        x = self.topk_pool(x, edge_index, batch=batch)[0]
        x = x.view(batch[-1] + 1, -1)

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.lin0(x)
        x = F.relu(x)

        x = self.lin1(x)
        x = F.relu(x)

        z = x  # extract last layer features

        x = self.lin(x)

        return x, z


