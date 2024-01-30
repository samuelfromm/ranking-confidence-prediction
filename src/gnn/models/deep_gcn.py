import torch
from torch_geometric.nn import GINEConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn import GINConv
import torch.nn as nn
from torch_geometric.nn import DeepGCNLayer, GENConv
import torch_geometric

from torch_sparse import SparseTensor


class DeepGCN(torch.nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        model_params,
    ):
        super(DeepGCN, self).__init__()

        num_layers = model_params.get("model_num_layers")
        hidden_channels = model_params.get("model_num_hidden_channels")
        dropout_rate = model_params.get("model_dropout_rate")
        dropout_rate_DeepGCNLayer = model_params.get("model_dropout_rate_DeepGCNLayer")

        # Set default values
        if num_layers is None:
            num_layers = 2
        if hidden_channels is None:
            hidden_channels = 8
        if dropout_rate is None:
            dropout_rate = 0.1
        if dropout_rate_DeepGCNLayer is None:
            dropout_rate_DeepGCNLayer = 0.1

        self.dropout_rate = dropout_rate
        self.node_encoder = nn.Linear(num_node_features, hidden_channels)
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_channels),
        )

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(
                hidden_channels,
                hidden_channels,
                aggr="softmax",
                t=1.0,
                learn_t=True,
                num_layers=2,
                norm="layer",
            )
            norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(
                conv,
                norm,
                act,
                block="res+",
                dropout=dropout_rate_DeepGCNLayer,
                ckpt_grad=i % 3,
            )
            self.layers.append(layer)

        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_channels, 1),
            # nn.ReLU(),
            # nn.Linear(4, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        # Add readout layer
        x = global_max_pool(x, batch)

        # x = self.lin(x)
        x = self.readout_mlp(x)
        x = x.sigmoid()
        return x


### TODO - convert to sparse MLP which takes care of edges
class EdgeEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict["user"][row], z_dict["movie"][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


# MASKING CODE (for feature analysis)
# mask = torch.ones(67).to(device="cuda")
# mask[-3] = 0
# mask = torch.cat((torch.ones(64), torch.zeros(3))).to(device="cuda")
# edge_attr = edge_attr * mask
