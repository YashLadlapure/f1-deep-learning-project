import torch
import torch.nn as nn


class GraphConvLayer(nn.Module):
    """
    One graph convolution layer.
    Each driver node aggregates info from neighboring drivers.
    Node  = one driver in the race
    Edge  = two drivers within N positions of each other on track
    """
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.self_fc     = nn.Linear(in_features, out_features)
        self.neighbor_fc = nn.Linear(in_features, out_features)
        self.act         = nn.ReLU()

    def forward(self, x, adj):
        # Self transformation
        self_out = self.self_fc(x)
        # Normalized neighbor aggregation
        deg          = adj.sum(dim=1, keepdim=True).clamp(min=1)
        neighbor_out = self.neighbor_fc((adj @ x) / deg)
        return self.act(self_out + neighbor_out)


class RaceGNN(nn.Module):
    """
    GNN that models the F1 race as a graph.
    Node  = one driver
    Edge  = two drivers are within threshold positions of each other
    Input:  node_features (num_drivers, in_features), adj (num_drivers, num_drivers)
    Output: predicted lap time per driver (num_drivers,)
    """
    def __init__(self, in_features=11, hidden=64, num_layers=3):
        super(RaceGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConvLayer(in_features, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConvLayer(hidden, hidden))
        self.dropout = nn.Dropout(0.3)
        self.bn      = nn.BatchNorm1d(hidden)
        self.out     = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        x = self.bn(x)
        return self.out(x).squeeze(1)


def build_adjacency(positions, threshold=5):
    """
    Builds adjacency matrix for one lap of the race.
    Connects drivers within `threshold` positions of each other.
    positions : list of current race positions for each driver
    Returns   : torch.FloatTensor of shape (num_drivers, num_drivers)
    """
    n   = len(positions)
    adj = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i != j and abs(positions[i] - positions[j]) <= threshold:
                adj[i][j] = 1.0
    return adj
