import torch
import torch.nn as nn


class LSTM_GNN(nn.Module):
    """
    Combined Spatio-Temporal model for F1 lap time prediction.

    LSTM Branch  : Captures temporal patterns (how lap times evolve per driver)
    GNN Branch   : Captures spatial patterns (how drivers influence each other)
    Fusion Layer : Combines both branches for final prediction

    This is the main Deep Learning model of the project.
    """
    def __init__(
        self,
        lstm_input=11,
        lstm_hidden=128,
        lstm_layers=2,
        gnn_input=11,
        gnn_hidden=64,
        gnn_layers=3,
        dropout=0.2
    ):
        super(LSTM_GNN, self).__init__()

        # ── LSTM Branch ──────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = lstm_input,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout
        )
        self.lstm_bn = nn.BatchNorm1d(lstm_hidden)

        # ── GNN Branch ───────────────────────────────────────────
        self.gnn_layers_list = nn.ModuleList()
        self.gnn_self_fcs    = nn.ModuleList()
        self.gnn_nb_fcs      = nn.ModuleList()

        in_f = gnn_input
        for _ in range(gnn_layers):
            self.gnn_self_fcs.append(nn.Linear(in_f, gnn_hidden))
            self.gnn_nb_fcs.append(nn.Linear(in_f, gnn_hidden))
            in_f = gnn_hidden

        self.gnn_act     = nn.ReLU()
        self.gnn_dropout = nn.Dropout(dropout)
        self.gnn_bn      = nn.BatchNorm1d(gnn_hidden)

        # ── Fusion Layer ─────────────────────────────────────────
        fusion_input = lstm_hidden + gnn_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def gnn_forward(self, x, adj):
        """GNN forward: message passing across driver nodes."""
        for i in range(len(self.gnn_self_fcs)):
            self_out = self.gnn_self_fcs[i](x)
            deg      = adj.sum(dim=1, keepdim=True).clamp(min=1)
            nb_out   = self.gnn_nb_fcs[i]((adj @ x) / deg)
            x = self.gnn_act(self_out + nb_out)
            if i < len(self.gnn_self_fcs) - 1:
                x = self.gnn_dropout(x)
        return x

    def forward(self, lap_seq, current_feats, adj):
        """
        lap_seq       : (batch, seq_len, lstm_input)  - past laps per driver
        current_feats : (num_drivers, gnn_input)      - all drivers at current lap
        adj           : (num_drivers, num_drivers)    - race graph adjacency
        Returns       : (batch,) predicted lap times
        """
        # LSTM branch
        lstm_out, _ = self.lstm(lap_seq)
        lstm_feat   = lstm_out[:, -1, :]
        lstm_feat   = self.lstm_bn(lstm_feat)

        # GNN branch
        gnn_feat = self.gnn_forward(current_feats, adj)
        gnn_feat = self.gnn_bn(gnn_feat)
        gnn_feat = gnn_feat[:lap_seq.shape[0]]

        # Fusion
        combined = torch.cat([lstm_feat, gnn_feat], dim=1)
        out      = self.fusion(combined)
        return out.squeeze(1)
