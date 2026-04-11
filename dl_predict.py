"""
dl_predict.py  –  F1 Deep Learning Prediction Script
Usage:
  python dl_predict.py --model lstm --input data/processed/features.csv
  python dl_predict.py --model gnn  --input data/processed/features.csv
  python dl_predict.py --model lstm_gnn --input data/processed/features.csv
"""

import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Model imports ──────────────────────────────────────────────────────────────
from models.lstm_model import LSTMModel
from models.gnn_model import GNNModel
from models.lstm_gnn_model import LSTMGNNModel


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    feature_cols = ['LapNumber', 'TyreLife', 'Stint',
                    'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return df, X


def build_adj(n: int) -> torch.Tensor:
    """Simple fully-connected adjacency for GNN inference."""
    return torch.ones(n, n, dtype=torch.float32)


def predict(model_name: str, csv_path: str, weights_path: str = None):
    df, X = load_data(csv_path)
    n_features = X.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Build model ──────────────────────────────────────────────────────────
    if model_name == 'lstm':
        model = LSTMModel(input_size=n_features, hidden_size=64, num_layers=2)
    elif model_name == 'gnn':
        model = GNNModel(in_channels=n_features, hidden_channels=32, out_channels=1)
    elif model_name == 'lstm_gnn':
        model = LSTMGNNModel(input_size=n_features, hidden_size=64,
                             gnn_hidden=32, num_layers=2)
    else:
        raise ValueError(f'Unknown model: {model_name}')

    # ── Load weights if provided ──────────────────────────────────────────────
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f'Loaded weights from {weights_path}')
    model.to(device).eval()

    # ── Run inference ─────────────────────────────────────────────────────────
    preds = []
    seq_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
    node_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    adj_tensor  = build_adj(len(X)).to(device)

    with torch.no_grad():
        if model_name == 'lstm':
            out = model(seq_tensor)
            preds = out.squeeze().cpu().numpy()
        elif model_name == 'gnn':
            out = model(node_tensor, adj_tensor)
            preds = out.squeeze().cpu().numpy()
        elif model_name == 'lstm_gnn':
            out = model(seq_tensor, node_tensor, adj_tensor)
            preds = out.squeeze().cpu().numpy()

    df['PredictedLapTime_ms'] = preds if preds.ndim > 0 else [float(preds)]
    out_path = csv_path.replace('.csv', f'_{model_name}_predictions.csv')
    df.to_csv(out_path, index=False)
    print(f'Predictions saved to {out_path}')
    print(df[['LapNumber', 'PredictedLapTime_ms']].head(10).to_string(index=False))
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='F1 DL Inference')
    parser.add_argument('--model', choices=['lstm', 'gnn', 'lstm_gnn'],
                        required=True, help='Model architecture')
    parser.add_argument('--input', required=True, help='Path to feature CSV')
    parser.add_argument('--weights', default=None,
                        help='Path to .pt weights file (optional)')
    args = parser.parse_args()
    predict(args.model, args.input, args.weights)
