#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_models_unified.py
複数のモデルアーキテクチャを統一的に訓練・評価

サポートするモデル:
1. NARX (MLP-based NARX)
2. Linear ARX
3. LSTM
4. GRU
5. Transformer
6. 1D-CNN + Dense
"""
import os, json, argparse, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, Tuple, Optional

def torch_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==================== Data Loading ====================
def load_csv(path: str):
    """CSVロード"""
    print(f"[Loading] {os.path.basename(path)}")
    df = pd.read_csv(path)
    
    required = ['t[s]', 'p1_cmd[MPa]', 'p2_cmd[MPa]', 'theta[rad]']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    
    df = df.sort_values('t[s]').drop_duplicates(subset=['t[s]']).reset_index(drop=True)
    df = df.dropna(subset=['theta[rad]', 'p1_cmd[MPa]', 'p2_cmd[MPa]'])
    
    t = df['t[s]'].values
    dt = np.median(np.diff(t)) if len(t) > 2 else 0.005
    
    # 微分項計算
    if 'dp1_cmd_dt[MPa/s]' not in df.columns:
        p1_cmd = df['p1_cmd[MPa]'].values
        p2_cmd = df['p2_cmd[MPa]'].values
        
        dp1_dt = np.zeros_like(p1_cmd)
        dp2_dt = np.zeros_like(p2_cmd)
        
        if len(p1_cmd) > 2:
            dp1_dt[1:-1] = (p1_cmd[2:] - p1_cmd[:-2]) / (2 * dt)
            dp1_dt[0] = (p1_cmd[1] - p1_cmd[0]) / dt
            dp1_dt[-1] = (p1_cmd[-1] - p1_cmd[-2]) / dt
            
            dp2_dt[1:-1] = (p2_cmd[2:] - p2_cmd[:-2]) / (2 * dt)
            dp2_dt[0] = (p2_cmd[1] - p2_cmd[0]) / dt
            dp2_dt[-1] = (p2_cmd[-1] - p2_cmd[-2]) / dt
        
        df['dp1_cmd_dt[MPa/s]'] = dp1_dt
        df['dp2_cmd_dt[MPa/s]'] = dp2_dt
    
    if 'dz[m]' not in df.columns:
        df['dz[m]'] = 0.0
    
    print(f"  → {len(df)} samples, dt={dt*1000:.2f}ms")
    return df, float(dt)

def make_feature_cols(use_dz=False):
    """特徴量カラムリスト"""
    base = ['theta[rad]', 'p1_cmd[MPa]', 'p2_cmd[MPa]',
            'dp1_cmd_dt[MPa/s]', 'dp2_cmd_dt[MPa/s]']
    if use_dz:
        base.append('dz[m]')
    return base

def build_sequences(df: pd.DataFrame, lags: int, feat_cols):
    """時系列データ構築 - shape: (N, lags, n_features)"""
    df = df.reset_index(drop=True)
    N = len(df)
    n_feat = len(feat_cols)
    
    if N < lags + 2:
        return None, None
    
    Y_full = df['theta[rad]'].values.astype(np.float32)
    X_list, idx_list = [], []
    
    for t in range(lags, N - 1):
        seq = []
        for k in range(lags):
            row = df.iloc[t - lags + k][feat_cols].values.astype(np.float32)
            seq.append(row)
        X_list.append(np.array(seq))  # (lags, n_feat)
        idx_list.append(t)
    
    if not X_list:
        return None, None
    
    X = np.stack(X_list, axis=0)  # (N, lags, n_feat)
    Y = Y_full[np.array(idx_list) + 1].reshape(-1, 1)
    
    print(f"  Built sequences: X={X.shape}, Y={Y.shape}")
    return X, Y

def stack_sessions(csv_list, lags, feat_cols):
    """複数CSVを統合"""
    X_all, Y_all = [], []
    
    for path in csv_list:
        df, _ = load_csv(path)
        X, Y = build_sequences(df, lags, feat_cols)
        if X is None:
            print(f"  [SKIP] {os.path.basename(path)}")
            continue
        X_all.append(X)
        Y_all.append(Y)
    
    if not X_all:
        raise ValueError("No usable samples")
    
    X_combined = np.concatenate(X_all, axis=0)
    Y_combined = np.concatenate(Y_all, axis=0)
    
    print(f"[Total] X={X_combined.shape}, Y={Y_combined.shape}")
    return X_combined, Y_combined

# ==================== Model Definitions ====================

class LinearARX(nn.Module):
    """線形ARXモデル"""
    def __init__(self, lags, n_features):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(lags * n_features, 1)
    
    def forward(self, x):
        # x: (batch, lags, n_features)
        x = self.flatten(x)
        return self.linear(x)

class MLP_NARX(nn.Module):
    """非線形NARX (MLP)"""
    def __init__(self, lags, n_features, hidden=[192, 192], dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten()
        
        layers = []
        in_dim = lags * n_features
        for h in hidden:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU()
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

class LSTM_Model(nn.Module):
    """LSTMモデル"""
    def __init__(self, n_features, hidden=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden, 1)
    
    def forward(self, x):
        # x: (batch, lags, n_features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 最後のタイムステップ
        return self.fc(out)

class GRU_Model(nn.Module):
    """GRUモデル"""
    def __init__(self, n_features, hidden=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            n_features, hidden, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

class TransformerModel(nn.Module):
    """Transformerモデル"""
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(n_features, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: (batch, lags, n_features)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # 最後のタイムステップ
        return self.fc(x)

class CNN_Model(nn.Module):
    """1D CNN + Denseモデル"""
    def __init__(self, n_features, hidden=128, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x):
        # x: (batch, lags, n_features) -> (batch, n_features, lags)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        return self.fc(x)

# ==================== Model Factory ====================

def create_model(model_type: str, lags: int, n_features: int, args) -> nn.Module:
    """モデル生成"""
    if model_type == 'linear_arx':
        return LinearARX(lags, n_features)
    
    elif model_type == 'narx':
        return MLP_NARX(lags, n_features, 
                       hidden=[args.hidden, args.hidden],
                       dropout=args.dropout)
    
    elif model_type == 'lstm':
        return LSTM_Model(n_features, 
                         hidden=args.hidden,
                         num_layers=2,
                         dropout=args.dropout)
    
    elif model_type == 'gru':
        return GRU_Model(n_features,
                        hidden=args.hidden,
                        num_layers=2,
                        dropout=args.dropout)
    
    elif model_type == 'transformer':
        return TransformerModel(n_features,
                               d_model=64,
                               nhead=4,
                               num_layers=2,
                               dropout=args.dropout)
    
    elif model_type == 'cnn':
        return CNN_Model(n_features,
                        hidden=args.hidden,
                        dropout=args.dropout)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ==================== Training ====================

def train_model(model, X_train, Y_train, X_val, Y_val, args, device):
    """モデル訓練"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    # Data to tensors
    X_tr = torch.from_numpy(X_train).float().to(device)
    Y_tr = torch.from_numpy(Y_train).float().to(device)
    X_va = torch.from_numpy(X_val).float().to(device)
    Y_va = torch.from_numpy(Y_val).float().to(device)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    bs = args.batch_size
    N = X_tr.shape[0]
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = torch.randperm(N, device=device)
        total_loss = 0.0
        
        for i0 in range(0, N, bs):
            sel = idx[i0:i0+bs]
            xb, yb = X_tr[sel], Y_tr[sel]
            
            optimizer.zero_grad()
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item() * len(sel)
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_va)
            val_loss = criterion(y_val_pred, Y_va).item()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}: train_loss={total_loss/N:.6f}, val_loss={val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stop at epoch {epoch}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model

def evaluate_model(model, X, Y, device):
    """モデル評価"""
    model.eval()
    X_t = torch.from_numpy(X).float().to(device)
    Y_t = torch.from_numpy(Y).float().to(device)
    
    with torch.no_grad():
        Y_pred = model(X_t)
        err = Y_pred - Y_t
        mse = torch.mean(err**2).item()
        mae = torch.mean(torch.abs(err)).item()
        bias = torch.mean(err).item()
    
    return {
        'rmse': float(np.sqrt(mse)),
        'mae': float(mae),
        'bias': float(bias),
        'n': len(Y)
    }

# ==================== Main Training Loop ====================

def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--dyn_csvs', nargs='+', required=True)
    parser.add_argument('--out_dir', type=str, default='models_comparison')
    
    # Model selection
    parser.add_argument('--models', nargs='+', 
                       default=['linear_arx', 'narx', 'lstm', 'gru', 'transformer', 'cnn'],
                       choices=['linear_arx', 'narx', 'lstm', 'gru', 'transformer', 'cnn'])
    
    # Model parameters
    parser.add_argument('--lags', type=int, default=24)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--use_dz', action='store_true')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=30)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    
    torch_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    print(f"\n{'='*70}")
    print(f" Multi-Model Training & Comparison")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Lags: {args.lags}")
    
    # Load data
    feat_cols = make_feature_cols(args.use_dz)
    n_features = len(feat_cols)
    
    dyn_csvs = list(args.dyn_csvs)
    if len(dyn_csvs) == 1:
        train_csvs = val_csvs = test_csvs = dyn_csvs
    elif len(dyn_csvs) == 2:
        train_csvs, val_csvs, test_csvs = [dyn_csvs[0]], [dyn_csvs[1]], [dyn_csvs[1]]
    else:
        train_csvs = dyn_csvs[:-2]
        val_csvs = [dyn_csvs[-2]]
        test_csvs = [dyn_csvs[-1]]
    
    print(f"\n[Dataset]")
    print(f"  Train: {len(train_csvs)} files")
    print(f"  Val:   {len(val_csvs)} files")
    print(f"  Test:  {len(test_csvs)} files")
    
    X_train, Y_train = stack_sessions(train_csvs, args.lags, feat_cols)
    X_val, Y_val = stack_sessions(val_csvs, args.lags, feat_cols)
    X_test, Y_test = stack_sessions(test_csvs, args.lags, feat_cols)
    
    # Train each model
    results = {}
    
    for model_type in args.models:
        print(f"\n{'='*70}")
        print(f" Training: {model_type.upper()}")
        print(f"{'='*70}")
        
        model = create_model(model_type, args.lags, n_features, args)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Train
        model = train_model(model, X_train, Y_train, X_val, Y_val, args, device)
        
        # Evaluate
        train_metrics = evaluate_model(model, X_train, Y_train, device)
        val_metrics = evaluate_model(model, X_val, Y_val, device)
        test_metrics = evaluate_model(model, X_test, Y_test, device)
        
        print(f"\n  Results:")
        print(f"    Train RMSE: {train_metrics['rmse']:.5f}")
        print(f"    Val   RMSE: {val_metrics['rmse']:.5f}")
        print(f"    Test  RMSE: {test_metrics['rmse']:.5f}")
        
        # Save model
        model_dir = os.path.join(args.out_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
        
        meta = {
            'model_type': model_type,
            'lags': args.lags,
            'n_features': n_features,
            'feature_names': feat_cols,
            'hidden': args.hidden,
            'dropout': args.dropout,
            'train_files': [os.path.basename(p) for p in train_csvs],
            'val_files': [os.path.basename(p) for p in val_csvs],
            'test_files': [os.path.basename(p) for p in test_csvs]
        }
        
        with open(os.path.join(model_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        
        metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
        
        with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        results[model_type] = {
            'metrics': metrics,
            'n_params': sum(p.numel() for p in model.parameters())
        }
    
    # Summary
    print(f"\n{'='*70}")
    print(f" SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Params':<10} {'Train RMSE':<12} {'Val RMSE':<12} {'Test RMSE':<12}")
    print(f"{'-'*70}")
    
    for model_type in args.models:
        r = results[model_type]
        print(f"{model_type:<15} {r['n_params']:<10} "
              f"{r['metrics']['train']['rmse']:<12.5f} "
              f"{r['metrics']['val']['rmse']:<12.5f} "
              f"{r['metrics']['test']['rmse']:<12.5f}")
    
    # Save summary
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Saved] {args.out_dir}/")

if __name__ == '__main__':
    main()