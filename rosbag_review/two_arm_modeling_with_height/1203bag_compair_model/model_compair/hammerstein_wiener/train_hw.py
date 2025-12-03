#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_hammerstein_wiener.py
Hammerstein-Wienerモデルの訓練

構造: 入力非線形 → 線形動特性 → 出力非線形
    [p1, p2] → [NL_input] → [Linear Dynamics] → [NL_output] → θ

利点:
- 物理的解釈が容易
- プロセス制御で標準的
- 線形部分は理論的に扱いやすい
"""
import os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

class HammersteinWienerModel(nn.Module):
    """Hammerstein-Wienerモデル
    
    構造:
    1. Input Nonlinearity: [p1, p2] → v (hidden_dim)
    2. Linear Dynamics: v_{t-lags:t} → x (hidden_dim)
    3. Output Nonlinearity: x → θ
    """
    
    def __init__(self, n_inputs=2, lags=24, hidden_dim=64, nl_hidden=32):
        super().__init__()
        
        self.n_inputs = n_inputs
        self.lags = lags
        self.hidden_dim = hidden_dim
        
        # 1. Input nonlinearity (static)
        self.input_nl = nn.Sequential(
            nn.Linear(n_inputs, nl_hidden),
            nn.Tanh(),
            nn.Linear(nl_hidden, hidden_dim),
            nn.Tanh()
        )
        
        # 2. Linear dynamics (ARX-like)
        # v_{t-1:t-lags} → x_t
        self.linear_dynamics = nn.Linear(hidden_dim * lags, hidden_dim, bias=True)
        
        # 3. Output nonlinearity (static)
        self.output_nl = nn.Sequential(
            nn.Linear(hidden_dim, nl_hidden),
            nn.Tanh(),
            nn.Linear(nl_hidden, 1)
        )
        
        print(f"[HW Model] Input NL: {n_inputs} → {hidden_dim}")
        print(f"[HW Model] Linear Dynamics: {hidden_dim}x{lags} → {hidden_dim}")
        print(f"[HW Model] Output NL: {hidden_dim} → 1")
    
    def forward(self, u_hist):
        """
        Args:
            u_hist: (batch, lags, n_inputs) - 過去の入力
        
        Returns:
            theta: (batch, 1) - 予測角度
        """
        batch_size = u_hist.shape[0]
        
        # 1. Apply input nonlinearity to each time step
        # (batch, lags, n_inputs) → (batch, lags, hidden_dim)
        v_hist = []
        for t in range(self.lags):
            v_t = self.input_nl(u_hist[:, t, :])
            v_hist.append(v_t)
        
        v_hist = torch.stack(v_hist, dim=1)  # (batch, lags, hidden_dim)
        
        # 2. Linear dynamics
        v_flat = v_hist.reshape(batch_size, -1)  # (batch, lags * hidden_dim)
        x = self.linear_dynamics(v_flat)  # (batch, hidden_dim)
        
        # 3. Output nonlinearity
        theta = self.output_nl(x)  # (batch, 1)
        
        return theta
    
    def get_linear_weights(self):
        """線形部分の重みを取得"""
        return self.linear_dynamics.weight.data.cpu().numpy()

def load_and_prepare_data(csv_list, lags):
    """データロードと前処理"""
    all_theta, all_p1, all_p2 = [], [], []
    
    for path in csv_list:
        print(f"[Loading] {os.path.basename(path)}")
        df = pd.read_csv(path)
        
        required = ['theta[rad]', 'p1_cmd[MPa]', 'p2_cmd[MPa]']
        if not all(c in df.columns for c in required):
            print(f"  [SKIP] Missing columns")
            continue
        
        df = df.dropna(subset=required)
        
        all_theta.append(df['theta[rad]'].values)
        all_p1.append(df['p1_cmd[MPa]'].values)
        all_p2.append(df['p2_cmd[MPa]'].values)
    
    # Build dataset
    X_list, Y_list = [], []
    
    for theta, p1, p2 in zip(all_theta, all_p1, all_p2):
        N = len(theta)
        if N < lags + 2:
            continue
        
        for t in range(lags, N - 1):
            # Input history: [p1, p2] for lags time steps
            u_hist = []
            for k in range(lags):
                u_hist.append([p1[t - lags + k], p2[t - lags + k]])
            
            X_list.append(u_hist)
            Y_list.append(theta[t + 1])
    
    X = np.array(X_list, dtype=np.float32)  # (N, lags, 2)
    Y = np.array(Y_list, dtype=np.float32).reshape(-1, 1)  # (N, 1)
    
    print(f"[Dataset] X={X.shape}, Y={Y.shape}")
    return X, Y

def train_model(model, train_loader, val_loader, device, epochs=300, lr=1e-3, patience=30):
    """モデル訓練"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                Y_pred = model(X_batch)
                loss = criterion(Y_pred, Y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d}: Train={np.sqrt(train_loss):.5f}, "
                  f"Val={np.sqrt(val_loss):.5f}")
        
        if patience_counter >= patience:
            print(f"[Early Stop] at epoch {epoch+1}")
            break
    
    return model

def evaluate(model, loader, device):
    """評価"""
    model.eval()
    Y_true, Y_pred = [], []
    
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            
            Y_true.append(Y_batch.numpy())
            Y_pred.append(pred.cpu().numpy())
    
    Y_true = np.vstack(Y_true)
    Y_pred = np.vstack(Y_pred)
    
    rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))
    mae = mean_absolute_error(Y_true, Y_pred)
    
    return {'rmse': float(rmse), 'mae': float(mae), 'n': len(Y_true)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dyn_csvs', nargs='+', required=True)
    parser.add_argument('--out_dir', type=str, default='hw_model')
    parser.add_argument('--lags', type=int, default=24)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nl_hidden', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    
    device = torch.device('cpu' if args.cpu else 
                         ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    print(f"\n{'='*70}")
    print(f" Hammerstein-Wiener Model Training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Lags: {args.lags}")
    print(f"Hidden dim: {args.hidden_dim}")
    
    # Split data
    csvs = list(args.dyn_csvs)
    if len(csvs) == 1:
        train_csvs = val_csvs = test_csvs = csvs
    elif len(csvs) == 2:
        train_csvs, val_csvs, test_csvs = [csvs[0]], [csvs[1]], [csvs[1]]
    else:
        train_csvs = csvs[:-2]
        val_csvs = [csvs[-2]]
        test_csvs = [csvs[-1]]
    
    print(f"\n[Dataset Split]")
    print(f"  Train: {len(train_csvs)} files")
    print(f"  Val:   {len(val_csvs)} files")
    print(f"  Test:  {len(test_csvs)} files")
    
    # Load data
    X_train, Y_train = load_and_prepare_data(train_csvs, args.lags)
    X_val, Y_val = load_and_prepare_data(val_csvs, args.lags)
    X_test, Y_test = load_and_prepare_data(test_csvs, args.lags)
    
    # Create dataloaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = HammersteinWienerModel(
        n_inputs=2,
        lags=args.lags,
        hidden_dim=args.hidden_dim,
        nl_hidden=args.nl_hidden
    ).to(device)
    
    print(f"\n[Model Parameters] {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print(f"\n[Training]")
    model = train_model(model, train_loader, val_loader, device,
                       args.epochs, args.lr, args.patience)
    
    # Evaluate
    print(f"\n[Evaluation]")
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"  Train: RMSE={train_metrics['rmse']:.5f}, MAE={train_metrics['mae']:.5f}")
    print(f"  Val:   RMSE={val_metrics['rmse']:.5f}, MAE={val_metrics['mae']:.5f}")
    print(f"  Test:  RMSE={test_metrics['rmse']:.5f}, MAE={test_metrics['mae']:.5f}")
    
    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'hw_model.pt'))
    
    meta = {
        'model_type': 'Hammerstein-Wiener',
        'lags': args.lags,
        'hidden_dim': args.hidden_dim,
        'nl_hidden': args.nl_hidden,
        'n_inputs': 2,
        'train_files': [os.path.basename(p) for p in train_csvs],
        'val_files': [os.path.basename(p) for p in val_csvs],
        'test_files': [os.path.basename(p) for p in test_csvs]
    }
    
    with open(os.path.join(args.out_dir, 'hw_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
    
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[Saved] {args.out_dir}/")
    print(f"  - hw_model.pt")
    print(f"  - hw_meta.json")
    print(f"  - metrics.json")

if __name__ == '__main__':
    main()