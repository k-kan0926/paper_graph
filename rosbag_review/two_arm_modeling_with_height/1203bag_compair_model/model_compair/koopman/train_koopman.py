#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_koopman.py
Koopman Operatorモデルの訓練

理論:
非線形系 x_{t+1} = f(x_t, u_t) を高次元線形空間に埋め込み:
    z_{t+1} = K z_t + B u_t  (線形!)
    y_t = C z_t

ここで z_t = ψ(x_t, u_t) は観測可能量（observables）

手法:
- Extended DMD (Dynamic Mode Decomposition with Control)
- または Deep Koopman (ニューラルネットで ψ を学習)

利点:
- 線形制御理論が使える
- 理論的に美しい
- 近年人気上昇中
"""
import os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

class KoopmanModel(nn.Module):
    """Deep Koopmanモデル
    
    構造:
    1. Encoder: (θ, p1, p2) → z (高次元潜在空間)
    2. Linear Dynamics: z_{t+1} = K z_t + B u_t
    3. Decoder: z → θ
    """
    
    def __init__(self, state_dim=2, control_dim=2, latent_dim=64):
        super().__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.latent_dim = latent_dim
        
        # Encoder: (θ, p1, p2) → z
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + control_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Linear Koopman operator
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(control_dim, latent_dim, bias=False)
        
        # Decoder: z → θ
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        print(f"[Koopman] Latent dim: {latent_dim}")
        print(f"[Koopman] State: {state_dim}, Control: {control_dim}")
    
    def encode(self, x, u):
        """状態と入力を高次元空間に埋め込み"""
        xu = torch.cat([x, u], dim=-1)
        z = self.encoder(xu)
        return z
    
    def dynamics(self, z, u):
        """線形動特性: z_{t+1} = K z_t + B u_t"""
        z_next = self.K(z) + self.B(u)
        return z_next
    
    def decode(self, z):
        """潜在状態から観測へ"""
        theta = self.decoder(z)
        return theta
    
    def forward(self, x, u):
        """1ステップ予測"""
        z = self.encode(x, u)
        z_next = self.dynamics(z, u)
        theta_next = self.decode(z_next)
        return theta_next, z, z_next
    
    def get_koopman_matrix(self):
        """Koopman演算子の行列を取得"""
        return self.K.weight.data.cpu().numpy()

def load_and_prepare_data(csv_list):
    """データロードと前処理"""
    all_data = []
    
    for path in csv_list:
        print(f"[Loading] {os.path.basename(path)}")
        df = pd.read_csv(path)
        
        required = ['theta[rad]', 'p1_cmd[MPa]', 'p2_cmd[MPa]']
        if not all(c in df.columns for c in required):
            print(f"  [SKIP] Missing columns")
            continue
        
        df = df.dropna(subset=required)
        all_data.append(df[required].values)
    
    # Build dataset: (x_t, u_t) → x_{t+1}
    X_state, X_control, Y_state = [], [], []
    
    for data in all_data:
        N = len(data)
        if N < 2:
            continue
        
        for t in range(N - 1):
            theta_t = data[t, 0]
            p1_t, p2_t = data[t, 1], data[t, 2]
            theta_next = data[t + 1, 0]
            
            # State: [θ, θ^2] (非線形観測可能量を含める)
            x_t = [theta_t, theta_t**2]
            u_t = [p1_t, p2_t]
            x_next = [theta_next, theta_next**2]
            
            X_state.append(x_t)
            X_control.append(u_t)
            Y_state.append(x_next)
    
    X_state = np.array(X_state, dtype=np.float32)
    X_control = np.array(X_control, dtype=np.float32)
    Y_state = np.array(Y_state, dtype=np.float32)
    
    print(f"[Dataset] X_state={X_state.shape}, X_control={X_control.shape}, Y_state={Y_state.shape}")
    return X_state, X_control, Y_state

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
        train_loss_recon = 0.0
        train_loss_linear = 0.0
        
        for X_state, X_control, Y_state in train_loader:
            X_state = X_state.to(device)
            X_control = X_control.to(device)
            Y_state = Y_state.to(device)
            
            optimizer.zero_grad()
            
            # Prediction
            theta_next_pred, z, z_next = model(X_state, X_control)
            
            # Loss 1: Reconstruction (θ prediction)
            theta_next_true = Y_state[:, 0:1]  # First element is θ
            loss_recon = criterion(theta_next_pred, theta_next_true)
            
            # Loss 2: Linearity in latent space
            # z_next should equal K z + B u
            z_target = model.encode(Y_state, X_control)
            loss_linear = criterion(z_next, z_target)
            
            # Total loss
            loss = loss_recon + 0.1 * loss_linear
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_loss_recon += loss_recon.item()
            train_loss_linear += loss_linear.item()
        
        train_loss /= len(train_loader)
        train_loss_recon /= len(train_loader)
        train_loss_linear /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_state, X_control, Y_state in val_loader:
                X_state = X_state.to(device)
                X_control = X_control.to(device)
                Y_state = Y_state.to(device)
                
                theta_next_pred, z, z_next = model(X_state, X_control)
                theta_next_true = Y_state[:, 0:1]
                loss = criterion(theta_next_pred, theta_next_true)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d}: Train={np.sqrt(train_loss_recon):.5f}, "
                  f"Linear={train_loss_linear:.5f}, Val={np.sqrt(val_loss):.5f}")
        
        if patience_counter >= patience:
            print(f"[Early Stop] at epoch {epoch+1}")
            break
    
    return model

def evaluate(model, loader, device):
    """評価"""
    model.eval()
    Y_true, Y_pred = [], []
    
    with torch.no_grad():
        for X_state, X_control, Y_state in loader:
            X_state = X_state.to(device)
            X_control = X_control.to(device)
            
            theta_next_pred, _, _ = model(X_state, X_control)
            theta_next_true = Y_state[:, 0:1]
            
            Y_true.append(theta_next_true.numpy())
            Y_pred.append(theta_next_pred.cpu().numpy())
    
    Y_true = np.vstack(Y_true)
    Y_pred = np.vstack(Y_pred)
    
    rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))
    mae = mean_absolute_error(Y_true, Y_pred)
    
    return {'rmse': float(rmse), 'mae': float(mae), 'n': len(Y_true)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dyn_csvs', nargs='+', required=True)
    parser.add_argument('--out_dir', type=str, default='koopman_model')
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    
    device = torch.device('cpu' if args.cpu else 
                         ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    print(f"\n{'='*70}")
    print(f" Koopman Operator Model Training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Latent dim: {args.latent_dim}")
    
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
    X_state_train, X_control_train, Y_state_train = load_and_prepare_data(train_csvs)
    X_state_val, X_control_val, Y_state_val = load_and_prepare_data(val_csvs)
    X_state_test, X_control_test, Y_state_test = load_and_prepare_data(test_csvs)
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_state_train),
        torch.from_numpy(X_control_train),
        torch.from_numpy(Y_state_train)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_state_val),
        torch.from_numpy(X_control_val),
        torch.from_numpy(Y_state_val)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_state_test),
        torch.from_numpy(X_control_test),
        torch.from_numpy(Y_state_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = KoopmanModel(
        state_dim=2,  # [θ, θ^2]
        control_dim=2,  # [p1, p2]
        latent_dim=args.latent_dim
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
    
    # Koopman matrix analysis
    K_matrix = model.get_koopman_matrix()
    eigenvalues = np.linalg.eigvals(K_matrix)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    
    print(f"\n[Koopman Analysis]")
    print(f"  Matrix shape: {K_matrix.shape}")
    print(f"  Max eigenvalue magnitude: {max_eigenvalue:.3f}")
    print(f"  Stable: {'Yes' if max_eigenvalue < 1 else 'No'}")
    
    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'koopman_model.pt'))
    
    meta = {
        'model_type': 'Koopman',
        'state_dim': 2,
        'control_dim': 2,
        'latent_dim': args.latent_dim,
        'max_eigenvalue': float(max_eigenvalue),
        'stable': bool(max_eigenvalue < 1),
        'train_files': [os.path.basename(p) for p in train_csvs],
        'val_files': [os.path.basename(p) for p in val_csvs],
        'test_files': [os.path.basename(p) for p in test_csvs]
    }
    
    with open(os.path.join(args.out_dir, 'koopman_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
    
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[Saved] {args.out_dir}/")
    print(f"  - koopman_model.pt")
    print(f"  - koopman_meta.json")
    print(f"  - metrics.json")

if __name__ == '__main__':
    main()