#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gru_nlss.py
GRU-based Nonlinear State Space Model (GRU-NLSS)の訓練

構造:
- 潜在状態: h ∈ R^n
- 状態遷移: h_{t+1} = GRU(h_t, u_t)
- 観測方程式: θ_t = g(h_t)  (MLPデコーダ)

利点:
- 長期依存性を捉えやすい
- 状態空間モデルとして解釈可能
- 機械学習系制御で標準的
"""
import os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

class GRU_NLSS(nn.Module):
    """GRU-based Nonlinear State Space Model"""
    
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # State transition: h_{t+1} = GRU(h_t, u_t)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Observation decoder: h_t → θ_t
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        print(f"[GRU-NLSS] Input: {input_dim}, Hidden: {hidden_dim}, Layers: {num_layers}")
    
    def forward(self, u_seq, h0=None):
        """
        Args:
            u_seq: (batch, seq_len, input_dim) - 制御入力シーケンス
            h0: (num_layers, batch, hidden_dim) - 初期状態（オプション）
        
        Returns:
            theta: (batch, seq_len, 1) - 予測角度シーケンス
            h: (num_layers, batch, hidden_dim) - 最終隠れ状態
        """
        # GRU forward
        h_seq, h_final = self.gru(u_seq, h0)  # h_seq: (batch, seq_len, hidden_dim)
        
        # Decode each time step
        batch_size, seq_len, _ = h_seq.shape
        h_flat = h_seq.reshape(batch_size * seq_len, self.hidden_dim)
        
        theta_flat = self.decoder(h_flat)  # (batch * seq_len, 1)
        theta = theta_flat.reshape(batch_size, seq_len, 1)
        
        return theta, h_final
    
    def init_hidden(self, batch_size, device):
        """初期隠れ状態を作成"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

def load_and_prepare_sequences(csv_list, seq_len=30):
    """シーケンスデータを作成"""
    all_sequences_u = []
    all_sequences_theta = []
    
    for path in csv_list:
        print(f"[Loading] {os.path.basename(path)}")
        df = pd.read_csv(path)
        
        required = ['theta[rad]', 'p1_cmd[MPa]', 'p2_cmd[MPa]']
        if not all(c in df.columns for c in required):
            print(f"  [SKIP] Missing columns")
            continue
        
        df = df.dropna(subset=required)
        
        theta = df['theta[rad]'].values
        p1 = df['p1_cmd[MPa]'].values
        p2 = df['p2_cmd[MPa]'].values
        
        N = len(theta)
        if N < seq_len + 1:
            continue
        
        # Create overlapping sequences
        for start in range(0, N - seq_len, seq_len // 2):
            if start + seq_len + 1 > N:
                break
            
            # Input sequence: u_{t:t+seq_len}
            u_seq = np.column_stack([
                p1[start:start+seq_len],
                p2[start:start+seq_len]
            ])
            
            # Target sequence: θ_{t+1:t+seq_len+1}
            theta_seq = theta[start+1:start+seq_len+1]
            
            all_sequences_u.append(u_seq)
            all_sequences_theta.append(theta_seq)
    
    U = np.array(all_sequences_u, dtype=np.float32)
    Theta = np.array(all_sequences_theta, dtype=np.float32)
    
    print(f"[Dataset] U={U.shape}, Theta={Theta.shape}")
    return U, Theta

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
        
        for U_batch, Theta_batch in train_loader:
            U_batch = U_batch.to(device)
            Theta_batch = Theta_batch.to(device)
            
            batch_size = U_batch.shape[0]
            h0 = model.init_hidden(batch_size, device)
            
            optimizer.zero_grad()
            
            theta_pred, _ = model(U_batch, h0)
            theta_pred = theta_pred.squeeze(-1)  # (batch, seq_len)
            
            loss = criterion(theta_pred, Theta_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for U_batch, Theta_batch in val_loader:
                U_batch = U_batch.to(device)
                Theta_batch = Theta_batch.to(device)
                
                batch_size = U_batch.shape[0]
                h0 = model.init_hidden(batch_size, device)
                
                theta_pred, _ = model(U_batch, h0)
                theta_pred = theta_pred.squeeze(-1)
                
                loss = criterion(theta_pred, Theta_batch)
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
        for U_batch, Theta_batch in loader:
            U_batch = U_batch.to(device)
            
            batch_size = U_batch.shape[0]
            h0 = model.init_hidden(batch_size, device)
            
            theta_pred, _ = model(U_batch, h0)
            theta_pred = theta_pred.squeeze(-1)
            
            Y_true.append(Theta_batch.numpy())
            Y_pred.append(theta_pred.cpu().numpy())
    
    Y_true = np.vstack(Y_true)
    Y_pred = np.vstack(Y_pred)
    
    rmse = np.sqrt(mean_squared_error(Y_true.flatten(), Y_pred.flatten()))
    mae = mean_absolute_error(Y_true.flatten(), Y_pred.flatten())
    
    return {'rmse': float(rmse), 'mae': float(mae), 'n': Y_true.size}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dyn_csvs', nargs='+', required=True)
    parser.add_argument('--out_dir', type=str, default='gru_nlss_model')
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    
    device = torch.device('cpu' if args.cpu else 
                         ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    print(f"\n{'='*70}")
    print(f" GRU-NLSS Model Training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Sequence length: {args.seq_len}")
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
    U_train, Theta_train = load_and_prepare_sequences(train_csvs, args.seq_len)
    U_val, Theta_val = load_and_prepare_sequences(val_csvs, args.seq_len)
    U_test, Theta_test = load_and_prepare_sequences(test_csvs, args.seq_len)
    
    # Create dataloaders
    train_dataset = TensorDataset(torch.from_numpy(U_train), torch.from_numpy(Theta_train))
    val_dataset = TensorDataset(torch.from_numpy(U_val), torch.from_numpy(Theta_val))
    test_dataset = TensorDataset(torch.from_numpy(U_test), torch.from_numpy(Theta_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = GRU_NLSS(
        input_dim=2,  # [p1, p2]
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
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
    
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'gru_nlss_model.pt'))
    
    meta = {
        'model_type': 'GRU-NLSS',
        'input_dim': 2,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'seq_len': args.seq_len,
        'train_files': [os.path.basename(p) for p in train_csvs],
        'val_files': [os.path.basename(p) for p in val_csvs],
        'test_files': [os.path.basename(p) for p in test_csvs]
    }
    
    with open(os.path.join(args.out_dir, 'gru_nlss_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
    
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[Saved] {args.out_dir}/")
    print(f"  - gru_nlss_model.pt")
    print(f"  - gru_nlss_meta.json")
    print(f"  - metrics.json")

if __name__ == '__main__':
    main()