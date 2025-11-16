#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_narx_residual.py
Residual接続付きNARX訓練スクリプト

Key Feature:
  θ(t+1) = θ(t) + Δθ  (Δθをネットワークで予測)
  
Rollout安定性が大幅に向上
"""
import os, json, argparse, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

def torch_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ========== Data Loading ==========
def load_csv(path: str):
    """CSVロード"""
    print(f"[Loading] {os.path.basename(path)}")
    df = pd.read_csv(path)
    
    # 必須カラム
    required = ['t[s]', 'p1_cmd[MPa]', 'p2_cmd[MPa]', 'theta[rad]']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    
    df = df.sort_values('t[s]').drop_duplicates(subset=['t[s]']).reset_index(drop=True)
    df = df.dropna(subset=['theta[rad]', 'p1_cmd[MPa]', 'p2_cmd[MPa]'])
    
    t = df['t[s]'].values
    dt = np.median(np.diff(t)) if len(t) > 2 else 0.005
    
    # 微分項計算
    if 'dp1_cmd_dt[MPa/s]' not in df.columns:
        p1 = df['p1_cmd[MPa]'].values
        p2 = df['p2_cmd[MPa]'].values
        
        dp1 = np.zeros_like(p1)
        dp2 = np.zeros_like(p2)
        
        if len(p1) > 2:
            dp1[1:-1] = (p1[2:] - p1[:-2]) / (2 * dt)
            dp1[0] = (p1[1] - p1[0]) / dt
            dp1[-1] = (p1[-1] - p1[-2]) / dt
            
            dp2[1:-1] = (p2[2:] - p2[:-2]) / (2 * dt)
            dp2[0] = (p2[1] - p2[0]) / dt
            dp2[-1] = (p2[-1] - p2[-2]) / dt
        
        df['dp1_cmd_dt[MPa/s]'] = dp1
        df['dp2_cmd_dt[MPa/s]'] = dp2
    
    if 'dz[m]' not in df.columns:
        df['dz[m]'] = 0.0
    
    print(f"  → {len(df)} samples, dt={dt*1000:.2f}ms")
    return df, float(dt)

def make_feature_cols(use_dz=False):
    """特徴量リスト"""
    base = ['theta[rad]', 
            'p1_cmd[MPa]', 'p2_cmd[MPa]',
            'dp1_cmd_dt[MPa/s]', 'dp2_cmd_dt[MPa/s]']
    if use_dz:
        base.append('dz[m]')
    return base

def build_sequences_from_df(df: pd.DataFrame, lags: int, delay: int, feat_cols):
    """ラグ付き特徴量作成"""
    df = df.reset_index(drop=True)
    N = len(df)
    if N < (lags + delay + 2):
        return None, None
    
    Y_full = df['theta[rad]'].values.astype(np.float32)
    X_list, idx_list = [], []
    
    for t_idx in range(lags + delay, N - 1):
        base = t_idx - delay
        fv = []
        ok = True
        for k in range(lags):
            i = base - k
            if i < 0:
                ok = False
                break
            row = df.iloc[i][feat_cols].values.astype(np.float32)
            fv.append(row)
        
        if not ok:
            continue
        
        X_list.append(np.concatenate(fv, axis=0))
        idx_list.append(t_idx)
    
    if not X_list:
        return None, None
    
    X = np.vstack(X_list).astype(np.float32)
    Y = Y_full[np.array(idx_list) + 1].reshape(-1, 1)
    
    return X, Y

def standardize_fit(X):
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return mu, std

def standardize_apply(X, mu, std):
    return (X - mu) / std

def stack_sessions(csv_list, lags, delay, feat_cols):
    """複数CSVを統合"""
    X_all, Y_all, slices = [], [], []
    offset = 0
    
    for path in csv_list:
        df, _ = load_csv(path)
        X, Y = build_sequences_from_df(df, lags, delay, feat_cols)
        if X is None:
            continue
        X_all.append(X)
        Y_all.append(Y)
        slices.append((path, offset, offset + len(Y)))
        offset += len(Y)
    
    if not X_all:
        raise ValueError("No usable samples")
    
    return np.vstack(X_all).astype(np.float32), np.vstack(Y_all).astype(np.float32), slices

def theta_range_from_csvs(csvs):
    """theta範囲取得"""
    th_min, th_max = +np.inf, -np.inf
    for p in csvs:
        df, _ = load_csv(p)
        th = df['theta[rad]'].values
        th_min = min(th_min, float(np.nanmin(th)))
        th_max = max(th_max, float(np.nanmax(th)))
    if not np.isfinite(th_min):
        th_min, th_max = -math.pi, math.pi
    return th_min, th_max

# ========== Residual NARX Model ==========
class ResidualNARX(nn.Module):
    """
    Residual接続付きNARXモデル
    
    Architecture:
        θ(t+1) = θ(t) + Δθ
        Δθ = MLP(x(t))
    
    Benefits:
        - 予測安定性向上（誤差が累積しにくい）
        - 勾配伝播改善（skip connection）
        - 物理的妥当性（連続性保証）
    """
    def __init__(self, in_dim, hidden=[224, 224], out_dim=1, dropout=0.08, 
                 residual_scale=1.0):
        """
        Args:
            in_dim: 入力次元（lags × n_features）
            hidden: 隠れ層次元のリスト
            out_dim: 出力次元（通常1: theta）
            dropout: Dropout率
            residual_scale: residual接続のスケーリング（初期値）
        """
        super().__init__()
        
        # メインネットワーク（Δθを予測）
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        
        layers.append(nn.Linear(d, out_dim))
        self.delta_net = nn.Sequential(*layers)
        
        # Residual接続用の重み（学習可能）
        # θ(t+1) = α * θ(t) + Δθ
        # 通常は α=1 だが、学習で最適化させることも可能
        self.residual_weight = nn.Parameter(torch.tensor([residual_scale]))
        
        # 初期化: Δθの初期予測を小さくする
        nn.init.xavier_uniform_(layers[-1].weight, gain=0.1)
        nn.init.zeros_(layers[-1].bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, in_dim) 入力特徴量
               x[:, 0] が θ(t) であることを仮定
        
        Returns:
            theta_next: (batch, 1) θ(t+1)の予測値
        """
        batch_size = x.size(0)
        
        # 現在の角度 θ(t) を抽出
        # NOTE: feature_colsの順序に依存（theta[rad]が最初）
        theta_t = x[:, 0:1]  # (batch, 1)
        
        # 変化量 Δθ を予測
        delta_theta = self.delta_net(x)  # (batch, 1)
        
        # Residual接続: θ(t+1) = α*θ(t) + Δθ
        theta_next = self.residual_weight * theta_t + delta_theta
        
        return theta_next
    
    def get_delta_prediction(self, x):
        """Δθのみを取得（デバッグ用）"""
        return self.delta_net(x)

# ========== Rollout Evaluation ==========
def rollout_predict(model, lags, delay, feat_cols, mu, std, df, steps=400,
                    device='cpu', clamp_theta=True, theta_minmax=None, teacher_beta=0.0):
    """Rollout評価（Residual対応）"""
    model.eval()
    df = df.reset_index(drop=True)
    N = len(df)
    start_idx = lags + delay
    end_idx = N - 1
    
    if end_idx <= start_idx:
        return None
    
    if theta_minmax is None:
        tmin = float(np.nanpercentile(df['theta[rad]'].values, 0.1))
        tmax = float(np.nanpercentile(df['theta[rad]'].values, 99.9))
        theta_minmax = (min(tmin, -2*math.pi), max(tmax, 2*math.pi))
    
    th_lo, th_hi = theta_minmax
    
    t0 = max(start_idx, N - steps - 2)
    preds_th = []
    theta_true = df['theta[rad]'].values
    
    hist_theta = deque([theta_true[i] for i in range(t0, max(0, t0-lags), -1)], maxlen=lags)
    
    for base in range(t0, end_idx):
        fv = []
        for k in range(lags):
            idx = base - k
            if idx < 0:
                break
            row = df.iloc[idx][feat_cols].values.astype(np.float32).copy()
            
            # thetaを予測値で上書き
            theta_idx = feat_cols.index('theta[rad]')
            if k < len(hist_theta):
                row[theta_idx] = list(hist_theta)[k]
            fv.append(row)
        
        if len(fv) != lags:
            break
        
        x = np.concatenate(fv, axis=0)[None, :]
        x_std = standardize_apply(x, mu, std)
        xt = torch.from_numpy(x_std).float().to(device)
        
        with torch.no_grad():
            y_hat = model(xt).cpu().numpy().reshape(-1)
        
        if not np.all(np.isfinite(y_hat)):
            print(f"[WARN] NaN at step {base}")
            break
        
        th_hat = float(y_hat[0])
        
        if clamp_theta:
            th_hat = float(np.clip(th_hat, th_lo, th_hi))
        
        # Teacher forcing
        th_used = (1.0 - teacher_beta) * th_hat + teacher_beta * theta_true[base + 1]
        
        hist_theta.appendleft(th_used)
        preds_th.append(th_hat)
        
        # Divergence guard
        if abs(th_used) > 10.0 * max(abs(th_lo), abs(th_hi)):
            print(f"[WARN] Diverged at step {base}")
            break
    
    y_true_seq = theta_true[t0 + 1 : t0 + 1 + len(preds_th)]
    if len(preds_th) == 0:
        return None
    
    preds_th = np.array(preds_th)
    err_th = preds_th - y_true_seq
    
    return {
        'rmse': float(np.sqrt(np.mean(err_th**2))),
        'mae': float(np.mean(np.abs(err_th))),
        'bias': float(np.mean(err_th)),
        'n': int(len(preds_th))
    }

# ========== Training ==========
def train(args):
    torch_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    
    feat_cols = make_feature_cols(use_dz=args.use_dz)
    
    print(f"\n{'='*70}")
    print(f" Residual-NARX Training")
    print(f"{'='*70}")
    print(f"Model: θ(t+1) = α*θ(t) + Δθ,  Δθ = MLP(x)")
    print(f"Features per lag: {feat_cols}")
    print(f"Lags: {args.lags}, Delay: {args.delay}")
    print(f"Input dim: {args.lags * len(feat_cols)}\n")
    
    # Dataset split
    dyn_csvs = list(args.dyn_csvs)
    if len(dyn_csvs) <= 2:
        train_dyn, val_dyn, test_dyn = dyn_csvs, dyn_csvs, dyn_csvs
    else:
        train_dyn = dyn_csvs[:-2]
        val_dyn = [dyn_csvs[-2]]
        test_dyn = [dyn_csvs[-1]]
    
    print(f"[Dataset Split]")
    print(f"  Train: {len(train_dyn)} files")
    print(f"  Val:   {len(val_dyn)} files")
    print(f"  Test:  {len(test_dyn)} files\n")
    
    # Build datasets
    print(f"[Building Datasets]")
    X_tr, Y_tr, tr_slices = stack_sessions(train_dyn, args.lags, args.delay, feat_cols)
    X_va, Y_va, va_slices = stack_sessions(val_dyn, args.lags, args.delay, feat_cols)
    X_te, Y_te, te_slices = stack_sessions(test_dyn, args.lags, args.delay, feat_cols)
    
    print(f"\n[Data Shapes]")
    print(f"  Train: X={X_tr.shape}, Y={Y_tr.shape}")
    print(f"  Val:   X={X_va.shape}, Y={Y_va.shape}")
    print(f"  Test:  X={X_te.shape}, Y={Y_te.shape}")
    
    # Normalize
    mu, std = standardize_fit(X_tr)
    X_tr_s = standardize_apply(X_tr, mu, std)
    X_va_s = standardize_apply(X_va, mu, std)
    X_te_s = standardize_apply(X_te, mu, std)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"\n[Device] {device}")
    
    # Tensors
    Xtr = torch.from_numpy(X_tr_s).float().to(device)
    ytr = torch.from_numpy(Y_tr).float().to(device)
    Xva = torch.from_numpy(X_va_s).float().to(device)
    yva = torch.from_numpy(Y_va).float().to(device)
    Xte = torch.from_numpy(X_te_s).float().to(device)
    yte = torch.from_numpy(Y_te).float().to(device)
    
    # Model
    in_dim = Xtr.shape[1]
    model = ResidualNARX(
        in_dim=in_dim,
        hidden=[args.hidden, args.hidden],
        out_dim=1,
        dropout=args.dropout,
        residual_scale=1.0
    ).to(device)
    
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(f"\n[Model]")
    print(f"  Type:       Residual-NARX")
    print(f"  Input dim:  {in_dim}")
    print(f"  Hidden:     {args.hidden} x 2")
    print(f"  Output dim: 1 (theta)")
    print(f"  Params:     {sum(p.numel() for p in model.parameters())}")
    
    # Static consistency pool
    static_pool = []
    if args.stat_csvs and args.ss_lambda > 0.0:
        print(f"\n[Static Consistency]")
        for spath in args.stat_csvs:
            sdf, dt_est = load_csv(spath)
            if len(sdf) < 3:
                continue
            dtheta_dt = np.gradient(sdf['theta[rad]'].values, dt_est)
            mask = np.abs(dtheta_dt) < args.ss_theta_eps
            sdf_qs = sdf.loc[mask].copy().reset_index(drop=True)
            Xss, Yss = build_sequences_from_df(sdf_qs, args.lags, args.delay, feat_cols)
            if Xss is not None and len(Yss) > 0:
                Xss_s = standardize_apply(Xss, mu, std)
                static_pool.append((
                    torch.from_numpy(Xss_s).float().to(device),
                    torch.from_numpy(Yss).float().to(device)
                ))
                print(f"  {os.path.basename(spath)}: {len(Yss)} samples")
        print(f"  Lambda: {args.ss_lambda}")
    
    # Loss functions
    def mse_loss(yhat, ytrue):
        return torch.mean((yhat - ytrue)**2)
    
    def delta_regularization(model, X):
        """Δθの正則化（変化量が大きすぎないように）"""
        delta = model.get_delta_prediction(X)
        return torch.mean(delta ** 2)
    
    def evaluate(X, Y):
        model.eval()
        with torch.no_grad():
            Yh = model(X)
            err = Yh - Y
            mse = torch.mean(err**2)
            mae = torch.mean(torch.abs(err))
            bias = torch.mean(err)
        return {
            'rmse': float(torch.sqrt(mse).item()),
            'mae': float(mae.item()),
            'bias': float(bias.item())
        }
    
    # Training loop
    best_val = float('inf')
    best_state = None
    no_improve = 0
    bs = args.batch_size
    
    print(f"\n[Training]")
    print(f"  Epochs: {args.epochs}, Batch: {bs}, LR: {args.lr}")
    print(f"  Patience: {args.patience}, Delta reg: {args.delta_reg_lambda}\n")
    
    for ep in range(1, args.epochs + 1):
        model.train()
        N = Xtr.shape[0]
        idx = torch.randperm(N, device=device)
        total_loss = 0.0
        
        for i0 in range(0, N, bs):
            sel = idx[i0:i0 + bs]
            xb, yb = Xtr[sel], ytr[sel]
            
            yhat = model(xb)
            loss = mse_loss(yhat, yb)
            
            # Δθ正則化
            if args.delta_reg_lambda > 0:
                loss_delta = delta_regularization(model, xb)
                loss = loss + args.delta_reg_lambda * loss_delta
            
            # Static consistency
            if static_pool and args.ss_lambda > 0.0:
                Xss_t, Yss_t = random.choice(static_pool)
                take = min(bs, Xss_t.shape[0])
                ridx = torch.randint(0, Xss_t.shape[0], (take,), device=device)
                ss_loss = mse_loss(model(Xss_t[ridx]), Yss_t[ridx])
                loss = loss + args.ss_lambda * ss_loss
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            
            total_loss += loss.item()
        
        # Evaluate
        tr_metrics = evaluate(Xtr, ytr)
        va_metrics = evaluate(Xva, yva)
        
        if ep % 10 == 0 or ep == 1:
            res_weight = float(model.residual_weight.item())
            print(f"[{ep:03d}] loss={total_loss/max(1,N):.6f} | "
                  f"TR rmse={tr_metrics['rmse']:.5f} | VA rmse={va_metrics['rmse']:.5f} | "
                  f"α={res_weight:.4f}")
        
        # Early stopping
        if va_metrics['rmse'] + 1e-6 < best_val:
            best_val = va_metrics['rmse']
            best_state = {'model': model.state_dict()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stop at epoch {ep}\n")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state['model'])
    
    # Final metrics
    tr_metrics = evaluate(Xtr, ytr)
    va_metrics = evaluate(Xva, yva)
    te_metrics = evaluate(Xte, yte)
    
    print(f"\n{'='*70}")
    print(f" Final Metrics")
    print(f"{'='*70}")
    print(f"  Train: rmse={tr_metrics['rmse']:.5f}, mae={tr_metrics['mae']:.5f}")
    print(f"  Val:   rmse={va_metrics['rmse']:.5f}, mae={va_metrics['mae']:.5f}")
    print(f"  Test:  rmse={te_metrics['rmse']:.5f}, mae={te_metrics['mae']:.5f}")
    print(f"  Residual weight α: {model.residual_weight.item():.4f}")
    
    # Rollout
    ro_metrics = None
    dt_est = 0.005
    theta_minmax = (-math.pi, math.pi)
    
    try:
        last_te_path = te_slices[-1][0]
        df_te, dt_est = load_csv(last_te_path)
        theta_minmax = theta_range_from_csvs([p for p, _, _ in tr_slices])
        
        print(f"\n[Rollout Evaluation]")
        ro_metrics = rollout_predict(
            model=model, lags=args.lags, delay=args.delay, feat_cols=feat_cols,
            mu=mu, std=std, df=df_te, steps=min(1200, len(df_te)),
            device=device, clamp_theta=True, theta_minmax=theta_minmax,
            teacher_beta=args.rollout_teacher_beta
        )
        if ro_metrics:
            print(f"  RMSE: {ro_metrics['rmse']:.5f} rad ({ro_metrics['rmse']*180/math.pi:.2f}°)")
            print(f"  MAE:  {ro_metrics['mae']:.5f} rad ({ro_metrics['mae']*180/math.pi:.2f}°)")
            print(f"  Steps: {ro_metrics['n']}")
    except Exception as e:
        print(f"[WARN] Rollout failed: {e}")
    
    # Save
    meta = {
        "model_type": "Residual_NARX_p1p2_cmd",
        "feature_names_single_slice": feat_cols,
        "lags": args.lags,
        "delay": args.delay,
        "delay_measured_ms": args.delay_measured_ms,
        "mu": mu.tolist(),
        "std": std.tolist(),
        "hidden": args.hidden,
        "dropout": args.dropout,
        "residual_weight": float(model.residual_weight.item()),
        "train_dyn_csvs": [os.path.basename(p) for p in train_dyn],
        "val_dyn_csvs": [os.path.basename(p) for p in val_dyn],
        "test_dyn_csvs": [os.path.basename(p) for p in test_dyn],
        "theta_train_minmax": list(theta_minmax),
        "dt_est": float(dt_est),
        "use_dz": args.use_dz
    }
    
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'narx_model.pt'))
    with open(os.path.join(args.out_dir, 'narx_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    metrics = {
        "train": tr_metrics,
        "val": va_metrics,
        "test": te_metrics,
        "rollout_test": ro_metrics
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[Saved]")
    print(f"  {os.path.join(args.out_dir, 'narx_model.pt')}")
    print(f"  {os.path.join(args.out_dir, 'narx_meta.json')}")
    print(f"  {os.path.join(args.out_dir, 'metrics.json')}")
    print(f"\n{'='*70}\n")

# ========== CLI ==========
def parse_args():
    ap = argparse.ArgumentParser(description='Residual-NARX Training')
    
    # Data
    ap.add_argument('--dyn_csvs', nargs='+', required=True)
    ap.add_argument('--stat_csvs', nargs='*', default=[])
    ap.add_argument('--out_dir', type=str, default='models/narx_residual')
    
    # Model
    ap.add_argument('--lags', type=int, default=24)
    ap.add_argument('--delay', type=int, default=17)
    ap.add_argument('--delay_measured_ms', type=float, default=83.58)
    ap.add_argument('--hidden', type=int, default=224)
    ap.add_argument('--dropout', type=float, default=0.08)
    ap.add_argument('--use_dz', action='store_true', default=False)
    
    # Training
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1.5e-4)
    ap.add_argument('--weight_decay', type=float, default=5e-6)
    ap.add_argument('--patience', type=int, default=35)
    ap.add_argument('--cpu', action='store_true')
    
    # Residual-specific
    ap.add_argument('--delta_reg_lambda', type=float, default=0.001,
                    help='Δθの正則化係数（大きすぎる変化を抑制）')
    
    # Static consistency
    ap.add_argument('--ss_lambda', type=float, default=0.08)
    ap.add_argument('--ss_theta_eps', type=float, default=0.006)
    
    # Rollout
    ap.add_argument('--rollout_teacher_beta', type=float, default=0.10)
    
    ap.add_argument('--seed', type=int, default=42)
    
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)