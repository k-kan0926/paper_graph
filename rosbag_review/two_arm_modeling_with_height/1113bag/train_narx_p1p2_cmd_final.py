#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_narx_p1p2_cmd_final_fixed.py
あなたのCSVフォーマットに完全対応した修正版
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

# ========== Data Loading (修正版) ==========
def load_csv(path: str, use_dz: bool = True):
    """
    あなたのCSVフォーマット用に最適化
    元のカラム名: t[s], p1_cmd[MPa], p2_cmd[MPa], ..., theta[rad], dz[m]
    """
    print(f"[Loading] {os.path.basename(path)}")
    df = pd.read_csv(path)
    
    # 実際のカラム名を表示（デバッグ用）
    print(f"  Columns: {list(df.columns)[:10]}{'...' if len(df.columns)>10 else ''}")
    
    # 必須カラムチェック
    required = ['t[s]', 'p1_cmd[MPa]', 'p2_cmd[MPa]', 'theta[rad]']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    
    # ソート & クリーニング
    df = df.sort_values('t[s]').drop_duplicates(subset=['t[s]']).reset_index(drop=True)
    df = df.dropna(subset=['theta[rad]', 'p1_cmd[MPa]', 'p2_cmd[MPa]'])
    
    # 時間間隔計算
    t = df['t[s]'].values
    dt = np.median(np.diff(t)) if len(t) > 2 else 0.005
    
    # 微分項がない場合は計算
    if 'dp1_cmd_dt[MPa/s]' not in df.columns:
        print(f"  [Compute] dp1_cmd_dt, dp2_cmd_dt")
        p1_cmd = df['p1_cmd[MPa]'].values
        p2_cmd = df['p2_cmd[MPa]'].values
        
        # 中央差分
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
    
    # dz補完
    if 'dz[m]' not in df.columns:
        print(f"  [Supplement] dz[m] = 0")
        df['dz[m]'] = 0.0
    
    print(f"  → {len(df)} samples, dt={dt*1000:.2f}ms\n")
    return df, float(dt)

def make_feature_cols(use_dz=True):
    """特徴量リスト（1ラグ分）"""
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
        print(f"  [WARN] Insufficient samples: {N} < {lags+delay+2}")
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
            try:
                row = df.iloc[i][feat_cols].values.astype(np.float32)
                fv.append(row)
            except KeyError as e:
                print(f"  [ERROR] Missing column in row {i}: {e}")
                print(f"    Available: {list(df.columns)}")
                raise
        
        if not ok:
            continue
        
        X_list.append(np.concatenate(fv, axis=0))
        idx_list.append(t_idx)
    
    if not X_list:
        return None, None
    
    X = np.vstack(X_list).astype(np.float32)
    Y = Y_full[np.array(idx_list) + 1].reshape(-1, 1)
    
    print(f"  Built sequences: X={X.shape}, Y={Y.shape}")
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
    
    print(f"\n[Stacking {len(csv_list)} sessions]")
    for path in csv_list:
        df, _ = load_csv(path)
        X, Y = build_sequences_from_df(df, lags, delay, feat_cols)
        if X is None:
            print(f"  [SKIP] {os.path.basename(path)}: insufficient data\n")
            continue
        X_all.append(X)
        Y_all.append(Y)
        slices.append((path, offset, offset + len(Y)))
        offset += len(Y)
    
    if not X_all:
        raise ValueError("No usable samples from given CSVs")
    
    X_combined = np.vstack(X_all).astype(np.float32)
    Y_combined = np.vstack(Y_all).astype(np.float32)
    
    print(f"[Total] X={X_combined.shape}, Y={Y_combined.shape}\n")
    return X_combined, Y_combined, slices

def theta_range_from_csvs(csvs):
    """訓練データのtheta範囲"""
    th_min, th_max = +np.inf, -np.inf
    for p in csvs:
        df, _ = load_csv(p)
        th = df['theta[rad]'].values
        th_min = min(th_min, float(np.nanmin(th)))
        th_max = max(th_max, float(np.nanmax(th)))
    if not np.isfinite(th_min):
        th_min, th_max = -math.pi, math.pi
    return th_min, th_max

# ========== Model ==========
class MLP_NARX(nn.Module):
    def __init__(self, in_dim, hidden=[192, 192], out_dim=1, dropout=0.0):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# ========== Rollout ==========
def rollout_predict(model, lags, delay, feat_cols, mu, std, df, steps=400,
                    device='cpu', clamp_theta=True, theta_minmax=None, teacher_beta=0.0):
    """Free-run rollout評価"""
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
    theta_used = theta_true.copy()
    
    hist_theta = deque([theta_true[i] for i in range(t0, max(0, t0-lags), -1)], maxlen=lags)
    
    for base in range(t0, end_idx):
        fv = []
        for k in range(lags):
            idx = base - k
            if idx < 0:
                break
            row = df.iloc[idx][feat_cols].values.astype(np.float32).copy()
            
            # Override theta with predicted value
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
            print(f"[WARN] NaN at step {base}, stopping rollout")
            break
        
        th_hat = float(y_hat[0])
        
        if clamp_theta:
            th_hat = float(np.clip(th_hat, th_lo, th_hi))
        
        th_used_next = (1.0 - teacher_beta) * th_hat + teacher_beta * theta_true[base + 1]
        
        hist_theta.appendleft(th_used_next)
        preds_th.append(th_hat)
        
        if abs(th_used_next) > 10.0 * max(abs(th_lo), abs(th_hi)):
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
    print(f" NARX Training (p1_cmd, p2_cmd)")
    print(f"{'='*70}")
    print(f"Features per lag: {feat_cols}")
    print(f"Lags: {args.lags}, Delay: {args.delay}")
    print(f"Total input dim: {args.lags * len(feat_cols)}\n")
    
    # Dataset split
    dyn_csvs = list(args.dyn_csvs)
    if len(dyn_csvs) == 1:
        train_dyn, val_dyn, test_dyn = dyn_csvs, dyn_csvs, dyn_csvs
    elif len(dyn_csvs) == 2:
        train_dyn, val_dyn, test_dyn = [dyn_csvs[0]], [dyn_csvs[1]], [dyn_csvs[1]]
    else:
        train_dyn = dyn_csvs[:-2]
        val_dyn = [dyn_csvs[-2]]
        test_dyn = [dyn_csvs[-1]]
    
    print(f"[Dataset Split]")
    print(f"  Train: {len(train_dyn)} files")
    print(f"  Val:   {len(val_dyn)} files")
    print(f"  Test:  {len(test_dyn)} files")
    
    # Build datasets
    X_tr, Y_tr, tr_slices = stack_sessions(train_dyn, args.lags, args.delay, feat_cols)
    X_va, Y_va, va_slices = stack_sessions(val_dyn, args.lags, args.delay, feat_cols)
    X_te, Y_te, te_slices = stack_sessions(test_dyn, args.lags, args.delay, feat_cols)
    
    # Normalize
    mu, std = standardize_fit(X_tr)
    X_tr_s = standardize_apply(X_tr, mu, std)
    X_va_s = standardize_apply(X_va, mu, std)
    X_te_s = standardize_apply(X_te, mu, std)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"[Device] {device}\n")
    
    # Tensors
    Xtr = torch.from_numpy(X_tr_s).float().to(device)
    ytr = torch.from_numpy(Y_tr).float().to(device)
    Xva = torch.from_numpy(X_va_s).float().to(device)
    yva = torch.from_numpy(Y_va).float().to(device)
    Xte = torch.from_numpy(X_te_s).float().to(device)
    yte = torch.from_numpy(Y_te).float().to(device)
    
    # Model
    in_dim = Xtr.shape[1]
    model = MLP_NARX(in_dim, hidden=[args.hidden, args.hidden], out_dim=1, dropout=args.dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(f"[Model]")
    print(f"  Input dim:  {in_dim}")
    print(f"  Hidden:     {args.hidden} x 2")
    print(f"  Output dim: 1 (theta)")
    print(f"  Params:     {sum(p.numel() for p in model.parameters())}\n")
    
    # Static consistency pool
    static_pool = []
    if args.stat_csvs and args.ss_lambda > 0.0:
        print(f"[Static Consistency]")
        for spath in args.stat_csvs:
            sdf, dt_est = load_csv(spath)
            if len(sdf) < 3:
                continue
            dtheta_dt = np.gradient(sdf['theta[rad]'].values, dt_est)
            mask = np.abs(dtheta_dt) < args.ss_theta_eps
            sdf_qs = sdf.loc[mask].copy().reset_index(drop=True)
            print(f"  {os.path.basename(spath)}: {mask.sum()} quasi-static samples")
            Xss, Yss = build_sequences_from_df(sdf_qs, args.lags, args.delay, feat_cols)
            if Xss is not None and len(Yss) > 0:
                Xss_s = standardize_apply(Xss, mu, std)
                static_pool.append((
                    torch.from_numpy(Xss_s).float().to(device),
                    torch.from_numpy(Yss).float().to(device)
                ))
        print(f"  Total: {len(static_pool)} subsets, lambda={args.ss_lambda}\n")
    
    # Loss & eval
    def mse_loss(yhat, ytrue):
        return torch.mean((yhat - ytrue)**2)
    
    def evaluate(X, Y):
        model.eval()
        with torch.no_grad():
            Yh = model(X)
            err = Yh - Y
            mse = torch.mean(err**2)
            mae = torch.mean(torch.abs(err))
            bias = torch.mean(err)
        return {'rmse': float(torch.sqrt(mse).item()),
                'mae': float(mae.item()),
                'bias': float(bias.item())}
    
    # Training loop
    best_val = float('inf')
    best_state = None
    no_improve = 0
    bs = args.batch_size
    
    print(f"[Training]")
    print(f"  Epochs: {args.epochs}, Batch size: {bs}, LR: {args.lr}")
    print(f"  Patience: {args.patience}\n")
    
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
        
        tr_metrics = evaluate(Xtr, ytr)
        va_metrics = evaluate(Xva, yva)
        
        if ep % 10 == 0 or ep == 1:
            print(f"[{ep:03d}] loss={total_loss/max(1,N):.6f} | "
                  f"TR rmse={tr_metrics['rmse']:.5f} | VA rmse={va_metrics['rmse']:.5f}")
        
        if va_metrics['rmse'] + 1e-6 < best_val:
            best_val = va_metrics['rmse']
            best_state = {'model': model.state_dict()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stop at epoch {ep} (patience={args.patience})\n")
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
            print(f"  RMSE: {ro_metrics['rmse']:.5f}, Steps: {ro_metrics['n']}")
    except Exception as e:
        print(f"[WARN] Rollout failed: {e}")
    
    # Save
    meta = {
        "model_type": "NARX_MLP_p1p2_cmd",
        "feature_names_single_slice": feat_cols,
        "lags": args.lags,
        "delay": args.delay,
        "delay_measured_ms": args.delay_measured_ms,
        "mu": mu.tolist(),
        "std": std.tolist(),
        "hidden": args.hidden,
        "dropout": args.dropout,
        "train_dyn_csvs": [os.path.basename(p) for p in train_dyn],
        "val_dyn_csvs": [os.path.basename(p) for p in val_dyn],
        "test_dyn_csvs": [os.path.basename(p) for p in test_dyn],
        "static_csvs": [os.path.basename(p) for p in args.stat_csvs] if args.stat_csvs else [],
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--dyn_csvs', nargs='+', required=True)
    ap.add_argument('--stat_csvs', nargs='*', default=[])
    ap.add_argument('--out_dir', type=str, default='out_narx_p1p2')
    ap.add_argument('--lags', type=int, default=24)
    ap.add_argument('--delay', type=int, default=17)
    ap.add_argument('--delay_measured_ms', type=float, default=None)
    ap.add_argument('--hidden', type=int, default=192)
    ap.add_argument('--dropout', type=float, default=0.05)
    ap.add_argument('--use_dz', action='store_true', default=False)
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--weight_decay', type=float, default=5e-6)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--ss_lambda', type=float, default=0.12)
    ap.add_argument('--ss_theta_eps', type=float, default=0.008)
    ap.add_argument('--rollout_teacher_beta', type=float, default=0.05)
    ap.add_argument('--p_max_each_side_MPa', type=float, default=0.70)
    ap.add_argument('--p1_rate_limit_MPa_s', type=float, default=3.5)
    ap.add_argument('--p2_rate_limit_MPa_s', type=float, default=3.5)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)