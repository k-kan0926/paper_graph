#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_arx.py
線形ARXモデルの訓練

モデル: θ_{t+1} = Σ(a_i * θ_{t-i}) + Σ(b_i * p1_{t-i}) + Σ(c_i * p2_{t-i}) + d

線形なので解析的に解ける → 制御も線形なので LQR/MPC が使いやすい
"""
import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

def load_csv(path):
    """CSVロード"""
    print(f"[Loading] {os.path.basename(path)}")
    df = pd.read_csv(path)
    
    required = ['t[s]', 'p1_cmd[MPa]', 'p2_cmd[MPa]', 'theta[rad]']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    df = df.sort_values('t[s]').drop_duplicates(subset=['t[s]']).reset_index(drop=True)
    df = df.dropna(subset=required)
    
    t = df['t[s]'].values
    dt = np.median(np.diff(t)) if len(t) > 2 else 0.005
    
    print(f"  → {len(df)} samples, dt={dt*1000:.2f}ms")
    return df, float(dt)

def build_arx_dataset(df, lags):
    """ARXデータセット構築
    
    X: [θ_{t-1}, ..., θ_{t-lags}, p1_{t-1}, ..., p1_{t-lags}, p2_{t-1}, ..., p2_{t-lags}]
    Y: θ_t
    """
    theta = df['theta[rad]'].values
    p1 = df['p1_cmd[MPa]'].values
    p2 = df['p2_cmd[MPa]'].values
    
    N = len(theta)
    if N < lags + 2:
        return None, None
    
    X_list, Y_list = [], []
    
    for t in range(lags, N - 1):
        features = []
        
        # Past theta
        for i in range(lags):
            features.append(theta[t - i])
        
        # Past p1
        for i in range(lags):
            features.append(p1[t - i])
        
        # Past p2
        for i in range(lags):
            features.append(p2[t - i])
        
        X_list.append(features)
        Y_list.append(theta[t + 1])
    
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32).reshape(-1, 1)
    
    print(f"  Built ARX dataset: X={X.shape}, Y={Y.shape}")
    return X, Y

def stack_sessions(csv_list, lags):
    """複数CSVを統合"""
    X_all, Y_all = [], []
    
    for path in csv_list:
        df, _ = load_csv(path)
        X, Y = build_arx_dataset(df, lags)
        if X is None:
            print(f"  [SKIP] {os.path.basename(path)}")
            continue
        X_all.append(X)
        Y_all.append(Y)
    
    if not X_all:
        raise ValueError("No usable samples")
    
    X_combined = np.vstack(X_all)
    Y_combined = np.vstack(Y_all)
    
    print(f"[Total] X={X_combined.shape}, Y={Y_combined.shape}")
    return X_combined, Y_combined

def evaluate(model, X, Y):
    """評価"""
    Y_pred = model.predict(X)
    mse = mean_squared_error(Y, Y_pred)
    mae = mean_absolute_error(Y, Y_pred)
    
    return {
        'rmse': float(np.sqrt(mse)),
        'mae': float(mae),
        'r2': float(model.score(X, Y)),
        'n': len(Y)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dyn_csvs', nargs='+', required=True)
    parser.add_argument('--out_dir', type=str, default='arx_model')
    parser.add_argument('--lags', type=int, default=24)
    parser.add_argument('--alpha', type=float, default=1.0, help='Ridge regularization')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f" Linear ARX Model Training")
    print(f"{'='*70}")
    print(f"Lags: {args.lags}")
    print(f"Regularization: {args.alpha}")
    
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
    
    print(f"\n[Dataset]")
    print(f"  Train: {len(train_csvs)} files")
    print(f"  Val:   {len(val_csvs)} files")
    print(f"  Test:  {len(test_csvs)} files")
    
    # Build datasets
    X_train, Y_train = stack_sessions(train_csvs, args.lags)
    X_val, Y_val = stack_sessions(val_csvs, args.lags)
    X_test, Y_test = stack_sessions(test_csvs, args.lags)
    
    # Train linear model
    print(f"\n[Training] Ridge Regression...")
    model = Ridge(alpha=args.alpha, fit_intercept=True)
    model.fit(X_train, Y_train.ravel())
    
    # Evaluate
    print(f"\n[Evaluation]")
    train_metrics = evaluate(model, X_train, Y_train)
    val_metrics = evaluate(model, X_val, Y_val)
    test_metrics = evaluate(model, X_test, Y_test)
    
    print(f"  Train: RMSE={train_metrics['rmse']:.5f}, R²={train_metrics['r2']:.3f}")
    print(f"  Val:   RMSE={val_metrics['rmse']:.5f}, R²={val_metrics['r2']:.3f}")
    print(f"  Test:  RMSE={test_metrics['rmse']:.5f}, R²={test_metrics['r2']:.3f}")
    
    # Save model
    os.makedirs(args.out_dir, exist_ok=True)
    
    with open(os.path.join(args.out_dir, 'arx_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    meta = {
        'model_type': 'ARX',
        'lags': args.lags,
        'n_features': X_train.shape[1],
        'alpha': args.alpha,
        'coefficients': {
            'weights': model.coef_.tolist(),
            'intercept': float(model.intercept_)
        },
        'train_files': [os.path.basename(p) for p in train_csvs],
        'val_files': [os.path.basename(p) for p in val_csvs],
        'test_files': [os.path.basename(p) for p in test_csvs]
    }
    
    with open(os.path.join(args.out_dir, 'arx_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Save metrics
    metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
    
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[Saved] {args.out_dir}/")
    print(f"  - arx_model.pkl")
    print(f"  - arx_meta.json")
    print(f"  - metrics.json")
    
    # Print model summary
    print(f"\n{'='*70}")
    print(f" Model Summary")
    print(f"{'='*70}")
    print(f"Input dimension: {X_train.shape[1]}")
    print(f"  θ coefficients: {args.lags}")
    print(f"  p1 coefficients: {args.lags}")
    print(f"  p2 coefficients: {args.lags}")
    print(f"Intercept: {model.intercept_:.6f}")
    print(f"Max |coefficient|: {np.max(np.abs(model.coef_)):.6f}")

if __name__ == '__main__':
    main()