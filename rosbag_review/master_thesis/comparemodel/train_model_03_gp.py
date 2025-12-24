#!/usr/bin/env python3
import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

# --- 設定 ---
DATA_DIR = "./processed_data"
MODEL_DIR = "./models/gp"
os.makedirs(MODEL_DIR, exist_ok=True)

# GPは計算が重いため、学習データを間引く (例: 2000点だけ使う)
# 本格的にやるなら GPyTorch などで Sparse GP を使う必要がありますが、
# まずはsklearnで簡易比較します。
MAX_TRAIN_SAMPLES = 10000
LAGS = 2 # GPは入力次元が増えると辛いので少なめに

INPUT_COLS = ['p1_cmd[MPa]', 'p2_cmd[MPa]']
TARGET_COLS = ['theta[rad]']

def create_lagged_features(df, input_cols, target_cols, lags):
    """Linearと同じラグ特徴量作成"""
    df_lagged = df.copy()
    feature_names = []
    
    for col in input_cols:
        for i in range(1, lags + 1):
            new_col = f'{col}_lag{i}'
            df_lagged[new_col] = df[col].shift(i)
            feature_names.append(new_col)
            
    for col in target_cols:
        for i in range(1, lags + 1):
            new_col = f'{col}_lag{i}'
            df_lagged[new_col] = df[col].shift(i)
            feature_names.append(new_col)
            
    df_lagged = df_lagged.dropna()
    return df_lagged, feature_names

def main():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    all_X = []
    all_y = []

    print("Loading data for GP...")
    for f in csv_files:
        df = pd.read_csv(f)
        df_proc, feature_names = create_lagged_features(df, INPUT_COLS, TARGET_COLS, LAGS)
        all_X.append(df_proc[feature_names].values)
        all_y.append(df_proc[TARGET_COLS].values)

    X_data = np.vstack(all_X)
    y_data = np.vstack(all_y)

    # データ間引き (ランダムサンプリング)
    if len(X_data) > MAX_TRAIN_SAMPLES:
        print(f"Downsampling data from {len(X_data)} to {MAX_TRAIN_SAMPLES} for GP performance.")
        indices = np.random.choice(len(X_data), MAX_TRAIN_SAMPLES, replace=False)
        X_data = X_data[indices]
        y_data = y_data[indices]

    # スケーリング
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_data)
    y_scaled = scaler_y.fit_transform(y_data)

    # カーネル定義: RBF (滑らかさ) + WhiteKernel (ノイズ成分)
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    
    print("Training Gaussian Process (this may take time)...")
    # n_restarts_optimizer=0 にして高速化（精度求めるなら増やす）
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=42)
    model.fit(X_scaled, y_scaled)

    print(f"GP Log Marginal Likelihood: {model.log_marginal_likelihood(model.kernel_.theta):.4f}")

    # 保存
    joblib.dump(model, os.path.join(MODEL_DIR, "model_gp.pkl"))
    joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, "scaler_y.pkl"))

    meta = {
        'lags': LAGS,
        'feature_names': feature_names,
        'input_cols': INPUT_COLS,
        'target_cols': TARGET_COLS,
        'model_type': 'gp'
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "meta.pkl"))
    print(f"Saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()