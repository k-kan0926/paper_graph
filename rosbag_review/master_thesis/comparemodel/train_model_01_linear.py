#!/usr/bin/env python3
import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- 設定 ---
DATA_DIR = "./processed_data"  # 前処理済みCSVがあるフォルダ
MODEL_DIR = "./models/linear"
os.makedirs(MODEL_DIR, exist_ok=True)

# 過去の何ステップを見るか（NARXのLagsに相当）
LAGS = 5

# 入力として使う列 (制御入力 + 過去の状態フィードバック)
INPUT_COLS = ['p1_cmd[MPa]', 'p2_cmd[MPa]'] 
# 予測したい列 (状態量)
TARGET_COLS = ['theta[rad]'] 

def create_lagged_features(df, input_cols, target_cols, lags):
    """ラグ特徴量の作成 (時刻 t-1 ... t-lags のデータを行方向に展開)"""
    df_lagged = df.copy()
    
    # 特徴量リスト
    feature_names = []
    
    # 1. 制御入力のラグ (u_{t-1}...u_{t-lags})
    for col in input_cols:
        for i in range(1, lags + 1):
            new_col = f'{col}_lag{i}'
            df_lagged[new_col] = df[col].shift(i)
            feature_names.append(new_col)
            
    # 2. 状態(正解)のラグ (y_{t-1}...y_{t-lags}) -> 自己回帰成分
    for col in target_cols:
        for i in range(1, lags + 1):
            new_col = f'{col}_lag{i}'
            df_lagged[new_col] = df[col].shift(i)
            feature_names.append(new_col)
            
    df_lagged = df_lagged.dropna()
    return df_lagged, feature_names

def main():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("CSV not found.")
        return

    all_X = []
    all_y = []

    print(f"Loading {len(csv_files)} files...")
    for f in csv_files:
        df = pd.read_csv(f)


        # ラグ特徴量の作成
        df_proc, feature_names = create_lagged_features(df, INPUT_COLS, TARGET_COLS, LAGS)
        
        X = df_proc[feature_names].values
        y = df_proc[TARGET_COLS].values
        all_X.append(X)
        all_y.append(y)

    X_data = np.vstack(all_X)
    y_data = np.vstack(all_y)

    # スケーリング (Linearモデルでも係数の解釈や数値安定性のため推奨)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X_data)
    y_scaled = scaler_y.fit_transform(y_data)

    # --- 学習 ---
    print("Training Linear ARX...")
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)

    # --- 保存 ---
    print(f"Score (R^2): {model.score(X_scaled, y_scaled):.4f}")
    
    joblib.dump(model, os.path.join(MODEL_DIR, "model_linear.pkl"))
    joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, "scaler_y.pkl"))
    
    # 評価時に必要なメタデータを保存
    meta = {
        'lags': LAGS,
        'feature_names': feature_names,
        'input_cols': INPUT_COLS,
        'target_cols': TARGET_COLS,
        'model_type': 'linear'
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "meta.pkl"))
    print(f"Saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()