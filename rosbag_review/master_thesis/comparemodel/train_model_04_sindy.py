#!/usr/bin/env python3
import os
import glob
import joblib
import numpy as np
import pandas as pd
import pysindy as ps
from sklearn.preprocessing import StandardScaler

# --- 設定 ---
DATA_DIR = "./processed_data"
MODEL_DIR = "./models/sindy"
os.makedirs(MODEL_DIR, exist_ok=True)

# 入力(制御)と状態(出力)
INPUT_COLS = ['p1_cmd[MPa]', 'p2_cmd[MPa]']
TARGET_COLS = ['theta[rad]'] 

# SINDyの設定: 多項式の次数 (degree=2 or 3 が一般的)
POLY_DEGREE = 3
# 閾値 (係数がこれより小さい項は0とみなして式を単純化)
THRESHOLD = 0.001

def main():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    # SINDyは時系列のリストを受け取れるので、ファイルごとにリスト化
    X_list = [] # 状態 [theta]
    U_list = [] # 制御入力 [p1, p2]

    print("Loading data for SINDy...")
    for f in csv_files:
        df = pd.read_csv(f)
        x_vals = df[TARGET_COLS].values
        u_vals = df[INPUT_COLS].values
        
        X_list.append(x_vals)
        U_list.append(u_vals)

    # --- SINDyモデル定義 ---
    # 離散時間 (discrete_time=True) : x[k+1] = f(x[k], u[k]) を推定
    optimizer = ps.STLSQ(threshold=THRESHOLD)
    feature_library = ps.PolynomialLibrary(degree=POLY_DEGREE, include_bias=True)

    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=feature_library,
        discrete_time=True, 
        feature_names=TARGET_COLS + INPUT_COLS 
    )

    print("Training SINDy model (finding equations)...")
    # multiple_trajectories=True で複数の実験データをまとめて学習
    model.fit(X_list, u=U_list, multiple_trajectories=True)

    # --- 結果表示 ---
    print("\n--- Discovered Equations (Discrete Time map) ---")
    model.print()
    
    # スコア計算 (R^2)
    score = model.score(X_list, u=U_list, multiple_trajectories=True)
    print(f"\nModel Score (R^2): {score:.4f}")

    # 保存
    # pysindyのモデルは直接pickleできない場合があるため、saveメソッドを使用
    model.save(os.path.join(MODEL_DIR, "model_sindy.json"))
    
    # メタデータ
    meta = {
        'input_cols': INPUT_COLS,
        'target_cols': TARGET_COLS,
        'poly_degree': POLY_DEGREE,
        'model_type': 'sindy'
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "meta.pkl"))
    print(f"Saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()