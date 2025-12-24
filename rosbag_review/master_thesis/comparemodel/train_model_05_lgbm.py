#!/usr/bin/env python3
import os
import glob
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- 設定 ---
DATA_DIR = "./processed_data"
MODEL_DIR = "./models/lgbm"
os.makedirs(MODEL_DIR, exist_ok=True)

LAGS = 5
INPUT_COLS = ['p1_cmd[MPa]', 'p2_cmd[MPa]']
TARGET_COLS = ['theta[rad]']

# LightGBM パラメータ
PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}
NUM_ROUND = 500

def create_lagged_features(df, input_cols, target_cols, lags):
    """Linearモデルと同じラグ特徴量作成関数"""
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

    print("Loading data for LightGBM...")
    for f in csv_files:
        df = pd.read_csv(f)
        df_proc, feature_names = create_lagged_features(df, INPUT_COLS, TARGET_COLS, LAGS)
        all_X.append(df_proc[feature_names].values)
        all_y.append(df_proc[TARGET_COLS].values)

    X_data = np.vstack(all_X)
    y_data = np.vstack(all_y).ravel() # LGBMは1次元ターゲットが基本

    # スケーリング (決定木系は必須ではないが、他と条件を揃えるため実施しても良い。今回はしないが、比較のためターゲットの正規化だけ考慮)
    # ここではターゲットのスケール感を見るためにそのまま学習させます（LGBMはスケール不変）
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    print("Training LightGBM...")
    model = lgb.train(
        PARAMS,
        lgb_train,
        num_boost_round=NUM_ROUND,
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[lgb.log_evaluation(50)]
    )

    # 保存
    joblib.dump(model, os.path.join(MODEL_DIR, "model_lgbm.pkl"))
    
    meta = {
        'lags': LAGS,
        'feature_names': feature_names,
        'input_cols': INPUT_COLS,
        'target_cols': TARGET_COLS,
        'model_type': 'lgbm'
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "meta.pkl"))
    print(f"Saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()