#!/usr/bin/env python3
import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import lightgbm as lgb
import pysindy as ps

# --- ユーザー定義クラス(LSTM)の再定義 (ロードに必要) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# --- 設定 ---
MODELS_ROOT = "./models"
TEST_DATA_DIR = "./data_preprocessed" # テストに使うデータフォルダ
TEST_FILE_INDEX = 0  # どのファイルをテストに使うか
SIM_STEPS = 200      # 何ステップ先までシミュレーションするか
START_IDX = 100      # テストデータのどこから開始するか

# 各モデルのフォルダ名
MODEL_TYPES = ['linear', 'lstm', 'gp', 'sindy', 'lgbm']
# MODEL_TYPES = ['linear', 'lstm'] # デバッグ用

class ModelWrapper:
    """各モデルの入出力の違いを吸収するラッパー"""
    def __init__(self, model_dir, model_type):
        self.model_type = model_type
        self.meta = joblib.load(os.path.join(model_dir, "meta.pkl"))
        self.model_dir = model_dir
        self.input_cols = self.meta['input_cols']
        self.target_cols = self.meta['target_cols']
        
        self._load_model()

    def _load_model(self):
        if self.model_type == 'linear':
            self.model = joblib.load(os.path.join(self.model_dir, "model_linear.pkl"))
            self.scaler_X = joblib.load(os.path.join(self.model_dir, "scaler_X.pkl"))
            self.scaler_y = joblib.load(os.path.join(self.model_dir, "scaler_y.pkl"))
            
        elif self.model_type == 'lstm':
            self.device = 'cpu'
            self.model = LSTMModel(
                self.meta['input_dim'], self.meta['hidden_size'], 
                self.meta['output_dim'], self.meta['num_layers']
            ).to(self.device)
            self.model.load_state_dict(torch.load(
                os.path.join(self.model_dir, "model_lstm.pth"), map_location=self.device
            ))
            self.model.eval()
            self.scaler_X = joblib.load(os.path.join(self.model_dir, "scaler_X.pkl"))
            self.scaler_y = joblib.load(os.path.join(self.model_dir, "scaler_y.pkl"))
            
        elif self.model_type == 'gp':
            self.model = joblib.load(os.path.join(self.model_dir, "model_gp.pkl"))
            self.scaler_X = joblib.load(os.path.join(self.model_dir, "scaler_X.pkl"))
            self.scaler_y = joblib.load(os.path.join(self.model_dir, "scaler_y.pkl"))
            
        elif self.model_type == 'sindy':
            self.model = ps.SINDy()
            self.model.load(os.path.join(self.model_dir, "model_sindy.json"))
            
        elif self.model_type == 'lgbm':
            self.model = joblib.load(os.path.join(self.model_dir, "model_lgbm.pkl"))

    def predict_next_step(self, history_df):
        """
        history_df: 過去の全データを含むDataFrame (末尾が現在時刻)
        戻り値: 次のステップの予測値 (theta)
        """
        # --- Linear / GP / LGBM (Lag Feature based) ---
        if self.model_type in ['linear', 'gp', 'lgbm']:
            lags = self.meta['lags']
            if len(history_df) < lags + 1:
                return 0.0 # データ不足
            
            # 特徴量作成 (最後の1行だけ必要)
            # 一時的にデータフレームを作ってshiftする簡易実装
            # 速度重視ならnumpy操作にするが、ここでは可読性重視
            temp_df = history_df.iloc[-(lags+1):].copy() 
            
            feats = []
            feature_names = self.meta['feature_names']
            
            # ラグ特徴量を手動構築
            row_vals = {}
            # 最新の入力を取得するためにターゲットとは別に処理が必要だが、
            # 学習時は「tでの入力」ではなく「t-1, t-2...」を使っている点に注意。
            # 今回のラグ定義: u_{t-1}...u_{t-lags}, y_{t-1}...y_{t-lags} で y_t を予測
            
            # 直近のデータを取得
            last_idx = temp_df.index[-1]
            
            input_vec = []
            
            # 特徴量名の順序に従って値を抽出
            # feature_names: ['p1_cmd_lag1', ..., 'theta_lag1', ...]
            vals = []
            for fname in feature_names:
                # 'p1_cmd_lag1' -> col='p1_cmd', lag=1
                col_name = "_".join(fname.split('_')[:-1])
                lag_num = int(fname.split('_')[-1].replace('lag', ''))
                
                # dfの末尾(t-1)から数えて lag_num-1 個前
                # iloc[-1] が t-1, iloc[-2] が t-2
                val = temp_df[col_name].iloc[-lag_num]
                vals.append(val)
            
            X_in = np.array([vals])
            
            if self.model_type in ['linear', 'gp']:
                X_scaled = self.scaler_X.transform(X_in)
                y_pred_scaled = self.model.predict(X_scaled)
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
                return y_pred[0][0]
            else: # lgbm
                y_pred = self.model.predict(X_in)
                return y_pred[0]

        # --- LSTM (Sequence based) ---
        elif self.model_type == 'lstm':
            seq_len = self.meta['seq_len']
            feat_cols = self.meta['feature_cols'] # p1, p2, theta
            
            if len(history_df) < seq_len:
                return 0.0
            
            # 直近 seq_len 個のデータを取得
            data_segment = history_df[feat_cols].iloc[-seq_len:].values
            
            # Scaling
            data_scaled = self.scaler_X.transform(data_segment) # LSTMは入力全てスケール済みと仮定
            
            X_tensor = torch.FloatTensor([data_scaled]).to(self.device)
            with torch.no_grad():
                y_pred_scaled = self.model(X_tensor).cpu().numpy()
            
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            return y_pred[0][0]

        # --- SINDy (State based) ---
        elif self.model_type == 'sindy':
            # SINDyは x[k], u[k] から x[k+1] を予測
            curr_x = history_df[self.target_cols].iloc[-1].values
            curr_u = history_df[self.input_cols].iloc[-1].values # 入力は外部から与えられるu[k]が必要
            
            # predictは [[x_next]] を返す
            x_next = self.model.predict([curr_x], u=[curr_u])
            return x_next[0][0]

def main():
    # 1. テストデータの読み込み
    csv_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return
    test_csv = csv_files[TEST_FILE_INDEX]
    print(f"Loading Test Data: {test_csv}")
    df_truth = pd.read_csv(test_csv)
    
    # 2. モデルのロード
    wrappers = {}
    for m_type in MODEL_TYPES:
        m_path = os.path.join(MODELS_ROOT, m_type)
        if os.path.exists(m_path):
            print(f"Loading model: {m_type}")
            try:
                wrappers[m_type] = ModelWrapper(m_path, m_type)
            except Exception as e:
                print(f"Failed to load {m_type}: {e}")
    
    if not wrappers:
        print("No models loaded.")
        return

    # 3. シミュレーション実行 (Free Running)
    # スタート地点までのデータは実測を使う
    # それ以降、入力(p_cmd)は実測を使うが、状態(theta)は自分の予測を使う
    
    results = {name: [] for name in wrappers.keys()}
    ground_truth = []
    
    print(f"\nStarting Simulation from index {START_IDX} for {SIM_STEPS} steps...")
    
    # シミュレーション用の履歴バッファ初期化
    # 各モデルごとに、自分の予測値を追記していくDataFrameを持つ
    sim_dfs = {}
    initial_history = df_truth.iloc[:START_IDX].copy()
    
    for name in wrappers.keys():
        sim_dfs[name] = initial_history.copy()

    # ループ
    for t in range(SIM_STEPS):
        curr_idx = START_IDX + t
        if curr_idx >= len(df_truth) - 1:
            break
            
        # 真の値（比較用）
        true_val = df_truth['theta'].iloc[curr_idx] # t時点の正解ではなく、比較は予測結果(t+1)で行う
        
        # 次の時刻への入力 u_t (これは既知とする=制御指令)
        # SINDy等で「現在の入力」を使うか「過去の入力」を使うか整合性に注意
        # ここではデータセットの行 t にある u_t を使って x_{t+1} を予測すると仮定
        
        # 未来の制御入力コマンド (シミュレーションなので既知)
        next_cmd_p1 = df_truth['p1_cmd'].iloc[curr_idx]
        next_cmd_p2 = df_truth['p2_cmd'].iloc[curr_idx]
        
        for name, wrapper in wrappers.items():
            # 現在のDataFrameバッファ
            curr_df = sim_dfs[name]
            
            # 次の状態を予測
            pred_theta = wrapper.predict_next_step(curr_df)
            
            # バッファ更新: 新しい行を追加
            # 時刻tの入力と、予測された時刻t+1の状態(theta)をどう管理するか
            # 簡易的に: 次の行を作り、thetaには予測値を、cmdには実測(未来の指令)を入れる
            
            new_row = curr_df.iloc[-1].copy()
            new_row['p1_cmd'] = next_cmd_p1
            new_row['p2_cmd'] = next_cmd_p2
            new_row['theta'] = pred_theta 
            # 他のカラムは更新しない(使わないため)
            
            sim_dfs[name] = pd.concat([curr_df, pd.DataFrame([new_row])], ignore_index=True)
            results[name].append(pred_theta)
            
        ground_truth.append(df_truth['theta'].iloc[curr_idx + 1]) # t+1の正解

    # 4. 結果の可視化と評価
    plt.figure(figsize=(12, 6))
    
    # Ground Truth
    time_steps = np.arange(len(ground_truth))
    plt.plot(time_steps, ground_truth, 'k--', linewidth=2, label='Ground Truth')
    
    metrics = []
    
    for name, preds in results.items():
        # RMSE計算
        mse = np.mean((np.array(preds) - np.array(ground_truth))**2)
        rmse = np.sqrt(mse)
        metrics.append({'Model': name, 'RMSE': rmse})
        
        plt.plot(time_steps, preds, label=f'{name} (RMSE={rmse:.3f})')

    plt.title(f'Multi-step Free Running Simulation (Steps={SIM_STEPS})')
    plt.xlabel('Time Step')
    plt.ylabel('Theta [rad]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

    # 評価テーブル表示
    print("\n--- Evaluation Results ---")
    res_df = pd.DataFrame(metrics).sort_values('RMSE')
    print(res_df)

if __name__ == "__main__":
    main()