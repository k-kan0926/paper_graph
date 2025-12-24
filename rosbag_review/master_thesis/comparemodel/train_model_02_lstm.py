#!/usr/bin/env python3
import os
import glob
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# --- 設定 ---
DATA_DIR = "./processed_data"
MODEL_DIR = "./models/lstm"
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LEN = 10  # 過去何ステップを入力系列として入れるか
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

INPUT_COLS = ['p1_cmd[MPa]', 'p2_cmd[MPa]']
TARGET_COLS = ['theta[rad]'] # 必要に応じて増やしてください

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # 最後のタイムステップの出力だけを使う (Many-to-One)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

def create_sequences(data_in, data_out, seq_len):
    """スライディングウィンドウで系列データを作成"""
    xs, ys = [], []
    for i in range(len(data_in) - seq_len):
        x = data_in[i:(i + seq_len)]
        y = data_out[i + seq_len] # 次の時刻を予測
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    # 全データを結合してからスケーリング
    df_list = [pd.read_csv(f) for f in csv_files]

    df_all = pd.concat(df_list, ignore_index=True)

    # 入力：制御入力 + 現在の状態 (NARX的アプローチの場合)
    # ここでは「入力系列を与えて次の状態を予測」
    # 入力特徴量 = [p1_cmd, p2_cmd, theta] (自己回帰を含む場合)
    train_feat_cols = INPUT_COLS + TARGET_COLS 
    
    raw_X = df_all[train_feat_cols].values
    raw_y = df_all[TARGET_COLS].values

    scaler_X = StandardScaler().fit(raw_X)
    scaler_y = StandardScaler().fit(raw_y)

    # ファイルごとに系列化して結合（時系列の不連続を防ぐため）
    X_seq_list, y_seq_list = [], []
    
    for df in df_list:
        data_in = scaler_X.transform(df[train_feat_cols].values)
        data_out = scaler_y.transform(df[TARGET_COLS].values)
        if len(data_in) > SEQ_LEN:
            xs, ys = create_sequences(data_in, data_out, SEQ_LEN)
            X_seq_list.append(xs)
            y_seq_list.append(ys)

    X_train = np.vstack(X_seq_list)
    y_train = np.vstack(y_seq_list)

    # Tensor化
    X_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_tensor = torch.FloatTensor(y_train).to(DEVICE)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # モデル定義
    input_dim = len(train_feat_cols)
    output_dim = len(TARGET_COLS)
    model = LSTMModel(input_dim, HIDDEN_SIZE, output_dim, NUM_LAYERS).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Training LSTM on {DEVICE}...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            output = model(bx)
            loss = criterion(output, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.5f}")

    # 保存
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model_lstm.pth"))
    joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, "scaler_y.pkl"))

    meta = {
        'seq_len': SEQ_LEN,
        'input_dim': input_dim,
        'hidden_size': HIDDEN_SIZE,
        'output_dim': output_dim,
        'num_layers': NUM_LAYERS,
        'feature_cols': train_feat_cols, # 入力に使った列順序
        'target_cols': TARGET_COLS,
        'model_type': 'lstm'
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "meta.pkl"))
    print(f"Saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()