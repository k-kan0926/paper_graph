#!/usr/bin/env python3
"""
preprocess_csvs_for_narx.py
Raw CSV → NARX訓練用に変換

Input CSV columns (your format):
  t[s], p1_cmd[MPa], p2_cmd[MPa], p1_meas[MPa], p2_meas[MPa],
  p_sum_cmd[MPa], p_diff_cmd[MPa], p_sum_meas[MPa], p_diff_meas[MPa],
  dp_sum_dt[MPa/s], dp_diff_dt[MPa/s], theta[rad], theta[deg], z[m], dz[m]

Output: 統一フォーマット + 微分項追加
"""
import pandas as pd
import numpy as np
import argparse
import os

def preprocess_csv(input_path, output_path):
    """
    CSVを読み込み、NARX用に整形
    - dp1_cmd/dt, dp2_cmd/dt を計算
    - 欠損値除去
    - 時系列ソート
    """
    df = pd.read_csv(input_path)
    
    # 必須カラムチェック
    required = ['t[s]', 'p1_cmd[MPa]', 'p2_cmd[MPa]', 'theta[rad]']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{input_path}: 欠損カラム {missing}")
    
    # ソート & 重複削除
    df = df.sort_values('t[s]').drop_duplicates(subset=['t[s]']).reset_index(drop=True)
    
    # 時間間隔
    t = df['t[s]'].values
    dt = np.median(np.diff(t))
    print(f"[{os.path.basename(input_path)}] dt={dt*1000:.2f}ms, samples={len(df)}")
    
    # 微分項の計算（中央差分）
    p1_cmd = df['p1_cmd[MPa]'].values
    p2_cmd = df['p2_cmd[MPa]'].values
    
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
    
    # dz補完（なければゼロ）
    if 'dz[m]' not in df.columns:
        df['dz[m]'] = 0.0
        print(f"  → dz[m] を0で補完")
    
    # 欠損値除去
    df = df.dropna(subset=['theta[rad]', 'p1_cmd[MPa]', 'p2_cmd[MPa]'])
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  → 保存: {output_path} ({len(df)} samples)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Raw CSVディレクトリ')
    parser.add_argument('--output_dir', required=True, help='出力ディレクトリ')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for fname in os.listdir(args.input_dir):
        if not fname.endswith('.csv'):
            continue
        
        input_path = os.path.join(args.input_dir, fname)
        output_path = os.path.join(args.output_dir, fname)
        
        try:
            preprocess_csv(input_path, output_path)
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

if __name__ == '__main__':
    main()