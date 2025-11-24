#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Usage:
python inverse7_2_narx_mppi_p1p2.py \
    --model-dir models/narx_p1p2_production2 \
    --theta-target-deg 30 \
    --horizon 15 \
    --steps 100 \
    --K 32 \
    --lambda 2.0 \
    --plot
"""

import os
import json
import time
import math
import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False

try:
    from scipy.interpolate import make_interp_spline
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ==================== Model Definition ====================

class MLP_NARX(nn.Module):
    """Production2と互換性のあるNARXモデル"""
    def __init__(self, in_dim, hidden=[256, 256], out_dim=1, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# ==================== MPPI Controller ====================

class NARX_MPPI_Simulator:
    """NARXモデルを用いたMPPIシミュレーター"""
    
    def __init__(self, args):
        # Load model
        self.load_model(args.model_dir)
        
        # Simulation parameters
        self.dt = args.dt if args.dt > 0 else float(self.meta.get('dt_est', 0.005))
        self.frame_skip = args.frame_skip
        self.sim_dt = self.dt * self.frame_skip  # 実効的な制御周期
        
        # MPPI parameters
        self.K = args.K
        self.H = args.horizon
        self.temperature = args.lam
        self.sigma_u = args.sigma_u
        
        # Cost weights
        self.w_tracking = args.w_tracking
        self.w_smooth = args.w_smooth
        self.w_effort = args.w_effort
        self.w_constraint = args.w_constraint
        
        # Physical limits
        self.p_max = args.p_max
        self.dp_max = args.dp_max  # MPa/s
        
        # Get training data bounds
        self.theta_min = -math.pi
        self.theta_max = math.pi
        if 'theta_train_minmax' in self.meta:
            self.theta_min = float(self.meta['theta_train_minmax'][0])
            self.theta_max = float(self.meta['theta_train_minmax'][1])
        
        # State variables
        self.theta_rad = 0.0
        self.p1_cmd = 0.0
        self.p2_cmd = 0.0
        
        # History buffers (for NARX features)
        maxlen = self.lags + 10
        self.hist_theta = deque([0.0] * maxlen, maxlen=maxlen)
        self.hist_p1_cmd = deque([0.0] * maxlen, maxlen=maxlen)
        self.hist_p2_cmd = deque([0.0] * maxlen, maxlen=maxlen)
        self.hist_dp1_dt = deque([0.0] * maxlen, maxlen=maxlen)
        self.hist_dp2_dt = deque([0.0] * maxlen, maxlen=maxlen)
        
        # Logging
        self.log_time = []
        self.log_theta = []
        self.log_theta_ref = []
        self.log_p1_cmd = []
        self.log_p2_cmd = []
        self.log_error = []
        self.log_cost = []

        # B-spline 参照用
        self.ref_spline = None   # B-spline オブジェクト
        self.ref_T = None        # 参照軌道が定義される総時間 [s]
        
        print(f"[MPPI Simulator] Initialized")
        print(f"  Model: {args.model_dir}")
        print(f"  dt: {self.sim_dt:.4f}s (frame_skip={self.frame_skip})")
        print(f"  MPPI: K={self.K}, H={self.H}, lambda={self.temperature}")
        print(f"  Device: {self.device}")
        print(f"  Theta range: [{math.degrees(self.theta_min):.1f}, {math.degrees(self.theta_max):.1f}] deg")
    
    def load_model(self, model_dir):
        """モデルとメタデータをロード"""
        meta_path = os.path.join(model_dir, 'narx_meta.json')
        model_path = os.path.join(model_dir, 'narx_model.pt')
        
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        
        self.lags = self.meta['lags']
        self.delay = self.meta['delay']
        self.feat_cols = self.meta['feature_names_single_slice']
        self.mu = np.array(self.meta['mu'], dtype=np.float32)
        self.std = np.array(self.meta['std'], dtype=np.float32)
        self.hidden = self.meta['hidden']
        self.dropout = self.meta.get('dropout', 0.0)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        in_dim = self.lags * len(self.feat_cols)
        
        self.model = MLP_NARX(
            in_dim,
            hidden=[self.hidden, self.hidden],
            out_dim=1,
            dropout=self.dropout
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[Model] Loaded: lags={self.lags}, delay={self.delay}, hidden={self.hidden}")
    
    def initialize_from_context(self, context_csv=None):
        """初期状態を設定"""
        if context_csv and os.path.exists(context_csv):
            df = pd.read_csv(context_csv)
            
            # 最後のlags分のデータから初期状態を構築
            tail_len = min(self.lags, len(df))
            tail_df = df.tail(tail_len)
            
            # 履歴を更新
            for i in range(tail_len-1, -1, -1):
                row = tail_df.iloc[i]
                self.hist_theta.appendleft(float(row.get('theta[rad]', 0.0)))
                self.hist_p1_cmd.appendleft(float(row.get('p1_cmd[MPa]', 0.0)))
                self.hist_p2_cmd.appendleft(float(row.get('p2_cmd[MPa]', 0.0)))
                self.hist_dp1_dt.appendleft(float(row.get('dp1_cmd_dt[MPa/s]', 0.0)))
                self.hist_dp2_dt.appendleft(float(row.get('dp2_cmd_dt[MPa/s]', 0.0)))
            
            if tail_len > 0:
                self.theta_rad = float(tail_df.iloc[-1].get('theta[rad]', 0.0))
                self.p1_cmd = float(tail_df.iloc[-1].get('p1_cmd[MPa]', 0.0))
                self.p2_cmd = float(tail_df.iloc[-1].get('p2_cmd[MPa]', 0.0))
            
            print(f"[Init] Loaded context from {context_csv}")
        else:
            # デフォルト初期化
            self.theta_rad = 0.0
            self.p1_cmd = 0.0
            self.p2_cmd = 0.0
            print(f"[Init] Using default initial state")

    def set_bspline_reference(self, waypoint_deg_list, total_time):
        """
        B-spline による参照軌道 θ_ref(t) を構築する
        waypoint_deg_list: [20, -30, 10] のような角度列[deg]
        total_time: 参照軌道が定義される総時間 [s] (例: steps * sim_dt)
        """
        if not _HAS_SCIPY:
            raise ImportError(
                "SciPy が必要です。`pip install scipy` でインストールしてください。"
            )
        if len(waypoint_deg_list) < 2:
            raise ValueError("B-spline 参照には少なくとも2点以上のwaypointが必要です。")

        # [deg] → [rad]
        wp_rad = np.radians(np.array(waypoint_deg_list, dtype=np.float32))

        # 時刻ノット: 0 ~ total_time を waypoint 数で等分
        t_knots = np.linspace(0.0, total_time, len(wp_rad))

        # B-spline の次数 k (制御点数-1 以下)
        k = min(3, len(wp_rad) - 1)

        self.ref_spline = make_interp_spline(t_knots, wp_rad, k=k)
        self.ref_T = float(total_time)

        print(f"[Ref] B-spline reference built: "
              f"waypoints(deg)={waypoint_deg_list}, T={self.ref_T:.3f}s, k={k}")

    
    def enforce_constraints(self, p1, p2, p1_prev, p2_prev, dt):
        """物理制約を適用"""
        # Rate limit
        dp_max_step = self.dp_max * dt
        p1 = np.clip(p1, p1_prev - dp_max_step, p1_prev + dp_max_step)
        p2 = np.clip(p2, p2_prev - dp_max_step, p2_prev + dp_max_step)
        # Box constraint
        p1 = np.clip(p1, 0.0, self.p_max)
        p2 = np.clip(p2, 0.0, self.p_max)
        return p1, p2
    
    def cost_function(self, theta, theta_ref, p1, p2, p1_prev, p2_prev,
                      dp1, dp2, k, H):
        """コスト関数"""
        # Tracking error
        err = theta_ref - theta
        cost = self.w_tracking * (err ** 2)
        
        # Terminal cost
        if k == H - 1:
            cost += self.w_tracking * 0.5 * (err ** 2)
        
        # Smoothness
        cost += self.w_smooth * (dp1 ** 2 + dp2 ** 2)
        
        # Effort
        cost += self.w_effort * (p1 ** 2 + p2 ** 2)
        
        # Soft constraint violation
        viol = 0.0
        if p1 < 0:
            viol += (-p1) ** 2
        if p2 < 0:
            viol += (-p2) ** 2
        if p1 > self.p_max:
            viol += (p1 - self.p_max) ** 2
        if p2 > self.p_max:
            viol += (p2 - self.p_max) ** 2
        
        # Theta範囲外ペナルティ
        if theta < self.theta_min:
            viol += (self.theta_min - theta) ** 2
        if theta > self.theta_max:
            viol += (theta - self.theta_max) ** 2
        
        cost += self.w_constraint * viol
        return cost
    
    def rollout_batch(self, theta0, p1_0, p2_0, U, theta_ref_traj):
        """バッチ推論によるroll-out"""
        K, H = U.shape[0], U.shape[1]
        dt = self.sim_dt
        
        theta_seq = np.zeros((K, H), dtype=np.float32)
        p1_seq = np.zeros((K, H), dtype=np.float32)
        p2_seq = np.zeros((K, H), dtype=np.float32)
        
        # 初期状態
        theta_k = np.full(K, theta0, dtype=np.float32)
        p1_k = np.full(K, p1_0, dtype=np.float32)
        p2_k = np.full(K, p2_0, dtype=np.float32)
        
        # 履歴を複製
        theta_hist0 = list(self.hist_theta)[:self.lags]
        p1_hist0 = list(self.hist_p1_cmd)[:self.lags]
        p2_hist0 = list(self.hist_p2_cmd)[:self.lags]
        dp1_hist0 = list(self.hist_dp1_dt)[:self.lags]
        dp2_hist0 = list(self.hist_dp2_dt)[:self.lags]
        
        def pad_hist(hist, fill):
            if len(hist) < self.lags:
                hist = hist + [hist[-1] if hist else fill] * (self.lags - len(hist))
            return hist[:self.lags]
        
        theta_hist0 = pad_hist(theta_hist0, theta0)
        p1_hist0 = pad_hist(p1_hist0, p1_0)
        p2_hist0 = pad_hist(p2_hist0, p2_0)
        dp1_hist0 = pad_hist(dp1_hist0, 0.0)
        dp2_hist0 = pad_hist(dp2_hist0, 0.0)
        
        # shape: (K, lags)
        theta_hist = np.tile(np.array(theta_hist0, dtype=np.float32), (K, 1))
        p1_hist = np.tile(np.array(p1_hist0, dtype=np.float32), (K, 1))
        p2_hist = np.tile(np.array(p2_hist0, dtype=np.float32), (K, 1))
        dp1_hist = np.tile(np.array(dp1_hist0, dtype=np.float32), (K, 1))
        dp2_hist = np.tile(np.array(dp2_hist0, dtype=np.float32), (K, 1))
        
        for h in range(H):
            # Apply control
            dp1 = U[:, h, 0]
            dp2 = U[:, h, 1]
            
            p1_prev = p1_k.copy()
            p2_prev = p2_k.copy()
            
            p1_k = p1_k + dp1
            p2_k = p2_k + dp2
            
            # 制約適用
            for i in range(K):
                p1_k[i], p2_k[i] = self.enforce_constraints(
                    p1_k[i], p2_k[i], p1_prev[i], p2_prev[i], dt
                )
            
            # dp/dt を計算
            dp1_dt = (p1_k - p1_prev) / dt
            dp2_dt = (p2_k - p2_prev) / dt
            
            # 履歴を更新
            theta_hist = np.concatenate(
                [theta_k[:, None], theta_hist[:, :-1]], axis=1
            )
            p1_hist = np.concatenate(
                [p1_k[:, None], p1_hist[:, :-1]], axis=1
            )
            p2_hist = np.concatenate(
                [p2_k[:, None], p2_hist[:, :-1]], axis=1
            )
            dp1_hist = np.concatenate(
                [dp1_dt[:, None], dp1_hist[:, :-1]], axis=1
            )
            dp2_hist = np.concatenate(
                [dp2_dt[:, None], dp2_hist[:, :-1]], axis=1
            )
            
            # NARX用特徴量を構築
            X_chunks = []
            for k in range(self.lags):
                X_chunks.append(theta_hist[:, k][:, None])
                X_chunks.append(p1_hist[:, k][:, None])
                X_chunks.append(p2_hist[:, k][:, None])
                X_chunks.append(dp1_hist[:, k][:, None])
                X_chunks.append(dp2_hist[:, k][:, None])
            
            X_batch = np.concatenate(X_chunks, axis=1).astype(np.float32)
            
            # 正規化
            X_norm = (X_batch - self.mu) / (self.std + 1e-8)
            
            # バッチ推論
            with torch.no_grad():
                Y_batch = self.model(torch.from_numpy(X_norm).to(self.device))
            theta_k = Y_batch.cpu().numpy().flatten()
            
            # Clamp theta to training range
            theta_k = np.clip(theta_k, self.theta_min, self.theta_max)
            
            # 保存
            theta_seq[:, h] = theta_k
            p1_seq[:, h] = p1_k
            p2_seq[:, h] = p2_k
        
        return theta_seq, p1_seq, p2_seq
    
    def mppi_step(self, theta_ref_traj):
        """MPPI制御ステップ"""
        theta = self.theta_rad
        p1_prev = self.p1_cmd
        p2_prev = self.p2_cmd
        
        # 制御ノイズサンプル
        U = np.random.normal(
            loc=0.0,
            scale=self.sigma_u,
            size=(self.K, self.H, 2)
        ).astype(np.float32)
        
        # Rollout
        theta_seq, p1_seq, p2_seq = self.rollout_batch(
            theta, p1_prev, p2_prev, U, theta_ref_traj
        )
        
        # コスト計算
        dt = self.sim_dt
        J = np.zeros(self.K, dtype=np.float32)
        
        for i in range(self.K):
            cost = 0.0
            p1_h, p2_h = p1_prev, p2_prev
            for h in range(self.H):
                dp1 = U[i, h, 0]
                dp2 = U[i, h, 1]
                cost += self.cost_function(
                    theta_seq[i, h], theta_ref_traj[h],
                    p1_seq[i, h], p2_seq[i, h],
                    p1_h, p2_h, dp1, dp2, h, self.H
                )
                p1_h = p1_seq[i, h]
                p2_h = p2_seq[i, h]
            J[i] = cost
        
        # MPPI weight computation
        beta = np.min(J)
        w = np.exp(-(J - beta) / max(1e-6, self.temperature))
        w_sum = np.sum(w) + 1e-9
        
        # 重み付き平均
        dU = np.sum(w[:, None, None] * U, axis=0) / w_sum
        
        # Apply first control
        dp1_cmd, dp2_cmd = dU[0, 0], dU[0, 1]
        p1_cmd = p1_prev + dp1_cmd
        p2_cmd = p2_prev + dp2_cmd
        
        # Final constraints
        p1_cmd, p2_cmd = self.enforce_constraints(
            p1_cmd, p2_cmd, p1_prev, p2_prev, dt
        )
        
        return p1_cmd, p2_cmd, float(np.min(J)), float(np.mean(J))
    
    def simulate_step(self, p1_cmd, p2_cmd):
        """シミュレーションステップ（実機の代わり）"""
        dt = self.sim_dt
        
        # Update history
        self.p1_cmd = p1_cmd
        self.p2_cmd = p2_cmd
        self.hist_p1_cmd.appendleft(p1_cmd)
        self.hist_p2_cmd.appendleft(p2_cmd)
        
        if len(list(self.hist_p1_cmd)) > 1:
            dp1_dt = (self.hist_p1_cmd[0] - self.hist_p1_cmd[1]) / dt
            dp2_dt = (self.hist_p2_cmd[0] - self.hist_p2_cmd[1]) / dt
        else:
            dp1_dt, dp2_dt = 0.0, 0.0
        
        self.hist_dp1_dt.appendleft(dp1_dt)
        self.hist_dp2_dt.appendleft(dp2_dt)
        
        # NARXモデルで次の状態を予測
        X_chunks = []
        for k in range(self.lags):
            X_chunks.extend([
                list(self.hist_theta)[k] if k < len(list(self.hist_theta)) else self.theta_rad,
                list(self.hist_p1_cmd)[k] if k < len(list(self.hist_p1_cmd)) else p1_cmd,
                list(self.hist_p2_cmd)[k] if k < len(list(self.hist_p2_cmd)) else p2_cmd,
                list(self.hist_dp1_dt)[k] if k < len(list(self.hist_dp1_dt)) else 0.0,
                list(self.hist_dp2_dt)[k] if k < len(list(self.hist_dp2_dt)) else 0.0,
            ])
        
        X = np.array(X_chunks, dtype=np.float32).reshape(1, -1)
        X_norm = (X - self.mu) / (self.std + 1e-8)
        
        with torch.no_grad():
            theta_next = self.model(torch.from_numpy(X_norm).to(self.device))
        
        self.theta_rad = float(theta_next.cpu().numpy().item())
        self.theta_rad = np.clip(self.theta_rad, self.theta_min, self.theta_max)
        self.hist_theta.appendleft(self.theta_rad)
    
    def run_simulation(self, theta_target_rad=None, steps=100, smooth_ref=True):
        """シミュレーション実行（単一目標 or B-spline 参照）"""

        if self.ref_spline is None and theta_target_rad is None:
            raise ValueError("theta_target_rad か B-spline 参照のどちらかを指定してください。")

        if self.ref_spline is None:
            print(f"\n[Simulation] Target: {math.degrees(theta_target_rad):.2f} deg, Steps: {steps}")
        else:
            print(f"\n[Simulation] B-spline reference, Steps: {steps}")

        for step in range(steps):
            t = step * self.sim_dt  # 現在時刻

            # ===== 参照軌道の生成 =====
            if self.ref_spline is not None:
                # グローバル B-spline 参照 θ_ref(t)
                t_h = t + np.arange(self.H) * self.sim_dt
                # B-spline 定義域 [0, self.ref_T] からはみ出さないようにクリップ
                t_h_clip = np.clip(t_h, 0.0, self.ref_T)
                theta_ref_traj = self.ref_spline(t_h_clip)
                # 安全のため学習範囲にクリップ
                theta_ref_traj = np.clip(theta_ref_traj, self.theta_min, self.theta_max)
            else:
                # これまでと同じ: 単一ターゲットに対する局所S字 or ステップ
                if smooth_ref:
                    theta_ref_traj = self.generate_smooth_reference(
                        self.theta_rad, theta_target_rad, self.H
                    )
                else:
                    theta_ref_traj = np.full(self.H, theta_target_rad)

            # ===== MPPI control =====
            p1_cmd, p2_cmd, j_min, j_mean = self.mppi_step(theta_ref_traj)

            # ===== システム応答のシミュレーション =====
            self.simulate_step(p1_cmd, p2_cmd)

            # ===== Logging =====
            # 「単一ターゲット」の場合は theta_target_rad との差
            # B-spline の場合は「その時刻の参照」との差を見るようにする
            current_ref = theta_ref_traj[0]
            error = current_ref - self.theta_rad

            self.log_time.append(t)
            self.log_theta.append(self.theta_rad)
            self.log_theta_ref.append(current_ref)
            self.log_p1_cmd.append(p1_cmd)
            self.log_p2_cmd.append(p2_cmd)
            self.log_error.append(error)
            self.log_cost.append(j_min)

            if step % 10 == 0:
                print(f"[{step:03d}] t={t:.2f}s, "
                      f"theta={math.degrees(self.theta_rad):.2f}°, "
                      f"ref={math.degrees(current_ref):.2f}°, "
                      f"err={math.degrees(error):.2f}°, "
                      f"p1={p1_cmd:.3f}, p2={p2_cmd:.3f}, J_min={j_min:.3f}")

        # ===== Final stats =====
        errors_deg = np.degrees(np.array(self.log_error))
        print(f"\n[Results]")
        print(f"  Final theta: {math.degrees(self.theta_rad):.2f}°")
        print(f"  Final error (w.r.t last ref): {errors_deg[-1]:.2f}°")
        print(f"  RMS error: {np.sqrt(np.mean(errors_deg**2)):.2f}°")
        print(f"  Max abs error: {np.max(np.abs(errors_deg)):.2f}°")

    
    def generate_smooth_reference(self, theta0, theta_target, horizon):
        """S字カーブ参照軌道を生成"""
        t = np.linspace(0, 1, horizon)
        s = 3 * t**2 - 2 * t**3  # S字カーブ
        theta_ref = theta0 + (theta_target - theta0) * s
        # Clamp to training range
        theta_ref = np.clip(theta_ref, self.theta_min, self.theta_max)
        return theta_ref
    
    def plot_results(self):
        """結果をプロット"""
        if not _HAS_PLT:
            print("[Warning] matplotlib not available, skipping plots")
            return
        
        t = np.array(self.log_time)
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
        
        # Theta tracking
        ax = axes[0]
        ax.plot(t, np.degrees(self.log_theta), label='theta')
        ax.plot(t, np.degrees(self.log_theta_ref), '--', label='theta_ref')
        ax.set_ylabel('Angle [deg]')
        ax.set_title('Theta Tracking')
        ax.grid(True)
        ax.legend()
        
        # Control inputs
        ax = axes[1]
        ax.plot(t, self.log_p1_cmd, label='p1_cmd')
        ax.plot(t, self.log_p2_cmd, label='p2_cmd')
        ax.set_ylabel('Pressure [MPa]')
        ax.set_title('Control Inputs')
        ax.grid(True)
        ax.legend()
        
        # Tracking error
        ax = axes[2]
        ax.plot(t, np.degrees(self.log_error))
        ax.set_ylabel('Error [deg]')
        ax.set_title('Tracking Error')
        ax.grid(True)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Cost
        ax = axes[3]
        ax.plot(t, self.log_cost)
        ax.set_ylabel('Cost')
        ax.set_xlabel('Time [s]')
        ax.set_title('MPPI Cost (J_min)')
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename):
        """結果をCSVに保存"""
        df = pd.DataFrame({
            't[s]': self.log_time,
            'theta[rad]': self.log_theta,
            'theta_ref[rad]': self.log_theta_ref,
            'error[rad]': self.log_error,
            'p1_cmd[MPa]': self.log_p1_cmd,
            'p2_cmd[MPa]': self.log_p2_cmd,
            'cost': self.log_cost
        })
        df.to_csv(filename, index=False)
        print(f"[Save] Results saved to {filename}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing narx_model.pt and narx_meta.json')
    
    # Simulation
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of simulation steps')
    parser.add_argument('--dt', type=float, default=0.0,
                        help='Time step (0 to use model default)')
    parser.add_argument('--frame-skip', type=int, default=2,
                        help='Frame skip for control')
    
    # MPPI parameters
    parser.add_argument('--K', type=int, default=32,
                        help='Number of rollout samples')
    parser.add_argument('--horizon', type=int, default=15,
                        help='Prediction horizon')
    parser.add_argument('--lambda', dest='lam', type=float, default=2.0,
                        help='Temperature parameter')
    parser.add_argument('--sigma-u', type=float, default=0.10,
                        help='Control noise std [MPa]')
    
    # Cost weights
    parser.add_argument('--w-tracking', type=float, default=30.0,
                        help='Tracking cost weight')
    parser.add_argument('--w-smooth', type=float, default=0.05,
                        help='Smoothness cost weight')
    parser.add_argument('--w-effort', type=float, default=0.01,
                        help='Control effort weight')
    parser.add_argument('--w-constraint', type=float, default=500.0,
                        help='Constraint violation weight')
    
    # Physical limits
    parser.add_argument('--p-max', type=float, default=0.70,
                        help='Maximum pressure [MPa]')
    parser.add_argument('--dp-max', type=float, default=3.5,
                        help='Maximum pressure rate [MPa/s]')
    
    # Context
    parser.add_argument('--context-csv', type=str, default='',
                        help='CSV file for initial context')
    
    # Output
    parser.add_argument('--plot', action='store_true',
                        help='Plot results')
    parser.add_argument('--save', type=str, default='',
                        help='Save results to CSV')

    # Target
    parser.add_argument('--theta-target-deg', type=float, default=30.0,
                        help='Target angle in degrees (単一目標)')
    parser.add_argument('--theta-waypoints-deg', type=float, nargs='+',
                        help='B-spline 参照用のwaypoint [deg] (例: 20 -30 10)')

    
    args = parser.parse_args()
    
    # Create simulator
    sim = NARX_MPPI_Simulator(args)

    # Initialize context
    sim.initialize_from_context(args.context_csv)

    total_time = args.steps * sim.sim_dt

    # ===== 参照のモードを決める =====
    if args.theta_waypoints_deg is not None and len(args.theta_waypoints_deg) >= 2:
        # B-spline 参照
        sim.set_bspline_reference(args.theta_waypoints_deg, total_time)
        sim.run_simulation(theta_target_rad=None, steps=args.steps, smooth_ref=False)
    else:
        # 従来通りの単一目標
        theta_target_rad = math.radians(args.theta_target_deg)
        sim.run_simulation(theta_target_rad, args.steps, smooth_ref=True)
    
    # Plot results
    if args.plot:
        sim.plot_results()
    
    # Save results
    if args.save:
        sim.save_results(args.save)


if __name__ == '__main__':
    main()