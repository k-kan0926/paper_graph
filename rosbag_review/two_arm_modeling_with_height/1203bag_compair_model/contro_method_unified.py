#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
control_methods_unified.py
複数の制御手法を統一的にシミュレーション・評価

サポートする制御手法:
1. MPPI (Model Predictive Path Integral)
2. CEM (Cross-Entropy Method)
3. Random Shooting
4. Gradient-based MPC
5. PID (ベースライン)
"""
import os, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Tuple, Optional, Dict
import pandas as pd

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False

# ==================== Model Loading ====================

def load_model_and_meta(model_dir: str, device):
    """モデルとメタデータをロード"""
    meta_path = os.path.join(model_dir, 'meta.json')
    model_path = os.path.join(model_dir, 'model.pt')
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    model_type = meta['model_type']
    lags = meta['lags']
    n_features = meta['n_features']
    hidden = meta.get('hidden', 128)
    dropout = meta.get('dropout', 0.0)
    
    # Create model
    if model_type == 'linear_arx':
        from train_models_unified import LinearARX
        model = LinearARX(lags, n_features)
    
    elif model_type == 'narx':
        from train_models_unified import MLP_NARX
        model = MLP_NARX(lags, n_features, hidden=[hidden, hidden], dropout=dropout)
    
    elif model_type == 'lstm':
        from train_models_unified import LSTM_Model
        model = LSTM_Model(n_features, hidden=hidden, num_layers=2, dropout=dropout)
    
    elif model_type == 'gru':
        from train_models_unified import GRU_Model
        model = GRU_Model(n_features, hidden=hidden, num_layers=2, dropout=dropout)
    
    elif model_type == 'transformer':
        from train_models_unified import TransformerModel
        model = TransformerModel(n_features, d_model=64, nhead=4, num_layers=2, dropout=dropout)
    
    elif model_type == 'cnn':
        from train_models_unified import CNN_Model
        model = CNN_Model(n_features, hidden=hidden, dropout=dropout)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"[Model] Loaded {model_type}: lags={lags}, features={n_features}")
    
    return model, meta

# ==================== Base Controller Class ====================

class BaseController:
    """制御器のベースクラス"""
    
    def __init__(self, model, meta, dt, device):
        self.model = model
        self.meta = meta
        self.dt = dt
        self.device = device
        
        self.lags = meta['lags']
        self.n_features = meta['n_features']
        self.feat_cols = meta['feature_names']
        
        # State variables
        self.theta_rad = 0.0
        self.p1_cmd = 0.0
        self.p2_cmd = 0.0
        
        # History buffers
        maxlen = self.lags + 10
        self.hist_theta = deque([0.0] * maxlen, maxlen=maxlen)
        self.hist_p1 = deque([0.0] * maxlen, maxlen=maxlen)
        self.hist_p2 = deque([0.0] * maxlen, maxlen=maxlen)
        self.hist_dp1 = deque([0.0] * maxlen, maxlen=maxlen)
        self.hist_dp2 = deque([0.0] * maxlen, maxlen=maxlen)
        
        # Physical limits
        self.p_max = 0.70  # MPa
        self.dp_max = 3.5  # MPa/s
        
        # Logging
        self.log_time = []
        self.log_theta = []
        self.log_theta_ref = []
        self.log_p1 = []
        self.log_p2 = []
        self.log_error = []
        self.log_cost = []
    
    def enforce_constraints(self, p1, p2, p1_prev, p2_prev):
        """物理制約適用"""
        # Rate limit
        dp_max_step = self.dp_max * self.dt
        p1 = np.clip(p1, p1_prev - dp_max_step, p1_prev + dp_max_step)
        p2 = np.clip(p2, p2_prev - dp_max_step, p2_prev + dp_max_step)
        # Box constraint
        p1 = np.clip(p1, 0.0, self.p_max)
        p2 = np.clip(p2, 0.0, self.p_max)
        return p1, p2
    
    def predict_step(self, theta, p1, p2, dp1, dp2):
        """1ステップ予測"""
        # Update history
        self.hist_theta.appendleft(theta)
        self.hist_p1.appendleft(p1)
        self.hist_p2.appendleft(p2)
        self.hist_dp1.appendleft(dp1)
        self.hist_dp2.appendleft(dp2)
        
        # Build feature vector
        x_list = []
        for k in range(self.lags):
            x_list.append([
                list(self.hist_theta)[k],
                list(self.hist_p1)[k],
                list(self.hist_p2)[k],
                list(self.hist_dp1)[k],
                list(self.hist_dp2)[k]
            ])
        
        x = np.array(x_list, dtype=np.float32)  # (lags, n_features)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1, lags, n_features)
        
        with torch.no_grad():
            theta_next = self.model(x).cpu().numpy().item()
        
        return float(theta_next)
    
    def compute_control(self, theta_ref):
        """制御入力計算 (サブクラスで実装)"""
        raise NotImplementedError
    
    def step(self, theta_ref):
        """シミュレーションステップ"""
        # Compute control
        p1_cmd, p2_cmd, cost = self.compute_control(theta_ref)
        
        # Apply constraints
        p1_cmd, p2_cmd = self.enforce_constraints(
            p1_cmd, p2_cmd, self.p1_cmd, self.p2_cmd
        )
        
        # Compute derivatives
        dp1 = (p1_cmd - self.p1_cmd) / self.dt
        dp2 = (p2_cmd - self.p2_cmd) / self.dt
        
        # Predict next state
        theta_next = self.predict_step(
            self.theta_rad, p1_cmd, p2_cmd, dp1, dp2
        )
        
        # Update state
        self.theta_rad = theta_next
        self.p1_cmd = p1_cmd
        self.p2_cmd = p2_cmd
        
        return theta_next, p1_cmd, p2_cmd, cost

# ==================== MPPI Controller ====================

class MPPI_Controller(BaseController):
    """MPPIコントローラー"""
    
    def __init__(self, model, meta, dt, device, K=32, H=15, lam=2.0, sigma_u=0.10):
        super().__init__(model, meta, dt, device)
        self.K = K
        self.H = H
        self.temperature = lam
        self.sigma_u = sigma_u
        
        # Cost weights
        self.w_track = 30.0
        self.w_smooth = 0.05
        self.w_effort = 0.01
    
    def compute_control(self, theta_ref):
        """MPPI制御"""
        # Sample control noise
        U = np.random.normal(0, self.sigma_u, (self.K, self.H, 2)).astype(np.float32)
        
        # Rollout
        J = np.zeros(self.K)
        
        for k in range(self.K):
            cost = 0.0
            theta = self.theta_rad
            p1, p2 = self.p1_cmd, self.p2_cmd
            
            # Copy history for this rollout
            hist_theta = list(self.hist_theta)[:self.lags]
            hist_p1 = list(self.hist_p1)[:self.lags]
            hist_p2 = list(self.hist_p2)[:self.lags]
            hist_dp1 = list(self.hist_dp1)[:self.lags]
            hist_dp2 = list(self.hist_dp2)[:self.lags]
            
            for h in range(self.H):
                dp1, dp2 = U[k, h, 0], U[k, h, 1]
                p1_prev, p2_prev = p1, p2
                
                p1 = p1 + dp1
                p2 = p2 + dp2
                p1, p2 = self.enforce_constraints(p1, p2, p1_prev, p2_prev)
                
                # Build features
                hist_theta = [theta] + hist_theta[:self.lags-1]
                hist_p1 = [p1] + hist_p1[:self.lags-1]
                hist_p2 = [p2] + hist_p2[:self.lags-1]
                hist_dp1 = [(p1-p1_prev)/self.dt] + hist_dp1[:self.lags-1]
                hist_dp2 = [(p2-p2_prev)/self.dt] + hist_dp2[:self.lags-1]
                
                x_list = []
                for i in range(self.lags):
                    x_list.append([hist_theta[i], hist_p1[i], hist_p2[i],
                                  hist_dp1[i], hist_dp2[i]])
                
                x = np.array(x_list, dtype=np.float32)
                x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    theta = self.model(x_t).cpu().numpy().item()
                
                # Cost
                err = theta_ref - theta
                cost += self.w_track * err**2
                cost += self.w_smooth * (dp1**2 + dp2**2)
                cost += self.w_effort * (p1**2 + p2**2)
            
            J[k] = cost
        
        # MPPI weighting
        beta = np.min(J)
        w = np.exp(-(J - beta) / max(1e-6, self.temperature))
        w = w / (np.sum(w) + 1e-9)
        
        # Weighted average
        dU = np.sum(w[:, None, None] * U, axis=0)
        
        # Apply first control
        p1_cmd = self.p1_cmd + dU[0, 0]
        p2_cmd = self.p2_cmd + dU[0, 1]
        
        return p1_cmd, p2_cmd, float(np.min(J))

# ==================== CEM Controller ====================

class CEM_Controller(BaseController):
    """Cross-Entropy Method コントローラー"""
    
    def __init__(self, model, meta, dt, device, K=64, H=15, elite_frac=0.2, n_iter=3):
        super().__init__(model, meta, dt, device)
        self.K = K
        self.H = H
        self.elite_frac = elite_frac
        self.n_iter = n_iter
        
        self.w_track = 30.0
        self.w_smooth = 0.05
        self.w_effort = 0.01
        
        # Initial distribution
        self.mean = np.zeros((H, 2), dtype=np.float32)
        self.std = np.ones((H, 2), dtype=np.float32) * 0.1
    
    def compute_control(self, theta_ref):
        """CEM制御"""
        mean = self.mean.copy()
        std = self.std.copy()
        
        for _ in range(self.n_iter):
            # Sample
            U = np.random.normal(mean[None, :, :], std[None, :, :],
                                (self.K, self.H, 2)).astype(np.float32)
            
            # Evaluate
            J = np.zeros(self.K)
            
            for k in range(self.K):
                cost = 0.0
                theta = self.theta_rad
                p1, p2 = self.p1_cmd, self.p2_cmd
                
                hist_theta = list(self.hist_theta)[:self.lags]
                hist_p1 = list(self.hist_p1)[:self.lags]
                hist_p2 = list(self.hist_p2)[:self.lags]
                hist_dp1 = list(self.hist_dp1)[:self.lags]
                hist_dp2 = list(self.hist_dp2)[:self.lags]
                
                for h in range(self.H):
                    dp1, dp2 = U[k, h, 0], U[k, h, 1]
                    p1_prev, p2_prev = p1, p2
                    
                    p1 = p1 + dp1
                    p2 = p2 + dp2
                    p1, p2 = self.enforce_constraints(p1, p2, p1_prev, p2_prev)
                    
                    hist_theta = [theta] + hist_theta[:self.lags-1]
                    hist_p1 = [p1] + hist_p1[:self.lags-1]
                    hist_p2 = [p2] + hist_p2[:self.lags-1]
                    hist_dp1 = [(p1-p1_prev)/self.dt] + hist_dp1[:self.lags-1]
                    hist_dp2 = [(p2-p2_prev)/self.dt] + hist_dp2[:self.lags-1]
                    
                    x_list = []
                    for i in range(self.lags):
                        x_list.append([hist_theta[i], hist_p1[i], hist_p2[i],
                                      hist_dp1[i], hist_dp2[i]])
                    
                    x = np.array(x_list, dtype=np.float32)
                    x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        theta = self.model(x_t).cpu().numpy().item()
                    
                    err = theta_ref - theta
                    cost += self.w_track * err**2
                    cost += self.w_smooth * (dp1**2 + dp2**2)
                    cost += self.w_effort * (p1**2 + p2**2)
                
                J[k] = cost
            
            # Select elites
            n_elite = max(1, int(self.K * self.elite_frac))
            elite_idx = np.argsort(J)[:n_elite]
            elite_U = U[elite_idx]
            
            # Update distribution
            mean = np.mean(elite_U, axis=0)
            std = np.std(elite_U, axis=0) + 1e-3
        
        # Use mean of final distribution
        p1_cmd = self.p1_cmd + mean[0, 0]
        p2_cmd = self.p2_cmd + mean[0, 1]
        
        self.mean = mean
        self.std = std
        
        return p1_cmd, p2_cmd, float(np.min(J))

# ==================== Random Shooting Controller ====================

class RandomShooting_Controller(BaseController):
    """Random Shootingコントローラー"""
    
    def __init__(self, model, meta, dt, device, K=64, H=15, sigma_u=0.15):
        super().__init__(model, meta, dt, device)
        self.K = K
        self.H = H
        self.sigma_u = sigma_u
        
        self.w_track = 30.0
        self.w_smooth = 0.05
        self.w_effort = 0.01
    
    def compute_control(self, theta_ref):
        """Random Shooting制御"""
        # Sample random controls
        U = np.random.normal(0, self.sigma_u, (self.K, self.H, 2)).astype(np.float32)
        
        # Evaluate
        J = np.zeros(self.K)
        
        for k in range(self.K):
            cost = 0.0
            theta = self.theta_rad
            p1, p2 = self.p1_cmd, self.p2_cmd
            
            hist_theta = list(self.hist_theta)[:self.lags]
            hist_p1 = list(self.hist_p1)[:self.lags]
            hist_p2 = list(self.hist_p2)[:self.lags]
            hist_dp1 = list(self.hist_dp1)[:self.lags]
            hist_dp2 = list(self.hist_dp2)[:self.lags]
            
            for h in range(self.H):
                dp1, dp2 = U[k, h, 0], U[k, h, 1]
                p1_prev, p2_prev = p1, p2
                
                p1 = p1 + dp1
                p2 = p2 + dp2
                p1, p2 = self.enforce_constraints(p1, p2, p1_prev, p2_prev)
                
                hist_theta = [theta] + hist_theta[:self.lags-1]
                hist_p1 = [p1] + hist_p1[:self.lags-1]
                hist_p2 = [p2] + hist_p2[:self.lags-1]
                hist_dp1 = [(p1-p1_prev)/self.dt] + hist_dp1[:self.lags-1]
                hist_dp2 = [(p2-p2_prev)/self.dt] + hist_dp2[:self.lags-1]
                
                x_list = []
                for i in range(self.lags):
                    x_list.append([hist_theta[i], hist_p1[i], hist_p2[i],
                                  hist_dp1[i], hist_dp2[i]])
                
                x = np.array(x_list, dtype=np.float32)
                x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    theta = self.model(x_t).cpu().numpy().item()
                
                err = theta_ref - theta
                cost += self.w_track * err**2
                cost += self.w_smooth * (dp1**2 + dp2**2)
                cost += self.w_effort * (p1**2 + p2**2)
            
            J[k] = cost
        
        # Select best
        best_idx = np.argmin(J)
        dU = U[best_idx]
        
        p1_cmd = self.p1_cmd + dU[0, 0]
        p2_cmd = self.p2_cmd + dU[0, 1]
        
        return p1_cmd, p2_cmd, float(np.min(J))

# ==================== PID Controller ====================

class PID_Controller(BaseController):
    """PIDコントローラー (ベースライン)"""
    
    def __init__(self, model, meta, dt, device, Kp=0.5, Ki=0.05, Kd=0.1):
        super().__init__(model, meta, dt, device)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.integral = 0.0
        self.prev_error = 0.0
    
    def compute_control(self, theta_ref):
        """PID制御"""
        error = theta_ref - self.theta_rad
        
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        
        # PID output
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        # Map to pressure commands (heuristic)
        if u > 0:
            p1_cmd = self.p1_cmd + abs(u) * 0.1
            p2_cmd = max(0, self.p2_cmd - abs(u) * 0.1)
        else:
            p1_cmd = max(0, self.p1_cmd - abs(u) * 0.1)
            p2_cmd = self.p2_cmd + abs(u) * 0.1
        
        self.prev_error = error
        
        return p1_cmd, p2_cmd, abs(error)

# ==================== Controller Factory ====================

def create_controller(controller_type, model, meta, dt, device, args):
    """コントローラー生成"""
    if controller_type == 'mppi':
        return MPPI_Controller(model, meta, dt, device,
                              K=args.K, H=args.horizon,
                              lam=args.lam, sigma_u=args.sigma_u)
    
    elif controller_type == 'cem':
        return CEM_Controller(model, meta, dt, device,
                             K=args.K, H=args.horizon,
                             elite_frac=0.2, n_iter=3)
    
    elif controller_type == 'random_shooting':
        return RandomShooting_Controller(model, meta, dt, device,
                                        K=args.K, H=args.horizon,
                                        sigma_u=args.sigma_u)
    
    elif controller_type == 'pid':
        return PID_Controller(model, meta, dt, device,
                             Kp=0.5, Ki=0.05, Kd=0.1)
    
    else:
        raise ValueError(f"Unknown controller: {controller_type}")

# ==================== Simulation ====================

def run_simulation(controller, theta_target_deg, steps):
    """シミュレーション実行"""
    theta_target = math.radians(theta_target_deg)
    
    print(f"\n[Simulation] Target: {theta_target_deg:.1f}°, Steps: {steps}")
    
    t = 0.0
    for step in range(steps):
        theta, p1, p2, cost = controller.step(theta_target)
        
        error = theta_target - theta
        controller.log_time.append(t)
        controller.log_theta.append(theta)
        controller.log_theta_ref.append(theta_target)
        controller.log_p1.append(p1)
        controller.log_p2.append(p2)
        controller.log_error.append(error)
        controller.log_cost.append(cost)
        
        if step % 20 == 0:
            print(f"  [{step:03d}] theta={math.degrees(theta):.2f}°, "
                  f"err={math.degrees(error):.2f}°, "
                  f"p1={p1:.3f}, p2={p2:.3f}")
        
        t += controller.dt
    
    # Metrics
    errors_deg = np.degrees(np.array(controller.log_error))
    print(f"\n[Results]")
    print(f"  Final theta: {math.degrees(controller.log_theta[-1]):.2f}°")
    print(f"  Final error: {errors_deg[-1]:.2f}°")
    print(f"  RMS error: {np.sqrt(np.mean(errors_deg**2)):.2f}°")
    print(f"  Max abs error: {np.max(np.abs(errors_deg)):.2f}°")
    print(f"  Mean cost: {np.mean(controller.log_cost):.3f}")

# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument('--model_dir', type=str, required=True)
    
    # Controller
    parser.add_argument('--controllers', nargs='+',
                       default=['mppi', 'cem', 'random_shooting', 'pid'],
                       choices=['mppi', 'cem', 'random_shooting', 'pid'])
    
    # Simulation
    parser.add_argument('--theta_target_deg', type=float, default=30.0)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.01)
    
    # Control parameters
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--lam', type=float, default=2.0)
    parser.add_argument('--sigma_u', type=float, default=0.10)
    
    # Output
    parser.add_argument('--out_dir', type=str, default='control_comparison')
    parser.add_argument('--plot', action='store_true')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f" Control Methods Comparison")
    print(f"{'='*70}")
    print(f"Model: {args.model_dir}")
    print(f"Controllers: {args.controllers}")
    print(f"Device: {device}")
    
    # Load model
    model, meta = load_model_and_meta(args.model_dir, device)
    
    # Run simulations
    results = {}
    
    for ctrl_type in args.controllers:
        print(f"\n{'='*70}")
        print(f" Controller: {ctrl_type.upper()}")
        print(f"{'='*70}")
        
        controller = create_controller(ctrl_type, model, meta, args.dt, device, args)
        run_simulation(controller, args.theta_target_deg, args.steps)
        
        # Save results
        ctrl_dir = os.path.join(args.out_dir, ctrl_type)
        os.makedirs(ctrl_dir, exist_ok=True)
        
        df = pd.DataFrame({
            't[s]': controller.log_time,
            'theta[rad]': controller.log_theta,
            'theta_ref[rad]': controller.log_theta_ref,
            'error[rad]': controller.log_error,
            'p1[MPa]': controller.log_p1,
            'p2[MPa]': controller.log_p2,
            'cost': controller.log_cost
        })
        df.to_csv(os.path.join(ctrl_dir, 'simulation.csv'), index=False)
        
        errors_deg = np.degrees(np.array(controller.log_error))
        results[ctrl_type] = {
            'rmse': float(np.sqrt(np.mean(errors_deg**2))),
            'mae': float(np.mean(np.abs(errors_deg))),
            'max_abs_error': float(np.max(np.abs(errors_deg))),
            'final_error': float(errors_deg[-1]),
            'mean_cost': float(np.mean(controller.log_cost))
        }
    
    # Summary
    print(f"\n{'='*70}")
    print(f" SUMMARY")
    print(f"{'='*70}")
    print(f"{'Controller':<20} {'RMSE(°)':<10} {'MAE(°)':<10} {'Max Err(°)':<12} {'Mean Cost':<12}")
    print(f"{'-'*70}")
    
    for ctrl_type in args.controllers:
        r = results[ctrl_type]
        print(f"{ctrl_type:<20} {r['rmse']:<10.3f} {r['mae']:<10.3f} "
              f"{r['max_abs_error']:<12.3f} {r['mean_cost']:<12.3f}")
    
    # Save summary
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Saved] {args.out_dir}/")

if __name__ == '__main__':
    main()