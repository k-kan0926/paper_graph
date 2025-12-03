#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
control_lstm_ss.py
LSTM-SSモデル用制御スクリプト

制御手法:
1. MPPI - サンプリングベース
2. Gradient-based MPC - 勾配降下
3. Random Shooting

LSTMの特徴:
- cell state + hidden state の2つの状態
- より長期の記憶能力
- GRUより計算コストが高いが表現力も高い
"""
import os, json, math, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except:
    _HAS_PLT = False

# ==================== Model Definition ====================
class LSTM_SS(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, u_seq, h0=None, c0=None):
        if h0 is not None and c0 is not None:
            h_seq, (h_final, c_final) = self.lstm(u_seq, (h0, c0))
        else:
            h_seq, (h_final, c_final) = self.lstm(u_seq)
        
        batch_size, seq_len, _ = h_seq.shape
        h_flat = h_seq.reshape(batch_size * seq_len, self.hidden_dim)
        
        theta_flat = self.decoder(h_flat)
        theta = theta_flat.reshape(batch_size, seq_len, 1)
        
        return theta, h_final, c_final
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return h0, c0

# ==================== Base Controller ====================
class BaseLSTMController:
    def __init__(self, model_dir, dt=0.01, device='cpu'):
        self.load_model(model_dir, device)
        self.dt = dt
        self.device = device
        
        self.p_max = 0.70
        self.dp_max = 3.5
        
        self.theta_rad = 0.0
        self.p1_cmd = 0.0
        self.p2_cmd = 0.0
        
        # LSTM states (hidden + cell)
        self.h, self.c = self.model.init_hidden(1, device)
        
        self.log = {
            't': [], 'theta': [], 'theta_ref': [], 'error': [],
            'p1': [], 'p2': [], 'cost': [], 'comp_time': []
        }
    
    def load_model(self, model_dir, device):
        with open(os.path.join(model_dir, 'lstm_ss_meta.json'), 'r') as f:
            self.meta = json.load(f)
        
        self.hidden_dim = self.meta['hidden_dim']
        self.num_layers = self.meta['num_layers']
        
        self.model = LSTM_SS(
            input_dim=2,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, 'lstm_ss_model.pt'),
                      map_location=device)
        )
        self.model.to(device)
        self.model.eval()
        
        print(f"[Model] Loaded LSTM-SS: hidden={self.hidden_dim}, layers={self.num_layers}")
    
    def enforce_constraints(self, p1, p2, p1_prev, p2_prev):
        dp_max_step = self.dp_max * self.dt
        p1 = np.clip(p1, p1_prev - dp_max_step, p1_prev + dp_max_step)
        p2 = np.clip(p2, p2_prev - dp_max_step, p2_prev + dp_max_step)
        p1 = np.clip(p1, 0.0, self.p_max)
        p2 = np.clip(p2, 0.0, self.p_max)
        return p1, p2
    
    def predict_step(self, p1, p2, h=None, c=None):
        """1ステップ予測"""
        u = torch.tensor([[p1, p2]], dtype=torch.float32, device=self.device)
        u = u.unsqueeze(1)  # (1, 1, 2) - batch, seq_len=1, input_dim
        
        if h is None:
            h = self.h
        if c is None:
            c = self.c
        
        with torch.no_grad():
            theta, h_new, c_new = self.model(u, h, c)
        
        theta_val = float(theta[0, 0, 0].cpu().numpy())
        
        return theta_val, h_new, c_new
    
    def compute_control(self, theta_ref):
        raise NotImplementedError
    
    def step(self, theta_ref):
        t_start = time.time()
        
        p1_cmd, p2_cmd, cost = self.compute_control(theta_ref)
        p1_cmd, p2_cmd = self.enforce_constraints(
            p1_cmd, p2_cmd, self.p1_cmd, self.p2_cmd
        )
        
        # Predict next state
        theta_next, h_new, c_new = self.predict_step(p1_cmd, p2_cmd, self.h, self.c)
        
        self.theta_rad = theta_next
        self.p1_cmd = p1_cmd
        self.p2_cmd = p2_cmd
        self.h = h_new
        self.c = c_new
        
        comp_time = time.time() - t_start
        
        return theta_next, p1_cmd, p2_cmd, cost, comp_time

# ==================== 1. MPPI Controller ====================
class LSTM_MPPI(BaseLSTMController):
    def __init__(self, model_dir, dt=0.01, device='cpu',
                 K=32, H=15, lam=2.0, sigma_u=0.10):
        super().__init__(model_dir, dt, device)
        self.K = K
        self.H = H
        self.temperature = lam
        self.sigma_u = sigma_u
        
        print(f"[MPPI] K={K}, H={H}")
    
    def compute_control(self, theta_ref):
        """MPPI制御"""
        U = np.random.normal(0, self.sigma_u, (self.K, self.H, 2)).astype(np.float32)
        J = np.zeros(self.K)
        
        for k in range(self.K):
            cost = 0.0
            h = self.h.clone()
            c = self.c.clone()
            
            for h_idx in range(self.H):
                p1 = self.p1_cmd + U[k, h_idx, 0]
                p2 = self.p2_cmd + U[k, h_idx, 1]
                
                p1 = np.clip(p1, 0, self.p_max)
                p2 = np.clip(p2, 0, self.p_max)
                
                # Predict
                theta, h, c = self.predict_step(p1, p2, h, c)
                
                err = theta_ref - theta
                cost += 30.0 * err**2 + 0.01 * (p1**2 + p2**2)
            
            J[k] = cost
        
        beta = np.min(J)
        w = np.exp(-(J - beta) / max(1e-6, self.temperature))
        w = w / (np.sum(w) + 1e-9)
        
        dU = np.sum(w[:, None, None] * U, axis=0)
        
        return self.p1_cmd + dU[0, 0], self.p2_cmd + dU[0, 1], float(np.min(J))

# ==================== 2. Gradient-based MPC ====================
class LSTM_GradientMPC(BaseLSTMController):
    """勾配降下でMPC"""
    
    def __init__(self, model_dir, dt=0.01, device='cpu',
                 horizon=10, n_iter=50, lr=0.01):
        super().__init__(model_dir, dt, device)
        self.horizon = horizon
        self.n_iter = n_iter
        self.lr = lr
        
        print(f"[GradientMPC] H={horizon}, n_iter={n_iter}")
    
    def compute_control(self, theta_ref):
        """勾配降下でMPCを解く"""
        H = self.horizon
        
        # Initialize control sequence
        u_seq = torch.tensor(
            [[self.p1_cmd, self.p2_cmd]] * H,
            dtype=torch.float32,
            requires_grad=True,
            device=self.device
        )
        u_seq = u_seq.unsqueeze(0)  # (1, H, 2)
        
        optimizer = torch.optim.Adam([u_seq], lr=self.lr)
        
        for _ in range(self.n_iter):
            optimizer.zero_grad()
            
            # Forward pass
            theta_seq, _, _ = self.model(u_seq, self.h, self.c)  # (1, H, 1)
            theta_seq = theta_seq.squeeze(0).squeeze(-1)  # (H,)
            
            # Cost
            tracking_error = (theta_seq - theta_ref)**2
            control_effort = torch.sum(u_seq**2)
            
            cost = 30.0 * torch.sum(tracking_error) + 0.01 * control_effort
            
            # Physical constraints (soft)
            cost += 100.0 * torch.sum(torch.relu(u_seq - self.p_max)**2)
            cost += 100.0 * torch.sum(torch.relu(-u_seq)**2)
            
            cost.backward()
            optimizer.step()
        
        u_opt = u_seq.detach().cpu().numpy()[0]
        p1_cmd = float(u_opt[0, 0])
        p2_cmd = float(u_opt[0, 1])
        
        return p1_cmd, p2_cmd, float(cost.item())

# ==================== 3. Random Shooting ====================
class LSTM_RandomShooting(BaseLSTMController):
    def __init__(self, model_dir, dt=0.01, device='cpu',
                 K=64, H=15, sigma_u=0.15):
        super().__init__(model_dir, dt, device)
        self.K = K
        self.H = H
        self.sigma_u = sigma_u
        
        print(f"[RandomShooting] K={K}, H={H}")
    
    def compute_control(self, theta_ref):
        U = np.random.normal(0, self.sigma_u, (self.K, self.H, 2)).astype(np.float32)
        J = np.zeros(self.K)
        
        for k in range(self.K):
            cost = 0.0
            h = self.h.clone()
            c = self.c.clone()
            
            for h_idx in range(self.H):
                p1 = self.p1_cmd + U[k, h_idx, 0]
                p2 = self.p2_cmd + U[k, h_idx, 1]
                
                p1 = np.clip(p1, 0, self.p_max)
                p2 = np.clip(p2, 0, self.p_max)
                
                theta, h, c = self.predict_step(p1, p2, h, c)
                
                err = theta_ref - theta
                cost += 30.0 * err**2 + 0.01 * (p1**2 + p2**2)
            
            J[k] = cost
        
        best_idx = np.argmin(J)
        dU = U[best_idx]
        
        return self.p1_cmd + dU[0, 0], self.p2_cmd + dU[0, 1], float(np.min(J))

# ==================== Simulation ====================
def run_simulation(controller, theta_target_deg, steps):
    theta_target = math.radians(theta_target_deg)
    
    print(f"\n[Simulation] Target: {theta_target_deg:.1f}°, Steps: {steps}")
    
    t = 0.0
    for step in range(steps):
        theta, p1, p2, cost, comp_time = controller.step(theta_target)
        
        error = theta_target - theta
        controller.log['t'].append(t)
        controller.log['theta'].append(theta)
        controller.log['theta_ref'].append(theta_target)
        controller.log['p1'].append(p1)
        controller.log['p2'].append(p2)
        controller.log['error'].append(error)
        controller.log['cost'].append(cost)
        controller.log['comp_time'].append(comp_time)
        
        if step % 20 == 0:
            print(f"  [{step:03d}] theta={math.degrees(theta):.2f}°, "
                  f"err={math.degrees(error):.2f}°")
        
        t += controller.dt
    
    errors_deg = np.degrees(np.array(controller.log['error']))
    print(f"\n[Results]")
    print(f"  Final error: {errors_deg[-1]:.2f}°")
    print(f"  RMS error: {np.sqrt(np.mean(errors_deg**2)):.2f}°")

def plot_results(controllers_dict, output_path):
    if not _HAS_PLT:
        return
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    for name, ctrl in controllers_dict.items():
        t = np.array(ctrl.log['t'])
        axes[0].plot(t, np.degrees(ctrl.log['theta']), label=name, linewidth=1.5)
        axes[1].plot(t, ctrl.log['p1'], label=f'{name} p1', linewidth=1)
        axes[1].plot(t, ctrl.log['p2'], '--', label=f'{name} p2', linewidth=1)
        axes[2].plot(t, np.degrees(ctrl.log['error']), label=name, linewidth=1)
        axes[3].plot(t, np.array(ctrl.log['comp_time'])*1000, label=name, linewidth=1)
    
    if len(controllers_dict) > 0:
        first_ctrl = list(controllers_dict.values())[0]
        t = np.array(first_ctrl.log['t'])
        axes[0].plot(t, np.degrees(first_ctrl.log['theta_ref']), 'k--', 
                    label='reference', linewidth=2, alpha=0.5)
    
    axes[0].set_ylabel('Angle [deg]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('Pressure [MPa]')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_ylabel('Error [deg]')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    axes[3].set_ylabel('Time [ms]')
    axes[3].set_xlabel('Time [s]')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[Saved] {output_path}")
    plt.close()

# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--controllers', nargs='+',
                       default=['mppi', 'gradient_mpc', 'random_shooting'],
                       choices=['mppi', 'gradient_mpc', 'random_shooting'])
    parser.add_argument('--theta_target_deg', type=float, default=30.0)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--out_dir', type=str, default='lstm_ss_control_results')
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    
    device = torch.device('cpu' if args.cpu else 
                         ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f" LSTM State Space Control Comparison")
    print(f"{'='*70}")
    
    controllers_dict = {}
    results = {}
    
    for ctrl_name in args.controllers:
        print(f"\n{'='*70}")
        print(f" Controller: {ctrl_name.upper()}")
        print(f"{'='*70}")
        
        if ctrl_name == 'mppi':
            ctrl = LSTM_MPPI(args.model_dir, args.dt, device, K=32, H=15)
        elif ctrl_name == 'gradient_mpc':
            ctrl = LSTM_GradientMPC(args.model_dir, args.dt, device, horizon=10, n_iter=50)
        elif ctrl_name == 'random_shooting':
            ctrl = LSTM_RandomShooting(args.model_dir, args.dt, device, K=64, H=15)
        
        run_simulation(ctrl, args.theta_target_deg, args.steps)
        controllers_dict[ctrl_name] = ctrl
        
        df = pd.DataFrame({
            't[s]': ctrl.log['t'],
            'theta[rad]': ctrl.log['theta'],
            'theta_ref[rad]': ctrl.log['theta_ref'],
            'error[rad]': ctrl.log['error'],
            'p1[MPa]': ctrl.log['p1'],
            'p2[MPa]': ctrl.log['p2'],
            'cost': ctrl.log['cost'],
            'comp_time[s]': ctrl.log['comp_time']
        })
        df.to_csv(os.path.join(args.out_dir, f'{ctrl_name}.csv'), index=False)
        
        errors_deg = np.degrees(np.array(ctrl.log['error']))
        results[ctrl_name] = {
            'rmse': float(np.sqrt(np.mean(errors_deg**2))),
            'mae': float(np.mean(np.abs(errors_deg))),
            'max_abs_error': float(np.max(np.abs(errors_deg))),
            'mean_comp_time_ms': float(np.mean(ctrl.log['comp_time']) * 1000)
        }
    
    plot_results(controllers_dict, os.path.join(args.out_dir, 'comparison.png'))
    
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f" SUMMARY")
    print(f"{'='*70}")
    print(f"{'Controller':<20} {'RMSE(°)':<10} {'MAE(°)':<10} {'Comp(ms)':<10}")
    print(f"{'-'*70}")
    
    for name in args.controllers:
        r = results[name]
        print(f"{name:<20} {r['rmse']:<10.3f} {r['mae']:<10.3f} {r['mean_comp_time_ms']:<10.1f}")
    
    print(f"\n{'='*70}")
    print(f" LSTM vs GRU Comparison")
    print(f"{'='*70}")
    print(f"Use this controller with both LSTM-SS and GRU-NLSS models")
    print(f"to compare their performance on the same task.")
    print(f"")
    print(f"Expected differences:")
    print(f"  - LSTM: Better long-term memory, but slower")
    print(f"  - GRU: Faster, simpler, often comparable performance")

if __name__ == '__main__':
    main()