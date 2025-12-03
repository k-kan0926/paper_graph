#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
control_hw.py
Hammerstein-Wienerモデル用制御スクリプト

制御手法:
1. Inverse Block Control - ブロックごとに逆写像
2. MPPI - サンプリングベース最適化
3. Linearized MPC - 線形部分を活用
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
class HammersteinWienerModel(nn.Module):
    def __init__(self, n_inputs=2, lags=24, hidden_dim=64, nl_hidden=32):
        super().__init__()
        
        self.n_inputs = n_inputs
        self.lags = lags
        self.hidden_dim = hidden_dim
        
        self.input_nl = nn.Sequential(
            nn.Linear(n_inputs, nl_hidden),
            nn.Tanh(),
            nn.Linear(nl_hidden, hidden_dim),
            nn.Tanh()
        )
        
        self.linear_dynamics = nn.Linear(hidden_dim * lags, hidden_dim, bias=True)
        
        self.output_nl = nn.Sequential(
            nn.Linear(hidden_dim, nl_hidden),
            nn.Tanh(),
            nn.Linear(nl_hidden, 1)
        )
    
    def forward(self, u_hist):
        batch_size = u_hist.shape[0]
        
        v_hist = []
        for t in range(self.lags):
            v_t = self.input_nl(u_hist[:, t, :])
            v_hist.append(v_t)
        
        v_hist = torch.stack(v_hist, dim=1)
        v_flat = v_hist.reshape(batch_size, -1)
        x = self.linear_dynamics(v_flat)
        theta = self.output_nl(x)
        
        return theta

# ==================== Base Controller ====================
class BaseHWController:
    def __init__(self, model_dir, dt=0.01, device='cpu'):
        self.load_model(model_dir, device)
        self.dt = dt
        self.device = device
        
        self.p_max = 0.70
        self.dp_max = 3.5
        
        self.theta_rad = 0.0
        self.p1_cmd = 0.0
        self.p2_cmd = 0.0
        
        maxlen = self.lags + 10
        self.hist_p1 = deque([0.0] * maxlen, maxlen=maxlen)
        self.hist_p2 = deque([0.0] * maxlen, maxlen=maxlen)
        
        self.log = {
            't': [], 'theta': [], 'theta_ref': [], 'error': [],
            'p1': [], 'p2': [], 'cost': [], 'comp_time': []
        }
    
    def load_model(self, model_dir, device):
        with open(os.path.join(model_dir, 'hw_meta.json'), 'r') as f:
            self.meta = json.load(f)
        
        self.lags = self.meta['lags']
        self.hidden_dim = self.meta['hidden_dim']
        self.nl_hidden = self.meta['nl_hidden']
        
        self.model = HammersteinWienerModel(
            n_inputs=2,
            lags=self.lags,
            hidden_dim=self.hidden_dim,
            nl_hidden=self.nl_hidden
        )
        
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, 'hw_model.pt'), 
                      map_location=device)
        )
        self.model.to(device)
        self.model.eval()
        
        print(f"[Model] Loaded HW: lags={self.lags}, hidden={self.hidden_dim}")
    
    def enforce_constraints(self, p1, p2, p1_prev, p2_prev):
        dp_max_step = self.dp_max * self.dt
        p1 = np.clip(p1, p1_prev - dp_max_step, p1_prev + dp_max_step)
        p2 = np.clip(p2, p2_prev - dp_max_step, p2_prev + dp_max_step)
        p1 = np.clip(p1, 0.0, self.p_max)
        p2 = np.clip(p2, 0.0, self.p_max)
        return p1, p2
    
    def predict(self, p1_hist, p2_hist):
        """1ステップ予測"""
        u_hist = np.zeros((1, self.lags, 2), dtype=np.float32)
        
        for k in range(self.lags):
            u_hist[0, k, 0] = p1_hist[k] if k < len(p1_hist) else 0.0
            u_hist[0, k, 1] = p2_hist[k] if k < len(p2_hist) else 0.0
        
        with torch.no_grad():
            theta = self.model(torch.from_numpy(u_hist).to(self.device))
        
        return float(theta.cpu().numpy().item())
    
    def compute_control(self, theta_ref):
        raise NotImplementedError
    
    def step(self, theta_ref):
        t_start = time.time()
        
        p1_cmd, p2_cmd, cost = self.compute_control(theta_ref)
        p1_cmd, p2_cmd = self.enforce_constraints(
            p1_cmd, p2_cmd, self.p1_cmd, self.p2_cmd
        )
        
        self.hist_p1.appendleft(p1_cmd)
        self.hist_p2.appendleft(p2_cmd)
        
        theta_next = self.predict(
            list(self.hist_p1)[:self.lags],
            list(self.hist_p2)[:self.lags]
        )
        
        self.theta_rad = theta_next
        self.p1_cmd = p1_cmd
        self.p2_cmd = p2_cmd
        
        comp_time = time.time() - t_start
        
        return theta_next, p1_cmd, p2_cmd, cost, comp_time

# ==================== 1. Inverse Block Control ====================
class HW_InverseBlock(BaseHWController):
    """逆ブロック制御
    
    θ_target → [Output NL]^{-1} → x_target
              → [Linear]^{-1} → v_target
              → [Input NL]^{-1} → u_target
    """
    
    def __init__(self, model_dir, dt=0.01, device='cpu', n_iter=50, lr=0.01):
        super().__init__(model_dir, dt, device)
        self.n_iter = n_iter
        self.lr = lr
        
        print(f"[InverseBlock] n_iter={n_iter}, lr={lr}")
    
    def compute_control(self, theta_ref):
        """勾配降下でブロックごとに逆を解く"""
        # Step 1: 出力非線形の逆を解く
        # x → Output_NL → θ_ref
        # x を最適化
        
        x_opt = torch.zeros(1, self.hidden_dim, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([x_opt], lr=self.lr)
        
        for _ in range(self.n_iter // 2):
            optimizer.zero_grad()
            theta_pred = self.model.output_nl(x_opt)
            loss = (theta_pred - theta_ref)**2
            loss.backward()
            optimizer.step()
        
        x_target = x_opt.detach()
        
        # Step 2: 線形動特性の逆を解く
        # v_hist → Linear → x_target
        # 線形なので解析的に解ける（疑似逆行列）
        
        W = self.model.linear_dynamics.weight.data  # (hidden_dim, hidden_dim * lags)
        b = self.model.linear_dynamics.bias.data    # (hidden_dim,)
        
        # W @ v_flat + b = x_target
        # v_flat = W^+ @ (x_target - b)
        
        try:
            W_pinv = torch.pinverse(W)
            v_flat_target = W_pinv @ (x_target.squeeze() - b)
            v_flat_target = v_flat_target.detach()
        except:
            # 疑似逆行列が計算できない場合は勾配降下
            v_flat_target = torch.zeros(self.hidden_dim * self.lags, 
                                       requires_grad=True, device=self.device)
            optimizer2 = torch.optim.Adam([v_flat_target], lr=self.lr)
            
            for _ in range(self.n_iter // 2):
                optimizer2.zero_grad()
                x_pred = self.model.linear_dynamics(v_flat_target.unsqueeze(0))
                loss = torch.sum((x_pred - x_target)**2)
                loss.backward()
                optimizer2.step()
            
            v_flat_target = v_flat_target.detach()
        
        # Step 3: 入力非線形の逆を解く
        # u → Input_NL → v (最新のタイムステップのみ)
        
        v_target_latest = v_flat_target[:self.hidden_dim]  # 最新のv
        
        u_opt = torch.tensor([self.p1_cmd, self.p2_cmd], 
                            requires_grad=True, device=self.device)
        optimizer3 = torch.optim.Adam([u_opt], lr=self.lr)
        
        for _ in range(self.n_iter):
            optimizer3.zero_grad()
            v_pred = self.model.input_nl(u_opt.unsqueeze(0))
            loss = torch.sum((v_pred - v_target_latest)**2)
            
            # 物理制約
            loss += 100.0 * torch.relu(u_opt[0] - self.p_max)**2
            loss += 100.0 * torch.relu(u_opt[1] - self.p_max)**2
            loss += 100.0 * torch.relu(-u_opt[0])**2
            loss += 100.0 * torch.relu(-u_opt[1])**2
            
            loss.backward()
            optimizer3.step()
        
        p1_cmd = float(u_opt[0].detach().cpu().numpy())
        p2_cmd = float(u_opt[1].detach().cpu().numpy())
        
        return p1_cmd, p2_cmd, float(loss.item())

# ==================== 2. MPPI Controller ====================
class HW_MPPI(BaseHWController):
    def __init__(self, model_dir, dt=0.01, device='cpu',
                 K=32, H=15, lam=2.0, sigma_u=0.10):
        super().__init__(model_dir, dt, device)
        self.K = K
        self.H = H
        self.temperature = lam
        self.sigma_u = sigma_u
        
        print(f"[MPPI] K={K}, H={H}")
    
    def compute_control(self, theta_ref):
        U = np.random.normal(0, self.sigma_u, (self.K, self.H, 2)).astype(np.float32)
        J = np.zeros(self.K)
        
        for k in range(self.K):
            cost = 0.0
            p1_hist = list(self.hist_p1)[:self.lags]
            p2_hist = list(self.hist_p2)[:self.lags]
            
            for h in range(self.H):
                p1 = p1_hist[0] + U[k, h, 0]
                p2 = p2_hist[0] + U[k, h, 1]
                
                p1 = np.clip(p1, 0, self.p_max)
                p2 = np.clip(p2, 0, self.p_max)
                
                p1_hist = [p1] + p1_hist[:self.lags-1]
                p2_hist = [p2] + p2_hist[:self.lags-1]
                
                theta = self.predict(p1_hist, p2_hist)
                
                err = theta_ref - theta
                cost += 30.0 * err**2 + 0.01 * (p1**2 + p2**2)
            
            J[k] = cost
        
        beta = np.min(J)
        w = np.exp(-(J - beta) / max(1e-6, self.temperature))
        w = w / (np.sum(w) + 1e-9)
        
        dU = np.sum(w[:, None, None] * U, axis=0)
        
        return self.p1_cmd + dU[0, 0], self.p2_cmd + dU[0, 1], float(np.min(J))

# ==================== 3. Linearized MPC ====================
class HW_LinearizedMPC(BaseHWController):
    """線形部分を活用したMPC"""
    
    def __init__(self, model_dir, dt=0.01, device='cpu',
                 horizon=10, Q=10.0, R=0.1):
        super().__init__(model_dir, dt, device)
        self.horizon = horizon
        self.Q = Q
        self.R = R
        
        print(f"[LinearizedMPC] H={horizon}")
    
    def compute_control(self, theta_ref):
        """簡易版: 勾配降下で最適化"""
        from scipy.optimize import minimize
        
        H = self.horizon
        u0 = np.array([self.p1_cmd, self.p2_cmd] * H)
        
        def cost_function(u):
            cost = 0.0
            p1_hist = list(self.hist_p1)[:self.lags]
            p2_hist = list(self.hist_p2)[:self.lags]
            
            for h in range(H):
                p1 = u[2*h]
                p2 = u[2*h + 1]
                
                p1_hist = [p1] + p1_hist[:self.lags-1]
                p2_hist = [p2] + p2_hist[:self.lags-1]
                
                theta = self.predict(p1_hist, p2_hist)
                
                cost += self.Q * (theta - theta_ref)**2
                cost += self.R * (p1**2 + p2**2)
            
            return cost
        
        bounds = [(0, self.p_max) for _ in range(2*H)]
        
        result = minimize(cost_function, u0, method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': 50})
        
        if result.success:
            u_opt = result.x
            p1_cmd = float(u_opt[0])
            p2_cmd = float(u_opt[1])
        else:
            p1_cmd = self.p1_cmd
            p2_cmd = self.p2_cmd
        
        return p1_cmd, p2_cmd, abs(theta_ref - self.theta_rad)

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
                       default=['inverse_block', 'mppi', 'linearized_mpc'],
                       choices=['inverse_block', 'mppi', 'linearized_mpc'])
    parser.add_argument('--theta_target_deg', type=float, default=30.0)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--out_dir', type=str, default='hw_control_results')
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    
    device = torch.device('cpu' if args.cpu else 
                         ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f" Hammerstein-Wiener Control Comparison")
    print(f"{'='*70}")
    
    controllers_dict = {}
    results = {}
    
    for ctrl_name in args.controllers:
        print(f"\n{'='*70}")
        print(f" Controller: {ctrl_name.upper()}")
        print(f"{'='*70}")
        
        if ctrl_name == 'inverse_block':
            ctrl = HW_InverseBlock(args.model_dir, args.dt, device, n_iter=50)
        elif ctrl_name == 'mppi':
            ctrl = HW_MPPI(args.model_dir, args.dt, device, K=32, H=15)
        elif ctrl_name == 'linearized_mpc':
            ctrl = HW_LinearizedMPC(args.model_dir, args.dt, device, horizon=10)
        
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

if __name__ == '__main__':
    main()