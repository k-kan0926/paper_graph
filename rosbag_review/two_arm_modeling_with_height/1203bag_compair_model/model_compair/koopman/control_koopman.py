#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
control_koopman.py
Koopmanモデル用制御スクリプト

制御手法:
1. Linear MPC (z空間で線形計画問題)
2. LQR (z空間でARE)
3. MPPI (参考実装)

Koopmanの利点:
- 非線形系を線形空間で扱える
- 線形制御理論が厳密に使える
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

try:
    from scipy.linalg import solve_discrete_are
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except:
    _HAS_SCIPY = False

# ==================== Model Definition ====================
class KoopmanModel(nn.Module):
    def __init__(self, state_dim=2, control_dim=2, latent_dim=64):
        super().__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + control_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(control_dim, latent_dim, bias=False)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def encode(self, x, u):
        xu = torch.cat([x, u], dim=-1)
        z = self.encoder(xu)
        return z
    
    def dynamics(self, z, u):
        z_next = self.K(z) + self.B(u)
        return z_next
    
    def decode(self, z):
        theta = self.decoder(z)
        return theta
    
    def forward(self, x, u):
        z = self.encode(x, u)
        z_next = self.dynamics(z, u)
        theta_next = self.decode(z_next)
        return theta_next, z, z_next

# ==================== Base Controller ====================
class BaseKoopmanController:
    def __init__(self, model_dir, dt=0.01, device='cpu'):
        self.load_model(model_dir, device)
        self.dt = dt
        self.device = device
        
        self.p_max = 0.70
        self.dp_max = 3.5
        
        self.theta_rad = 0.0
        self.p1_cmd = 0.0
        self.p2_cmd = 0.0
        
        # Latent state
        self.z = np.zeros(self.latent_dim, dtype=np.float32)
        
        self.log = {
            't': [], 'theta': [], 'theta_ref': [], 'error': [],
            'p1': [], 'p2': [], 'cost': [], 'comp_time': []
        }
    
    def load_model(self, model_dir, device):
        with open(os.path.join(model_dir, 'koopman_meta.json'), 'r') as f:
            self.meta = json.load(f)
        
        self.latent_dim = self.meta['latent_dim']
        self.state_dim = self.meta['state_dim']
        self.control_dim = self.meta['control_dim']
        
        self.model = KoopmanModel(
            state_dim=self.state_dim,
            control_dim=self.control_dim,
            latent_dim=self.latent_dim
        )
        
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, 'koopman_model.pt'),
                      map_location=device)
        )
        self.model.to(device)
        self.model.eval()
        
        # Extract linear matrices
        self.K_matrix = self.model.K.weight.data.cpu().numpy()
        self.B_matrix = self.model.B.weight.data.cpu().numpy()
        
        print(f"[Model] Loaded Koopman: latent_dim={self.latent_dim}")
        print(f"  K matrix: {self.K_matrix.shape}")
        print(f"  B matrix: {self.B_matrix.shape}")
    
    def enforce_constraints(self, p1, p2, p1_prev, p2_prev):
        dp_max_step = self.dp_max * self.dt
        p1 = np.clip(p1, p1_prev - dp_max_step, p1_prev + dp_max_step)
        p2 = np.clip(p2, p2_prev - dp_max_step, p2_prev + dp_max_step)
        p1 = np.clip(p1, 0.0, self.p_max)
        p2 = np.clip(p2, 0.0, self.p_max)
        return p1, p2
    
    def state_to_observable(self, theta):
        """状態を観測可能量に変換"""
        return np.array([theta, theta**2], dtype=np.float32)
    
    def encode_to_latent(self, theta, p1, p2):
        """エンコードして潜在状態を取得"""
        x = self.state_to_observable(theta)
        u = np.array([p1, p2], dtype=np.float32)
        
        x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
        u_t = torch.from_numpy(u).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            z = self.model.encode(x_t, u_t)
        
        return z.cpu().numpy().flatten()
    
    def decode_from_latent(self, z):
        """潜在状態からθをデコード"""
        z_t = torch.from_numpy(z).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            theta = self.model.decode(z_t)
        
        return float(theta.cpu().numpy().item())
    
    def predict_latent(self, z, u):
        """線形動特性: z_{t+1} = K z + B u"""
        z_next = self.K_matrix @ z + self.B_matrix @ u
        return z_next
    
    def compute_control(self, theta_ref):
        raise NotImplementedError
    
    def step(self, theta_ref):
        t_start = time.time()
        
        p1_cmd, p2_cmd, cost = self.compute_control(theta_ref)
        p1_cmd, p2_cmd = self.enforce_constraints(
            p1_cmd, p2_cmd, self.p1_cmd, self.p2_cmd
        )
        
        # Update latent state
        u = np.array([p1_cmd, p2_cmd], dtype=np.float32)
        self.z = self.predict_latent(self.z, u)
        
        # Decode to theta
        theta_next = self.decode_from_latent(self.z)
        
        # Re-encode for next step (to reduce drift)
        self.z = self.encode_to_latent(theta_next, p1_cmd, p2_cmd)
        
        self.theta_rad = theta_next
        self.p1_cmd = p1_cmd
        self.p2_cmd = p2_cmd
        
        comp_time = time.time() - t_start
        
        return theta_next, p1_cmd, p2_cmd, cost, comp_time

# ==================== 1. Linear MPC ====================
class Koopman_LinearMPC(BaseKoopmanController):
    """線形MPC（z空間で）"""
    
    def __init__(self, model_dir, dt=0.01, device='cpu',
                 horizon=10, Q_weight=10.0, R_weight=0.1):
        super().__init__(model_dir, dt, device)
        
        if not _HAS_SCIPY:
            raise ImportError("scipy required for MPC")
        
        self.horizon = horizon
        self.Q_weight = Q_weight
        self.R_weight = R_weight
        
        print(f"[LinearMPC] Horizon={horizon}, Q={Q_weight}, R={R_weight}")
    
    def compute_control(self, theta_ref):
        """MPCを解く（z空間で線形計画問題）"""
        H = self.horizon
        
        # Target latent state (θ_ref に対応する z_ref を推定)
        x_ref = self.state_to_observable(theta_ref)
        u_current = np.array([self.p1_cmd, self.p2_cmd], dtype=np.float32)
        
        x_ref_t = torch.from_numpy(x_ref).unsqueeze(0).to(self.device)
        u_current_t = torch.from_numpy(u_current).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            z_ref = self.model.encode(x_ref_t, u_current_t).cpu().numpy().flatten()
        
        # 初期推定値
        u0 = np.array([self.p1_cmd, self.p2_cmd] * H)
        
        def cost_function(u_flat):
            """コスト関数"""
            cost = 0.0
            z = self.z.copy()
            
            for h in range(H):
                u_h = u_flat[2*h:2*h+2]
                
                # Predict
                z = self.predict_latent(z, u_h)
                
                # State cost
                cost += self.Q_weight * np.sum((z - z_ref)**2)
                
                # Control cost
                cost += self.R_weight * np.sum(u_h**2)
            
            return cost
        
        # Bounds
        bounds = [(0, self.p_max) for _ in range(2*H)]
        
        # Optimize
        result = minimize(cost_function, u0, method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': 50})
        
        if result.success:
            u_opt = result.x
            p1_cmd = float(u_opt[0])
            p2_cmd = float(u_opt[1])
        else:
            p1_cmd = self.p1_cmd
            p2_cmd = self.p2_cmd
        
        cost = abs(theta_ref - self.theta_rad)
        
        return p1_cmd, p2_cmd, cost

# ==================== 2. LQR ====================
class Koopman_LQR(BaseKoopmanController):
    """LQR（z空間で）"""
    
    def __init__(self, model_dir, dt=0.01, device='cpu',
                 Q_weight=10.0, R_weight=0.1):
        super().__init__(model_dir, dt, device)
        
        if not _HAS_SCIPY:
            raise ImportError("scipy required for LQR")
        
        self.Q_weight = Q_weight
        self.R_weight = R_weight
        
        # Solve Discrete Algebraic Riccati Equation
        self.compute_lqr_gain()
        
        print(f"[LQR] Q={Q_weight}, R={R_weight}")
        print(f"  Gain shape: {self.lqr_gain.shape}")
    
    def compute_lqr_gain(self):
        """LQRゲインを計算"""
        # Q matrix (state cost)
        Q = self.Q_weight * np.eye(self.latent_dim)
        
        # R matrix (control cost)
        R = self.R_weight * np.eye(self.control_dim)
        
        # Solve DARE: A^T P A - P - A^T P B (R + B^T P B)^{-1} B^T P A + Q = 0
        try:
            P = solve_discrete_are(self.K_matrix, self.B_matrix, Q, R)
            
            # LQR gain: K = (R + B^T P B)^{-1} B^T P A
            BTPsB = self.B_matrix.T @ P @ self.B_matrix + R
            self.lqr_gain = np.linalg.solve(BTPsB, self.B_matrix.T @ P @ self.K_matrix)
            
        except Exception as e:
            print(f"[WARNING] LQR DARE failed: {e}")
            print(f"  Using simple proportional gain")
            self.lqr_gain = np.random.randn(self.control_dim, self.latent_dim) * 0.01
    
    def compute_control(self, theta_ref):
        """LQR制御: u = -K (z - z_ref)"""
        # Target latent state
        x_ref = self.state_to_observable(theta_ref)
        u_current = np.array([self.p1_cmd, self.p2_cmd], dtype=np.float32)
        
        x_ref_t = torch.from_numpy(x_ref).unsqueeze(0).to(self.device)
        u_current_t = torch.from_numpy(u_current).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            z_ref = self.model.encode(x_ref_t, u_current_t).cpu().numpy().flatten()
        
        # LQR control
        z_error = self.z - z_ref
        u = -self.lqr_gain @ z_error
        
        p1_cmd = float(u[0])
        p2_cmd = float(u[1])
        
        # Clip
        p1_cmd = np.clip(p1_cmd, 0, self.p_max)
        p2_cmd = np.clip(p2_cmd, 0, self.p_max)
        
        cost = abs(theta_ref - self.theta_rad)
        
        return p1_cmd, p2_cmd, cost

# ==================== 3. MPPI (参考) ====================
class Koopman_MPPI(BaseKoopmanController):
    """MPPI制御（参考実装）"""
    
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
            z = self.z.copy()
            
            for h in range(self.H):
                u = U[k, h]
                u = np.clip(u + np.array([self.p1_cmd, self.p2_cmd]), 0, self.p_max)
                
                # Predict in latent space
                z = self.predict_latent(z, u)
                
                # Decode to theta
                theta = self.decode_from_latent(z)
                
                err = theta_ref - theta
                cost += 30.0 * err**2 + 0.01 * np.sum(u**2)
            
            J[k] = cost
        
        beta = np.min(J)
        w = np.exp(-(J - beta) / max(1e-6, self.temperature))
        w = w / (np.sum(w) + 1e-9)
        
        dU = np.sum(w[:, None, None] * U, axis=0)
        
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
                       default=['linear_mpc', 'lqr', 'mppi'],
                       choices=['linear_mpc', 'lqr', 'mppi'])
    parser.add_argument('--theta_target_deg', type=float, default=30.0)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--out_dir', type=str, default='koopman_control_results')
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    
    device = torch.device('cpu' if args.cpu else 
                         ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f" Koopman Control Comparison")
    print(f"{'='*70}")
    
    controllers_dict = {}
    results = {}
    
    for ctrl_name in args.controllers:
        print(f"\n{'='*70}")
        print(f" Controller: {ctrl_name.upper()}")
        print(f"{'='*70}")
        
        try:
            if ctrl_name == 'linear_mpc':
                ctrl = Koopman_LinearMPC(args.model_dir, args.dt, device, horizon=10)
            elif ctrl_name == 'lqr':
                ctrl = Koopman_LQR(args.model_dir, args.dt, device, Q_weight=10.0)
            elif ctrl_name == 'mppi':
                ctrl = Koopman_MPPI(args.model_dir, args.dt, device, K=32, H=15)
            
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
        
        except Exception as e:
            print(f"[ERROR] {ctrl_name}: {e}")
    
    plot_results(controllers_dict, os.path.join(args.out_dir, 'comparison.png'))
    
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f" SUMMARY")
    print(f"{'='*70}")
    print(f"{'Controller':<20} {'RMSE(°)':<10} {'MAE(°)':<10} {'Comp(ms)':<10}")
    print(f"{'-'*70}")
    
    for name in controllers_dict.keys():
        r = results[name]
        print(f"{name:<20} {r['rmse']:<10.3f} {r['mae']:<10.3f} {r['mean_comp_time_ms']:<10.1f}")

if __name__ == '__main__':
    main()