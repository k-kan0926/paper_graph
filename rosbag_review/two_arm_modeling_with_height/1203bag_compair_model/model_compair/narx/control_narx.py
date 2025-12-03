#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
control_narx.py
NARX モデル用の包括的制御スクリプト

サポートする制御手法:
1. MPPI (Model Predictive Path Integral)
2. Inverse Mapping (逆写像制御)
3. PID
4. LQI (Linear Quadratic Integral)
5. CEM (Cross-Entropy Method)
6. Random Shooting
"""
import os, json, math, argparse, time
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import pandas as pd

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except:
    _HAS_PLT = False

# ==================== NARX Model ====================
class MLP_NARX(nn.Module):
    def __init__(self, in_dim, hidden=[192, 192], out_dim=1, dropout=0.0):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# ==================== Base Controller ====================
class BaseNARXController:
    """NARXコントローラーのベースクラス"""
    
    def __init__(self, model_dir, dt=0.01, device='cpu'):
        # Load model
        self.load_model(model_dir, device)
        self.dt = dt
        self.device = device
        
        # Physical limits
        self.p_max = 0.70
        self.dp_max = 3.5
        
        # State
        self.theta_rad = 0.0
        self.p1_cmd = 0.0
        self.p2_cmd = 0.0
        
        # History
        maxlen = self.lags + 10
        self.hist_theta = deque([0.0] * maxlen, maxlen=maxlen)
        self.hist_p1 = deque([0.0] * maxlen, maxlen=maxlen)
        self.hist_p2 = deque([0.0] * maxlen, maxlen=maxlen)
        self.hist_dp1 = deque([0.0] * maxlen, maxlen=maxlen)
        self.hist_dp2 = deque([0.0] * maxlen, maxlen=maxlen)
        
        # Logging
        self.log = {
            't': [], 'theta': [], 'theta_ref': [], 'error': [],
            'p1': [], 'p2': [], 'cost': [], 'comp_time': []
        }
    
    def load_model(self, model_dir, device):
        """モデルロード"""
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
        
        in_dim = self.lags * len(self.feat_cols)
        self.model = MLP_NARX(in_dim, [self.hidden, self.hidden], 1, 0.0)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        print(f"[Model] Loaded NARX: lags={self.lags}, delay={self.delay}")
    
    def enforce_constraints(self, p1, p2, p1_prev, p2_prev):
        dp_max_step = self.dp_max * self.dt
        p1 = np.clip(p1, p1_prev - dp_max_step, p1_prev + dp_max_step)
        p2 = np.clip(p2, p2_prev - dp_max_step, p2_prev + dp_max_step)
        p1 = np.clip(p1, 0.0, self.p_max)
        p2 = np.clip(p2, 0.0, self.p_max)
        return p1, p2
    
    def predict(self, theta, p1, p2, dp1, dp2):
        """1ステップ予測"""
        x_list = []
        for k in range(self.lags):
            x_list.extend([
                list(self.hist_theta)[k] if k < len(list(self.hist_theta)) else theta,
                list(self.hist_p1)[k] if k < len(list(self.hist_p1)) else p1,
                list(self.hist_p2)[k] if k < len(list(self.hist_p2)) else p2,
                list(self.hist_dp1)[k] if k < len(list(self.hist_dp1)) else dp1,
                list(self.hist_dp2)[k] if k < len(list(self.hist_dp2)) else dp2,
            ])
        
        x = np.array(x_list, dtype=np.float32).reshape(1, -1)
        x_norm = (x - self.mu) / (self.std + 1e-8)
        
        with torch.no_grad():
            theta_next = self.model(torch.from_numpy(x_norm).to(self.device))
        
        return float(theta_next.cpu().numpy().item())
    
    def compute_control(self, theta_ref):
        """制御入力計算 (サブクラスで実装)"""
        raise NotImplementedError
    
    def step(self, theta_ref):
        """シミュレーションステップ"""
        t_start = time.time()
        
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
        theta_next = self.predict(self.theta_rad, p1_cmd, p2_cmd, dp1, dp2)
        
        # Update state
        self.theta_rad = theta_next
        self.p1_cmd = p1_cmd
        self.p2_cmd = p2_cmd
        
        # Update history
        self.hist_theta.appendleft(theta_next)
        self.hist_p1.appendleft(p1_cmd)
        self.hist_p2.appendleft(p2_cmd)
        self.hist_dp1.appendleft(dp1)
        self.hist_dp2.appendleft(dp2)
        
        comp_time = time.time() - t_start
        
        return theta_next, p1_cmd, p2_cmd, cost, comp_time

# ==================== 1. MPPI Controller ====================
class NARX_MPPI(BaseNARXController):
    """MPPI制御"""
    
    def __init__(self, model_dir, dt=0.01, device='cpu',
                 K=32, H=15, lam=2.0, sigma_u=0.10):
        super().__init__(model_dir, dt, device)
        self.K = K
        self.H = H
        self.temperature = lam
        self.sigma_u = sigma_u
        
        self.w_track = 30.0
        self.w_smooth = 0.05
        self.w_effort = 0.01
        
        print(f"[MPPI] K={K}, H={H}, lambda={lam}")
    
    def compute_control(self, theta_ref):
        # Sample noise
        U = np.random.normal(0, self.sigma_u, (self.K, self.H, 2)).astype(np.float32)
        
        # Rollout
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
                
                p1, p2 = p1 + dp1, p2 + dp2
                p1, p2 = self.enforce_constraints(p1, p2, p1_prev, p2_prev)
                
                hist_theta = [theta] + hist_theta[:self.lags-1]
                hist_p1 = [p1] + hist_p1[:self.lags-1]
                hist_p2 = [p2] + hist_p2[:self.lags-1]
                hist_dp1 = [(p1-p1_prev)/self.dt] + hist_dp1[:self.lags-1]
                hist_dp2 = [(p2-p2_prev)/self.dt] + hist_dp2[:self.lags-1]
                
                x_list = []
                for i in range(self.lags):
                    x_list.extend([hist_theta[i], hist_p1[i], hist_p2[i],
                                  hist_dp1[i], hist_dp2[i]])
                
                x = np.array(x_list, dtype=np.float32).reshape(1, -1)
                x_norm = (x - self.mu) / (self.std + 1e-8)
                
                with torch.no_grad():
                    theta = self.model(torch.from_numpy(x_norm).to(self.device)).cpu().numpy().item()
                
                err = theta_ref - theta
                cost += self.w_track * err**2
                cost += self.w_smooth * (dp1**2 + dp2**2)
                cost += self.w_effort * (p1**2 + p2**2)
            
            J[k] = cost
        
        # MPPI weighting
        beta = np.min(J)
        w = np.exp(-(J - beta) / max(1e-6, self.temperature))
        w = w / (np.sum(w) + 1e-9)
        
        dU = np.sum(w[:, None, None] * U, axis=0)
        
        return self.p1_cmd + dU[0, 0], self.p2_cmd + dU[0, 1], float(np.min(J))

# ==================== 2. Inverse Mapping Controller ====================
class NARX_InverseMapping(BaseNARXController):
    """逆写像制御 - 勾配降下で最適化"""
    
    def __init__(self, model_dir, dt=0.01, device='cpu',
                 n_iter=50, lr=0.01):
        super().__init__(model_dir, dt, device)
        self.n_iter = n_iter
        self.lr = lr
        
        print(f"[Inverse] n_iter={n_iter}, lr={lr}")
    
    def compute_control(self, theta_ref):
        """逆写像を勾配降下で解く"""
        # 初期値（現在の圧力から少し変化）
        p1_opt = torch.tensor([self.p1_cmd], requires_grad=True, device=self.device)
        p2_opt = torch.tensor([self.p2_cmd], requires_grad=True, device=self.device)
        
        optimizer = torch.optim.Adam([p1_opt, p2_opt], lr=self.lr)
        
        for _ in range(self.n_iter):
            optimizer.zero_grad()
            
            # Build feature vector
            dp1 = (p1_opt.item() - self.p1_cmd) / self.dt
            dp2 = (p2_opt.item() - self.p2_cmd) / self.dt
            
            x_list = []
            for k in range(self.lags):
                x_list.extend([
                    list(self.hist_theta)[k] if k < len(list(self.hist_theta)) else self.theta_rad,
                    list(self.hist_p1)[k] if k == 0 else list(self.hist_p1)[k],
                    list(self.hist_p2)[k] if k == 0 else list(self.hist_p2)[k],
                    dp1 if k == 0 else list(self.hist_dp1)[k],
                    dp2 if k == 0 else list(self.hist_dp2)[k],
                ])
            
            # 最新の入力を反映
            x_list[1] = p1_opt.item()
            x_list[2] = p2_opt.item()
            x_list[3] = dp1
            x_list[4] = dp2
            
            x = torch.tensor(x_list, dtype=torch.float32, device=self.device).reshape(1, -1)
            x_norm = (x - torch.from_numpy(self.mu).to(self.device)) / (torch.from_numpy(self.std).to(self.device) + 1e-8)
            
            theta_pred = self.model(x_norm)
            
            # Loss: 目標角度との差
            loss = (theta_pred - theta_ref)**2
            
            # 物理制約のペナルティ
            loss += 100.0 * torch.relu(p1_opt - self.p_max)**2
            loss += 100.0 * torch.relu(p2_opt - self.p_max)**2
            loss += 100.0 * torch.relu(-p1_opt)**2
            loss += 100.0 * torch.relu(-p2_opt)**2
            
            loss.backward()
            optimizer.step()
        
        p1_cmd = float(p1_opt.detach().cpu().numpy())
        p2_cmd = float(p2_opt.detach().cpu().numpy())
        
        return p1_cmd, p2_cmd, float(loss.item())

# ==================== 3. PID Controller ====================
class NARX_PID(BaseNARXController):
    """PID制御"""
    
    def __init__(self, model_dir, dt=0.01, device='cpu',
                 Kp=0.5, Ki=0.05, Kd=0.1):
        super().__init__(model_dir, dt, device)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.integral = 0.0
        self.prev_error = 0.0
        
        print(f"[PID] Kp={Kp}, Ki={Ki}, Kd={Kd}")
    
    def compute_control(self, theta_ref):
        error = theta_ref - self.theta_rad
        
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        # Map to pressure (heuristic)
        if u > 0:
            p1_cmd = self.p1_cmd + abs(u) * 0.1
            p2_cmd = max(0, self.p2_cmd - abs(u) * 0.1)
        else:
            p1_cmd = max(0, self.p1_cmd - abs(u) * 0.1)
            p2_cmd = self.p2_cmd + abs(u) * 0.1
        
        self.prev_error = error
        
        return p1_cmd, p2_cmd, abs(error)

# ==================== 4. LQI Controller ====================
class NARX_LQI(BaseNARXController):
    """LQI制御 - 線形化してLQR"""
    
    def __init__(self, model_dir, dt=0.01, device='cpu',
                 Q_theta=10.0, Q_int=1.0, R=0.1):
        super().__init__(model_dir, dt, device)
        self.Q_theta = Q_theta
        self.Q_int = Q_int
        self.R = R
        
        self.integral = 0.0
        
        print(f"[LQI] Q_theta={Q_theta}, Q_int={Q_int}, R={R}")
    
    def compute_control(self, theta_ref):
        """局所線形化してLQR"""
        error = theta_ref - self.theta_rad
        self.integral += error * self.dt
        
        # ヤコビアンを数値微分で計算
        eps = 1e-4
        
        # ∂θ/∂p1
        theta_0 = self.predict(self.theta_rad, self.p1_cmd, self.p2_cmd, 0, 0)
        theta_p1 = self.predict(self.theta_rad, self.p1_cmd + eps, self.p2_cmd, eps/self.dt, 0)
        dtheta_dp1 = (theta_p1 - theta_0) / eps
        
        # ∂θ/∂p2
        theta_p2 = self.predict(self.theta_rad, self.p1_cmd, self.p2_cmd + eps, 0, eps/self.dt)
        dtheta_dp2 = (theta_p2 - theta_0) / eps
        
        # LQR-like control
        # u = -K * [error; integral]
        # 簡易版: u_p1 = k1 * error + k2 * integral
        k1 = self.Q_theta * dtheta_dp1
        k2 = self.Q_int * dtheta_dp1
        
        u_p1 = k1 * error + k2 * self.integral
        u_p2 = -k1 * error - k2 * self.integral  # 拮抗
        
        p1_cmd = self.p1_cmd + u_p1 * 0.1
        p2_cmd = self.p2_cmd + u_p2 * 0.1
        
        return p1_cmd, p2_cmd, abs(error)

# ==================== 5. CEM Controller ====================
class NARX_CEM(BaseNARXController):
    """Cross-Entropy Method"""
    
    def __init__(self, model_dir, dt=0.01, device='cpu',
                 K=64, H=15, elite_frac=0.2, n_iter=3):
        super().__init__(model_dir, dt, device)
        self.K = K
        self.H = H
        self.elite_frac = elite_frac
        self.n_iter = n_iter
        
        self.mean = np.zeros((H, 2), dtype=np.float32)
        self.std = np.ones((H, 2), dtype=np.float32) * 0.1
        
        self.w_track = 30.0
        self.w_smooth = 0.05
        self.w_effort = 0.01
        
        print(f"[CEM] K={K}, H={H}, elite={elite_frac}")
    
    def compute_control(self, theta_ref):
        mean, std = self.mean.copy(), self.std.copy()
        
        for _ in range(self.n_iter):
            U = np.random.normal(mean[None, :, :], std[None, :, :], (self.K, self.H, 2)).astype(np.float32)
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
                    
                    p1, p2 = p1 + dp1, p2 + dp2
                    p1, p2 = self.enforce_constraints(p1, p2, p1_prev, p2_prev)
                    
                    hist_theta = [theta] + hist_theta[:self.lags-1]
                    hist_p1 = [p1] + hist_p1[:self.lags-1]
                    hist_p2 = [p2] + hist_p2[:self.lags-1]
                    hist_dp1 = [(p1-p1_prev)/self.dt] + hist_dp1[:self.lags-1]
                    hist_dp2 = [(p2-p2_prev)/self.dt] + hist_dp2[:self.lags-1]
                    
                    x_list = []
                    for i in range(self.lags):
                        x_list.extend([hist_theta[i], hist_p1[i], hist_p2[i],
                                      hist_dp1[i], hist_dp2[i]])
                    
                    x = np.array(x_list, dtype=np.float32).reshape(1, -1)
                    x_norm = (x - self.mu) / (self.std + 1e-8)
                    
                    with torch.no_grad():
                        theta = self.model(torch.from_numpy(x_norm).to(self.device)).cpu().numpy().item()
                    
                    err = theta_ref - theta
                    cost += self.w_track * err**2
                    cost += self.w_smooth * (dp1**2 + dp2**2)
                    cost += self.w_effort * (p1**2 + p2**2)
                
                J[k] = cost
            
            # Select elites
            n_elite = max(1, int(self.K * self.elite_frac))
            elite_idx = np.argsort(J)[:n_elite]
            elite_U = U[elite_idx]
            
            mean = np.mean(elite_U, axis=0)
            std = np.std(elite_U, axis=0) + 1e-3
        
        self.mean, self.std = mean, std
        
        return self.p1_cmd + mean[0, 0], self.p2_cmd + mean[0, 1], float(np.min(J))

# ==================== 6. Random Shooting Controller ====================
class NARX_RandomShooting(BaseNARXController):
    """Random Shooting"""
    
    def __init__(self, model_dir, dt=0.01, device='cpu',
                 K=64, H=15, sigma_u=0.15):
        super().__init__(model_dir, dt, device)
        self.K = K
        self.H = H
        self.sigma_u = sigma_u
        
        self.w_track = 30.0
        self.w_smooth = 0.05
        self.w_effort = 0.01
        
        print(f"[RandomShooting] K={K}, H={H}")
    
    def compute_control(self, theta_ref):
        U = np.random.normal(0, self.sigma_u, (self.K, self.H, 2)).astype(np.float32)
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
                
                p1, p2 = p1 + dp1, p2 + dp2
                p1, p2 = self.enforce_constraints(p1, p2, p1_prev, p2_prev)
                
                hist_theta = [theta] + hist_theta[:self.lags-1]
                hist_p1 = [p1] + hist_p1[:self.lags-1]
                hist_p2 = [p2] + hist_p2[:self.lags-1]
                hist_dp1 = [(p1-p1_prev)/self.dt] + hist_dp1[:self.lags-1]
                hist_dp2 = [(p2-p2_prev)/self.dt] + hist_dp2[:self.lags-1]
                
                x_list = []
                for i in range(self.lags):
                    x_list.extend([hist_theta[i], hist_p1[i], hist_p2[i],
                                  hist_dp1[i], hist_dp2[i]])
                
                x = np.array(x_list, dtype=np.float32).reshape(1, -1)
                x_norm = (x - self.mu) / (self.std + 1e-8)
                
                with torch.no_grad():
                    theta = self.model(torch.from_numpy(x_norm).to(self.device)).cpu().numpy().item()
                
                err = theta_ref - theta
                cost += self.w_track * err**2
                cost += self.w_smooth * (dp1**2 + dp2**2)
                cost += self.w_effort * (p1**2 + p2**2)
            
            J[k] = cost
        
        best_idx = np.argmin(J)
        dU = U[best_idx]
        
        return self.p1_cmd + dU[0, 0], self.p2_cmd + dU[0, 1], float(np.min(J))

# ==================== Simulation Runner ====================
def run_simulation(controller, theta_target_deg, steps):
    """シミュレーション実行"""
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
                  f"err={math.degrees(error):.2f}°, "
                  f"time={comp_time*1000:.1f}ms")
        
        t += controller.dt
    
    # Metrics
    errors_deg = np.degrees(np.array(controller.log['error']))
    print(f"\n[Results]")
    print(f"  Final error: {errors_deg[-1]:.2f}°")
    print(f"  RMS error: {np.sqrt(np.mean(errors_deg**2)):.2f}°")
    print(f"  Max abs error: {np.max(np.abs(errors_deg)):.2f}°")
    print(f"  Mean cost: {np.mean(controller.log['cost']):.3f}")
    print(f"  Mean comp time: {np.mean(controller.log['comp_time'])*1000:.1f}ms")

def plot_results(controllers_dict, output_path):
    """複数制御手法の結果をプロット"""
    if not _HAS_PLT:
        return
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    for name, ctrl in controllers_dict.items():
        t = np.array(ctrl.log['t'])
        
        # Theta
        axes[0].plot(t, np.degrees(ctrl.log['theta']), label=name, linewidth=1.5)
        
        # Pressures
        axes[1].plot(t, ctrl.log['p1'], label=f'{name} p1', linewidth=1)
        axes[1].plot(t, ctrl.log['p2'], '--', label=f'{name} p2', linewidth=1)
        
        # Error
        axes[2].plot(t, np.degrees(ctrl.log['error']), label=name, linewidth=1)
        
        # Computation time
        axes[3].plot(t, np.array(ctrl.log['comp_time'])*1000, label=name, linewidth=1)
    
    # Reference
    if len(controllers_dict) > 0:
        first_ctrl = list(controllers_dict.values())[0]
        t = np.array(first_ctrl.log['t'])
        axes[0].plot(t, np.degrees(first_ctrl.log['theta_ref']), 'k--', 
                    label='reference', linewidth=2, alpha=0.5)
    
    axes[0].set_ylabel('Angle [deg]')
    axes[0].set_title('Theta Tracking')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('Pressure [MPa]')
    axes[1].set_title('Control Inputs')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_ylabel('Error [deg]')
    axes[2].set_title('Tracking Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    
    axes[3].set_ylabel('Time [ms]')
    axes[3].set_xlabel('Time [s]')
    axes[3].set_title('Computation Time')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.replace('.png', '.pdf'))
    print(f"[Saved] {output_path}")
    plt.close()

# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--controllers', nargs='+',
                       default=['mppi', 'inverse', 'pid', 'lqi', 'cem', 'random_shooting'],
                       choices=['mppi', 'inverse', 'pid', 'lqi', 'cem', 'random_shooting'])
    parser.add_argument('--theta_target_deg', type=float, default=30.0)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--out_dir', type=str, default='narx_control_results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f" NARX Control Methods Comparison")
    print(f"{'='*70}")
    print(f"Model: {args.model_dir}")
    print(f"Controllers: {args.controllers}")
    print(f"Device: {device}")
    
    # Run simulations
    controllers_dict = {}
    results = {}
    
    for ctrl_name in args.controllers:
        print(f"\n{'='*70}")
        print(f" Controller: {ctrl_name.upper()}")
        print(f"{'='*70}")
        
        if ctrl_name == 'mppi':
            ctrl = NARX_MPPI(args.model_dir, args.dt, device, K=32, H=15)
        elif ctrl_name == 'inverse':
            ctrl = NARX_InverseMapping(args.model_dir, args.dt, device, n_iter=50)
        elif ctrl_name == 'pid':
            ctrl = NARX_PID(args.model_dir, args.dt, device, Kp=0.5, Ki=0.05, Kd=0.1)
        elif ctrl_name == 'lqi':
            ctrl = NARX_LQI(args.model_dir, args.dt, device, Q_theta=10.0, R=0.1)
        elif ctrl_name == 'cem':
            ctrl = NARX_CEM(args.model_dir, args.dt, device, K=64, H=15)
        elif ctrl_name == 'random_shooting':
            ctrl = NARX_RandomShooting(args.model_dir, args.dt, device, K=64, H=15)
        
        run_simulation(ctrl, args.theta_target_deg, args.steps)
        
        controllers_dict[ctrl_name] = ctrl
        
        # Save results
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
            'final_error': float(errors_deg[-1]),
            'mean_cost': float(np.mean(ctrl.log['cost'])),
            'mean_comp_time_ms': float(np.mean(ctrl.log['comp_time']) * 1000)
        }
    
    # Plot comparison
    plot_results(controllers_dict, os.path.join(args.out_dir, 'comparison.png'))
    
    # Save summary
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f" SUMMARY")
    print(f"{'='*70}")
    print(f"{'Controller':<20} {'RMSE(°)':<10} {'MAE(°)':<10} {'Max Err(°)':<12} {'Comp(ms)':<10}")
    print(f"{'-'*70}")
    
    for name in args.controllers:
        r = results[name]
        print(f"{name:<20} {r['rmse']:<10.3f} {r['mae']:<10.3f} "
              f"{r['max_abs_error']:<12.3f} {r['mean_comp_time_ms']:<10.1f}")
    
    print(f"\n[Saved] {args.out_dir}/")

if __name__ == '__main__':
    main()