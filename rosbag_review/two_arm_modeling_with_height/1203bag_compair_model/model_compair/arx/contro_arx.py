#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
control_arx.py
線形ARXモデル用の制御スクリプト

線形モデルなので以下が使える:
1. Analytical Control (解析的制御) - 1ステップ先の最適入力
2. LQR (Linear Quadratic Regulator)
3. MPC (Model Predictive Control) - 線形計画問題
4. PID
5. MPPI (参考用)
"""
import os, json, math, argparse, time
import numpy as np
import pandas as pd
from collections import deque
import pickle

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except:
    _HAS_PLT = False

try:
    from scipy.linalg import solve_discrete_are
    from scipy.optimize import minimize, linprog
    _HAS_SCIPY = True
except:
    _HAS_SCIPY = False
    print("[WARNING] scipy not available, some controllers will be disabled")

# ==================== Base Controller ====================
class BaseARXController:
    """ARXコントローラーのベースクラス"""
    
    def __init__(self, model_dir, dt=0.01):
        self.load_model(model_dir)
        self.dt = dt
        
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
        
        # Logging
        self.log = {
            't': [], 'theta': [], 'theta_ref': [], 'error': [],
            'p1': [], 'p2': [], 'cost': [], 'comp_time': []
        }
    
    def load_model(self, model_dir):
        """モデルロード"""
        with open(os.path.join(model_dir, 'arx_meta.json'), 'r') as f:
            self.meta = json.load(f)
        
        with open(os.path.join(model_dir, 'arx_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        
        self.lags = self.meta['lags']
        self.coef = np.array(self.meta['coefficients']['weights'])
        self.intercept = self.meta['coefficients']['intercept']
        
        # Extract coefficient groups
        # [θ_{t-1}, ..., θ_{t-lags}, p1_{t-1}, ..., p1_{t-lags}, p2_{t-1}, ..., p2_{t-lags}]
        self.a_coef = self.coef[:self.lags]  # θ係数
        self.b_coef = self.coef[self.lags:2*self.lags]  # p1係数
        self.c_coef = self.coef[2*self.lags:3*self.lags]  # p2係数
        
        print(f"[Model] Loaded ARX: lags={self.lags}")
        print(f"  θ coef range: [{self.a_coef.min():.4f}, {self.a_coef.max():.4f}]")
        print(f"  p1 coef range: [{self.b_coef.min():.4f}, {self.b_coef.max():.4f}]")
        print(f"  p2 coef range: [{self.c_coef.min():.4f}, {self.c_coef.max():.4f}]")
    
    def enforce_constraints(self, p1, p2, p1_prev, p2_prev):
        dp_max_step = self.dp_max * self.dt
        p1 = np.clip(p1, p1_prev - dp_max_step, p1_prev + dp_max_step)
        p2 = np.clip(p2, p2_prev - dp_max_step, p2_prev + dp_max_step)
        p1 = np.clip(p1, 0.0, self.p_max)
        p2 = np.clip(p2, 0.0, self.p_max)
        return p1, p2
    
    def predict(self, theta_hist, p1_hist, p2_hist):
        """1ステップ予測 (線形)"""
        # Feature vector
        x = np.concatenate([theta_hist[:self.lags],
                           p1_hist[:self.lags],
                           p2_hist[:self.lags]])
        
        theta_next = np.dot(self.coef, x) + self.intercept
        return float(theta_next)
    
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
        
        # Predict next state
        self.hist_theta.appendleft(self.theta_rad)
        self.hist_p1.appendleft(p1_cmd)
        self.hist_p2.appendleft(p2_cmd)
        
        theta_next = self.predict(
            list(self.hist_theta)[:self.lags],
            list(self.hist_p1)[:self.lags],
            list(self.hist_p2)[:self.lags]
        )
        
        # Update state
        self.theta_rad = theta_next
        self.p1_cmd = p1_cmd
        self.p2_cmd = p2_cmd
        
        comp_time = time.time() - t_start
        
        return theta_next, p1_cmd, p2_cmd, cost, comp_time

# ==================== 1. Analytical Control ====================
class ARX_Analytical(BaseARXController):
    """解析的制御 - 1ステップ先が目標になるよう逆算"""
    
    def __init__(self, model_dir, dt=0.01, lambda_reg=0.1):
        super().__init__(model_dir, dt)
        self.lambda_reg = lambda_reg
        
        print(f"[Analytical] Regularization={lambda_reg}")
    
    def compute_control(self, theta_ref):
        """解析的に最適入力を計算
        
        θ_{t+1} = Σ a_i θ_{t-i} + Σ b_i p1_{t-i} + Σ c_i p2_{t-i} + d
        
        最新の入力 p1_t, p2_t を決定:
        θ_ref = a_0*θ_t + Σ_{i=1} a_i θ_{t-i} + b_0*p1_t + Σ_{i=1} b_i p1_{t-i} 
                + c_0*p2_t + Σ_{i=1} c_i p2_{t-i} + d
        
        → 2変数の線形方程式
        """
        # 過去の寄与
        theta_past = sum(self.a_coef[i] * list(self.hist_theta)[i] 
                        for i in range(self.lags))
        p1_past = sum(self.b_coef[i] * list(self.hist_p1)[i] 
                     for i in range(1, self.lags))
        p2_past = sum(self.c_coef[i] * list(self.hist_p2)[i] 
                     for i in range(1, self.lags))
        
        # 必要な入力の総和
        needed = theta_ref - theta_past - p1_past - p2_past - self.intercept
        
        # b_0*p1_t + c_0*p2_t = needed
        # 拮抗駆動の仮定: p1 ↑ → θ ↑, p2 ↑ → θ ↓
        # つまり b_0 > 0, c_0 < 0 が期待される
        
        b0 = self.b_coef[0]
        c0 = self.c_coef[0]
        
        # 正則化付き最小ノルム解
        # min ||[p1, p2]||^2 s.t. b0*p1 + c0*p2 = needed
        
        if abs(b0) + abs(c0) < 1e-6:
            # 係数が小さすぎる場合
            return self.p1_cmd, self.p2_cmd, abs(theta_ref - self.theta_rad)
        
        # 疑似逆行列的なアプローチ
        A = np.array([[b0, c0]])
        b = np.array([needed])
        
        # 正則化付き最小二乗
        # u = A^T (A A^T + λI)^{-1} b
        AAt = np.dot(A, A.T) + self.lambda_reg * np.eye(1)
        u = np.dot(A.T, np.linalg.solve(AAt, b))
        
        p1_cmd = float(u[0])
        p2_cmd = float(u[1])
        
        # 現在値からあまり離れないように
        p1_cmd = 0.7 * p1_cmd + 0.3 * self.p1_cmd
        p2_cmd = 0.7 * p2_cmd + 0.3 * self.p2_cmd
        
        cost = abs(theta_ref - self.theta_rad)
        
        return p1_cmd, p2_cmd, cost

# ==================== 2. LQR Controller ====================
class ARX_LQR(BaseARXController):
    """LQR制御 - 状態空間表現に変換してARE解く"""
    
    def __init__(self, model_dir, dt=0.01, Q=10.0, R=0.1):
        super().__init__(model_dir, dt)
        
        if not _HAS_SCIPY:
            raise ImportError("scipy required for LQR")
        
        self.Q_weight = Q
        self.R_weight = R
        
        # State-space form: x_{t+1} = A x_t + B u_t
        # x = [θ_t, θ_{t-1}, ..., p1_t, p1_{t-1}, ..., p2_t, p2_{t-1}, ...]
        self.setup_state_space()
        
        # Solve ARE
        self.compute_lqr_gain()
        
        print(f"[LQR] Q={Q}, R={R}")
    
    def setup_state_space(self):
        """状態空間表現を構築"""
        # 状態ベクトル: [θ, θ_{-1}, ..., p1, p1_{-1}, ..., p2, p2_{-1}, ...]
        # 入力: [p1_new, p2_new]
        
        n_states = 3 * self.lags  # θ, p1, p2 それぞれ lags 個
        n_inputs = 2  # p1, p2
        
        # A matrix
        A = np.zeros((n_states, n_states))
        
        # θ の更新: θ_{t+1} = Σ a_i θ_{t-i} + ...
        A[0, :self.lags] = self.a_coef
        A[0, self.lags:2*self.lags] = self.b_coef
        A[0, 2*self.lags:3*self.lags] = self.c_coef
        
        # θ のシフト: θ_{t} → θ_{t-1}
        for i in range(1, self.lags):
            A[i, i-1] = 1.0
        
        # p1 のシフト
        for i in range(self.lags, 2*self.lags-1):
            A[i+1, i] = 1.0
        
        # p2 のシフト
        for i in range(2*self.lags, 3*self.lags-1):
            A[i+1, i] = 1.0
        
        # B matrix
        B = np.zeros((n_states, n_inputs))
        B[self.lags, 0] = 1.0  # p1_new → p1 の最新値
        B[2*self.lags, 1] = 1.0  # p2_new → p2 の最新値
        
        self.A = A
        self.B = B
        self.n_states = n_states
        self.n_inputs = n_inputs
    
    def compute_lqr_gain(self):
        """LQRゲインを計算"""
        # Q matrix (状態コスト)
        Q = np.zeros((self.n_states, self.n_states))
        Q[0, 0] = self.Q_weight  # θ の誤差にペナルティ
        
        # R matrix (入力コスト)
        R = self.R_weight * np.eye(self.n_inputs)
        
        # Solve discrete-time ARE
        P = solve_discrete_are(self.A, self.B, Q, R)
        
        # Compute LQR gain: K = (R + B^T P B)^{-1} B^T P A
        BTPsB = self.B.T @ P @ self.B + R
        self.K = np.linalg.solve(BTPsB, self.B.T @ P @ self.A)
        
        print(f"  LQR gain shape: {self.K.shape}")
        print(f"  Gain magnitude: {np.linalg.norm(self.K):.3f}")
    
    def get_state_vector(self, theta_ref):
        """現在の状態ベクトルを取得 (目標からの偏差)"""
        x = np.zeros(self.n_states)
        
        # θ (目標からの偏差)
        for i in range(self.lags):
            x[i] = list(self.hist_theta)[i] - theta_ref
        
        # p1
        for i in range(self.lags):
            x[self.lags + i] = list(self.hist_p1)[i]
        
        # p2
        for i in range(self.lags):
            x[2*self.lags + i] = list(self.hist_p2)[i]
        
        return x
    
    def compute_control(self, theta_ref):
        """LQR制御"""
        x = self.get_state_vector(theta_ref)
        
        # u = -K x
        u = -self.K @ x
        
        p1_cmd = float(u[0])
        p2_cmd = float(u[1])
        
        # クリップ
        p1_cmd = np.clip(p1_cmd, 0, self.p_max)
        p2_cmd = np.clip(p2_cmd, 0, self.p_max)
        
        cost = abs(x[0])  # θ の偏差
        
        return p1_cmd, p2_cmd, cost

# ==================== 3. MPC Controller ====================
class ARX_MPC(BaseARXController):
    """MPC制御 - 線形計画問題として解く"""
    
    def __init__(self, model_dir, dt=0.01, horizon=10, 
                 Q=10.0, R=0.1):
        super().__init__(model_dir, dt)
        
        if not _HAS_SCIPY:
            raise ImportError("scipy required for MPC")
        
        self.horizon = horizon
        self.Q = Q
        self.R = R
        
        print(f"[MPC] Horizon={horizon}, Q={Q}, R={R}")
    
    def compute_control(self, theta_ref):
        """MPC - ホライゾン内で最適化"""
        from scipy.optimize import minimize
        
        H = self.horizon
        
        # 初期推定値 (現在値を維持)
        u0 = np.array([self.p1_cmd, self.p2_cmd] * H)
        
        def cost_function(u):
            """コスト関数"""
            cost = 0.0
            
            theta = self.theta_rad
            theta_hist = list(self.hist_theta)[:self.lags]
            p1_hist = list(self.hist_p1)[:self.lags]
            p2_hist = list(self.hist_p2)[:self.lags]
            
            for h in range(H):
                p1 = u[2*h]
                p2 = u[2*h + 1]
                
                # Predict
                theta_hist = [theta] + theta_hist[:self.lags-1]
                p1_hist = [p1] + p1_hist[:self.lags-1]
                p2_hist = [p2] + p2_hist[:self.lags-1]
                
                x = np.concatenate([theta_hist[:self.lags],
                                   p1_hist[:self.lags],
                                   p2_hist[:self.lags]])
                theta = np.dot(self.coef, x) + self.intercept
                
                # Cost
                cost += self.Q * (theta - theta_ref)**2
                cost += self.R * (p1**2 + p2**2)
            
            return cost
        
        # Constraints
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

# ==================== 4. PID Controller ====================
class ARX_PID(BaseARXController):
    """PID制御"""
    
    def __init__(self, model_dir, dt=0.01, Kp=0.5, Ki=0.05, Kd=0.1):
        super().__init__(model_dir, dt)
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
        
        # Map to pressure
        if u > 0:
            p1_cmd = self.p1_cmd + abs(u) * 0.1
            p2_cmd = max(0, self.p2_cmd - abs(u) * 0.1)
        else:
            p1_cmd = max(0, self.p1_cmd - abs(u) * 0.1)
            p2_cmd = self.p2_cmd + abs(u) * 0.1
        
        self.prev_error = error
        
        return p1_cmd, p2_cmd, abs(error)

# ==================== 5. MPPI (参考) ====================
class ARX_MPPI(BaseARXController):
    """MPPI制御 (参考実装)"""
    
    def __init__(self, model_dir, dt=0.01, K=32, H=15, lam=2.0, sigma_u=0.10):
        super().__init__(model_dir, dt)
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
            theta = self.theta_rad
            theta_hist = list(self.hist_theta)[:self.lags]
            p1_hist = list(self.hist_p1)[:self.lags]
            p2_hist = list(self.hist_p2)[:self.lags]
            
            for h in range(self.H):
                p1 = p1_hist[0] + U[k, h, 0]
                p2 = p2_hist[0] + U[k, h, 1]
                
                p1 = np.clip(p1, 0, self.p_max)
                p2 = np.clip(p2, 0, self.p_max)
                
                theta_hist = [theta] + theta_hist[:self.lags-1]
                p1_hist = [p1] + p1_hist[:self.lags-1]
                p2_hist = [p2] + p2_hist[:self.lags-1]
                
                x = np.concatenate([theta_hist[:self.lags],
                                   p1_hist[:self.lags],
                                   p2_hist[:self.lags]])
                theta = np.dot(self.coef, x) + self.intercept
                
                err = theta_ref - theta
                cost += 30.0 * err**2 + 0.01 * (p1**2 + p2**2)
            
            J[k] = cost
        
        beta = np.min(J)
        w = np.exp(-(J - beta) / max(1e-6, self.temperature))
        w = w / (np.sum(w) + 1e-9)
        
        dU = np.sum(w[:, None, None] * U, axis=0)
        
        return self.p1_cmd + dU[0, 0], self.p2_cmd + dU[0, 1], float(np.min(J))

# ==================== Simulation & Plotting ====================
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
    print(f"  Max abs error: {np.max(np.abs(errors_deg)):.2f}°")

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
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    
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
                       default=['analytical', 'lqr', 'mpc', 'pid', 'mppi'],
                       choices=['analytical', 'lqr', 'mpc', 'pid', 'mppi'])
    parser.add_argument('--theta_target_deg', type=float, default=30.0)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--out_dir', type=str, default='arx_control_results')
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f" ARX Control Methods Comparison")
    print(f"{'='*70}")
    
    controllers_dict = {}
    results = {}
    
    for ctrl_name in args.controllers:
        print(f"\n{'='*70}")
        print(f" Controller: {ctrl_name.upper()}")
        print(f"{'='*70}")
        
        try:
            if ctrl_name == 'analytical':
                ctrl = ARX_Analytical(args.model_dir, args.dt, lambda_reg=0.1)
            elif ctrl_name == 'lqr':
                ctrl = ARX_LQR(args.model_dir, args.dt, Q=10.0, R=0.1)
            elif ctrl_name == 'mpc':
                ctrl = ARX_MPC(args.model_dir, args.dt, horizon=10, Q=10.0, R=0.1)
            elif ctrl_name == 'pid':
                ctrl = ARX_PID(args.model_dir, args.dt, Kp=0.5, Ki=0.05, Kd=0.1)
            elif ctrl_name == 'mppi':
                ctrl = ARX_MPPI(args.model_dir, args.dt, K=32, H=15)
            
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
                'mean_comp_time_ms': float(np.mean(ctrl.log['comp_time']) * 1000)
            }
        
        except Exception as e:
            print(f"[ERROR] {ctrl_name}: {e}")
    
    # Plot comparison
    plot_results(controllers_dict, os.path.join(args.out_dir, 'comparison.png'))
    
    # Save summary
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f" SUMMARY")
    print(f"{'='*70}")
    print(f"{'Controller':<15} {'RMSE(°)':<10} {'MAE(°)':<10} {'Comp(ms)':<10}")
    print(f"{'-'*50}")
    
    for name in controllers_dict.keys():
        r = results[name]
        print(f"{name:<15} {r['rmse']:<10.3f} {r['mae']:<10.3f} {r['mean_comp_time_ms']:<10.1f}")

if __name__ == '__main__':
    main()