#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPPI Parameter Optimizer

Usage:
  # Optuna (Bayesian optimization) - 推奨
  python mppi_parameter_optimizer.py \
      --model-dir models/narx_p1p2_production2 \
      --method optuna \
      --n-trials 50 \
      --theta-target-deg 30

  # Grid search (徹底的だが遅い)
  python mppi_parameter_optimizer.py \
      --model-dir models/narx_p1p2_production2 \
      --method grid \
      --theta-target-deg 30

  # Random search (速い)
  python mppi_parameter_optimizer.py \
      --model-dir models/narx_p1p2_production2 \
      --method random \
      --n-trials 30 \
      --theta-target-deg 30
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Import the original MPPI simulator
# Assuming inverse7_2_narx_mppi_p1p2.py is in the same directory
try:
    from inverse7_2_narx_mppi_p1p2 import NARX_MPPI_Simulator
except ImportError:
    print("Error: inverse7_2_narx_mppi_p1p2.py must be in the same directory")
    sys.exit(1)


class ParameterOptimizer:
    """MPPIパラメータの最適化クラス"""
    
    def __init__(self, args):
        self.args = args
        self.results = []
        
        # 評価メトリクス
        self.best_params = None
        self.best_score = float('inf')
        
        print(f"\n{'='*60}")
        print(f"MPPI Parameter Optimizer")
        print(f"{'='*60}")
        print(f"Model: {args.model_dir}")
        print(f"Method: {args.method}")
        print(f"Target: {args.theta_target_deg}°")
        print(f"{'='*60}\n")
    
    def create_simulator_args(self, params):
        """パラメータからシミュレータ用のargsを作成"""
        class SimArgs:
            pass
        
        sim_args = SimArgs()
        sim_args.model_dir = self.args.model_dir
        sim_args.steps = self.args.steps
        sim_args.dt = self.args.dt
        sim_args.frame_skip = self.args.frame_skip
        
        # Optimizable parameters
        sim_args.K = int(params['K'])
        sim_args.horizon = int(params['horizon'])
        sim_args.lam = params['lambda']
        sim_args.sigma_u = params['sigma_u']
        sim_args.w_tracking = params['w_tracking']
        sim_args.w_smooth = params['w_smooth']
        sim_args.w_effort = params['w_effort']
        sim_args.w_constraint = params['w_constraint']
        
        # Fixed parameters
        sim_args.p_max = self.args.p_max
        sim_args.dp_max = self.args.dp_max
        sim_args.context_csv = self.args.context_csv
        
        return sim_args
    
    def evaluate_params(self, params, trial_id=None):
        """パラメータセットを評価"""
        try:
            # Create simulator with these parameters
            sim_args = self.create_simulator_args(params)
            sim = NARX_MPPI_Simulator(sim_args)
            
            # Initialize
            if self.args.context_csv:
                sim.initialize_from_context(self.args.context_csv)
            
            # Run simulation
            import math
            theta_target_rad = math.radians(self.args.theta_target_deg)
            sim.run_simulation(theta_target_rad, self.args.steps, smooth_ref=True)
            
            # Calculate metrics
            errors_deg = np.degrees(np.array(sim.log_error))
            
            # Primary metric: RMS error
            rms_error = np.sqrt(np.mean(errors_deg**2))
            
            # Secondary metrics
            max_abs_error = np.max(np.abs(errors_deg))
            final_error = np.abs(errors_deg[-1])
            settling_time = self.calculate_settling_time(errors_deg)
            overshoot = self.calculate_overshoot(np.array(sim.log_theta), theta_target_rad)
            
            # Control smoothness (rate of change)
            p1_cmd = np.array(sim.log_p1_cmd)
            p2_cmd = np.array(sim.log_p2_cmd)
            smoothness = np.mean(np.abs(np.diff(p1_cmd)) + np.abs(np.diff(p2_cmd)))
            
            # Composite score (weighted sum)
            score = (
                1.0 * rms_error +           # Primary: RMS error
                0.3 * max_abs_error +        # Peak error
                0.5 * final_error +          # Steady-state error
                0.2 * settling_time +        # Speed of convergence
                0.1 * overshoot +            # Overshoot penalty
                0.05 * smoothness            # Control smoothness
            )
            
            result = {
                'trial_id': trial_id,
                'params': params.copy(),
                'rms_error': rms_error,
                'max_abs_error': max_abs_error,
                'final_error': final_error,
                'settling_time': settling_time,
                'overshoot': overshoot,
                'smoothness': smoothness,
                'score': score
            }
            
            self.results.append(result)
            
            # Update best
            if score < self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                print(f"\n🎯 New best score: {score:.3f}")
                print(f"   RMS error: {rms_error:.2f}°")
                print(f"   Final error: {final_error:.2f}°")
                print(f"   Params: {params}")
            
            return score, result
            
        except Exception as e:
            print(f"❌ Trial failed: {e}")
            return float('inf'), None
    
    def calculate_settling_time(self, errors_deg, threshold=2.0):
        """整定時間を計算（誤差がthreshold以下になる時間）"""
        try:
            settling_idx = np.where(np.abs(errors_deg) <= threshold)[0]
            if len(settling_idx) > 0:
                # Last time error exceeded threshold
                last_exceed = np.where(np.abs(errors_deg) > threshold)[0]
                if len(last_exceed) > 0:
                    return float(last_exceed[-1])
            return float(len(errors_deg))
        except:
            return float(len(errors_deg))
    
    def calculate_overshoot(self, theta_log, theta_target):
        """オーバーシュートを計算"""
        try:
            theta_log_deg = np.degrees(theta_log)
            theta_target_deg = np.degrees(theta_target)
            
            if theta_target_deg > theta_log_deg[0]:
                # Positive direction
                overshoot = np.max(theta_log_deg - theta_target_deg)
            else:
                # Negative direction
                overshoot = np.max(theta_target_deg - theta_log_deg)
            
            return max(0.0, overshoot)
        except:
            return 0.0
    
    def optimize_optuna(self):
        """Optuna (Bayesian optimization)"""
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            print("❌ Optuna not installed. Install with: pip install optuna")
            return
        
        def objective(trial):
            params = {
                'K': trial.suggest_int('K', 16, 128, step=16),
                'horizon': trial.suggest_int('horizon', 8, 30, step=2),
                'lambda': trial.suggest_float('lambda', 0.5, 10.0, log=True),
                'sigma_u': trial.suggest_float('sigma_u', 0.03, 0.3, log=True),
                'w_tracking': trial.suggest_float('w_tracking', 10.0, 100.0, log=True),
                'w_smooth': trial.suggest_float('w_smooth', 0.01, 1.0, log=True),
                'w_effort': trial.suggest_float('w_effort', 0.001, 0.1, log=True),
                'w_constraint': trial.suggest_float('w_constraint', 100.0, 1000.0, log=True),
            }
            
            score, _ = self.evaluate_params(params, trial.number)
            return score
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(
            objective,
            n_trials=self.args.n_trials,
            show_progress_bar=True
        )
        
        print(f"\n{'='*60}")
        print(f"Optimization completed!")
        print(f"Best score: {study.best_value:.3f}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")
        
        return study
    
    def optimize_random(self):
        """Random search"""
        print(f"\nStarting random search with {self.args.n_trials} trials...\n")
        
        for trial_id in range(self.args.n_trials):
            params = {
                'K': int(np.random.choice([16, 32, 48, 64, 96, 128])),
                'horizon': int(np.random.choice([8, 10, 12, 15, 18, 20, 25, 30])),
                'lambda': float(np.exp(np.random.uniform(np.log(0.5), np.log(10.0)))),
                'sigma_u': float(np.exp(np.random.uniform(np.log(0.03), np.log(0.3)))),
                'w_tracking': float(np.exp(np.random.uniform(np.log(10.0), np.log(100.0)))),
                'w_smooth': float(np.exp(np.random.uniform(np.log(0.01), np.log(1.0)))),
                'w_effort': float(np.exp(np.random.uniform(np.log(0.001), np.log(0.1)))),
                'w_constraint': float(np.exp(np.random.uniform(np.log(100.0), np.log(1000.0)))),
            }
            
            print(f"Trial {trial_id + 1}/{self.args.n_trials}")
            self.evaluate_params(params, trial_id)
    
    def optimize_grid(self):
        """Grid search (coarse grid)"""
        print(f"\nStarting grid search...\n")
        
        # Define coarse grid
        grid = {
            'K': [32, 64],
            'horizon': [10, 15, 20],
            'lambda': [1.0, 2.0, 5.0],
            'sigma_u': [0.05, 0.10, 0.15],
            'w_tracking': [20.0, 30.0, 50.0],
            'w_smooth': [0.05, 0.1],
            'w_effort': [0.01, 0.05],
            'w_constraint': [300.0, 500.0],
        }
        
        # Generate all combinations
        import itertools
        keys = grid.keys()
        values = grid.values()
        combinations = list(itertools.product(*values))
        
        total = len(combinations)
        print(f"Total combinations: {total}\n")
        
        for trial_id, combination in enumerate(combinations):
            params = dict(zip(keys, combination))
            print(f"Trial {trial_id + 1}/{total}")
            self.evaluate_params(params, trial_id)
    
    def save_results(self):
        """結果を保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        results_dir = Path("optimization_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save all results to CSV
        if self.results:
            records = []
            for r in self.results:
                record = {
                    'trial_id': r['trial_id'],
                    'score': r['score'],
                    'rms_error': r['rms_error'],
                    'max_abs_error': r['max_abs_error'],
                    'final_error': r['final_error'],
                    'settling_time': r['settling_time'],
                    'overshoot': r['overshoot'],
                    'smoothness': r['smoothness'],
                }
                record.update(r['params'])
                records.append(record)
            
            df = pd.DataFrame(records)
            csv_path = results_dir / f"optimization_results_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n📊 Results saved to: {csv_path}")
            
            # Save best parameters as JSON
            if self.best_params:
                best_path = results_dir / f"best_params_{timestamp}.json"
                with open(best_path, 'w') as f:
                    json.dump({
                        'best_score': float(self.best_score),
                        'best_params': self.best_params,
                        'method': self.args.method,
                        'target_deg': self.args.theta_target_deg,
                        'timestamp': timestamp
                    }, f, indent=2)
                print(f"🏆 Best parameters saved to: {best_path}")
                
                # Print best parameters
                print(f"\n{'='*60}")
                print(f"BEST PARAMETERS (Score: {self.best_score:.3f})")
                print(f"{'='*60}")
                for key, value in self.best_params.items():
                    print(f"  --{key.replace('_', '-')} {value}")
                print(f"{'='*60}\n")
    
    def plot_results(self):
        """結果を可視化"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Score history
        ax = axes[0, 0]
        scores = [r['score'] for r in self.results]
        ax.plot(scores, 'o-', alpha=0.6)
        ax.axhline(y=self.best_score, color='r', linestyle='--', 
                   label=f'Best: {self.best_score:.2f}')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Score')
        ax.set_title('Optimization Progress')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # RMS error distribution
        ax = axes[0, 1]
        rms_errors = [r['rms_error'] for r in self.results]
        ax.hist(rms_errors, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(rms_errors), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(rms_errors):.2f}°')
        ax.set_xlabel('RMS Error [deg]')
        ax.set_ylabel('Count')
        ax.set_title('RMS Error Distribution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Parameter correlation (K vs horizon)
        ax = axes[1, 0]
        K_values = [r['params']['K'] for r in self.results]
        horizon_values = [r['params']['horizon'] for r in self.results]
        scatter = ax.scatter(K_values, horizon_values, c=scores, 
                           cmap='viridis_r', alpha=0.6, s=100)
        ax.set_xlabel('K (samples)')
        ax.set_ylabel('Horizon')
        ax.set_title('Parameter Space (K vs Horizon)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Score')
        
        # Top metrics comparison
        ax = axes[1, 1]
        if len(self.results) >= 5:
            # Get top 5 results
            sorted_results = sorted(self.results, key=lambda x: x['score'])[:5]
            metrics = ['rms_error', 'max_abs_error', 'final_error', 'settling_time']
            
            x = np.arange(len(metrics))
            width = 0.15
            
            for i, r in enumerate(sorted_results):
                values = [r[m] for m in metrics]
                ax.bar(x + i*width, values, width, label=f"Trial {r['trial_id']}", alpha=0.7)
            
            ax.set_xlabel('Metric')
            ax.set_ylabel('Value')
            ax.set_title('Top 5 Trials - Metrics Comparison')
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels(['RMS', 'Max', 'Final', 'Settling'])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = Path("optimization_results") / f"optimization_plot_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📈 Plot saved to: {plot_path}")
        
        if self.args.plot:
            plt.show()
    
    def run(self):
        """最適化を実行"""
        if self.args.method == 'optuna':
            study = self.optimize_optuna()
        elif self.args.method == 'random':
            self.optimize_random()
        elif self.args.method == 'grid':
            self.optimize_grid()
        else:
            raise ValueError(f"Unknown method: {self.args.method}")
        
        self.save_results()
        self.plot_results()


def main():
    parser = argparse.ArgumentParser(
        description='MPPI Parameter Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model and target
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing NARX model')
    parser.add_argument('--theta-target-deg', type=float, default=30.0,
                        help='Target angle in degrees')
    
    # Optimization method
    parser.add_argument('--method', type=str, 
                        choices=['optuna', 'random', 'grid'],
                        default='optuna',
                        help='Optimization method')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials (for optuna/random)')
    
    # Simulation settings
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of simulation steps per trial')
    parser.add_argument('--dt', type=float, default=0.0,
                        help='Time step (0 to use model default)')
    parser.add_argument('--frame-skip', type=int, default=2,
                        help='Frame skip for control')
    
    # Fixed physical constraints
    parser.add_argument('--p-max', type=float, default=0.70,
                        help='Maximum pressure [MPa]')
    parser.add_argument('--dp-max', type=float, default=3.5,
                        help='Maximum pressure rate [MPa/s]')
    
    # Context
    parser.add_argument('--context-csv', type=str, default='',
                        help='CSV file for initial context')
    
    # Output
    parser.add_argument('--plot', action='store_true',
                        help='Show plots')
    
    args = parser.parse_args()
    
    optimizer = ParameterOptimizer(args)
    optimizer.run()


if __name__ == '__main__':
    main()