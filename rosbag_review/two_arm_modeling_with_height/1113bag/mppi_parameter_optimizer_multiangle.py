#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPPI Parameter Optimizer (Multi-Angle Evaluation)

複数の目標角度で評価し、ロバストなパラメータを発見します。

Usage:
  # デフォルト（7つの角度で評価）
  python mppi_parameter_optimizer_multiangle.py \
      --model-dir models/narx_p1p2_production2 \
      --method optuna \
      --n-trials 50

  # カスタム角度セット
  python mppi_parameter_optimizer_multiangle.py \
      --model-dir models/narx_p1p2_production2 \
      --method optuna \
      --n-trials 50 \
      --test-angles-deg -30 -15 0 15 30

  # より細かい評価（時間がかかる）
  python mppi_parameter_optimizer_multiangle.py \
      --model-dir models/narx_p1p2_production2 \
      --method optuna \
      --n-trials 50 \
      --test-angles-deg -30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30
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
from collections import defaultdict

# Import the original MPPI simulator
try:
    from inverse7_2_narx_mppi_p1p2 import NARX_MPPI_Simulator
except ImportError:
    print("Error: inverse7_2_narx_mppi_p1p2.py must be in the same directory")
    sys.exit(1)


class MultiAngleParameterOptimizer:
    """複数角度でMPPIパラメータを最適化"""
    
    def __init__(self, args):
        self.args = args
        self.results = []
        
        # 評価角度のリスト
        if args.test_angles_deg:
            self.test_angles = args.test_angles_deg
        else:
            # デフォルト: -30°から30°まで7点
            self.test_angles = [-30, -20, -10, 0, 10, 20, 30]
        
        # 評価メトリクス
        self.best_params = None
        self.best_score = float('inf')
        
        print(f"\n{'='*70}")
        print(f"MPPI Parameter Optimizer (Multi-Angle Evaluation)")
        print(f"{'='*70}")
        print(f"Model: {args.model_dir}")
        print(f"Method: {args.method}")
        print(f"Test angles: {self.test_angles}°")
        print(f"Number of test angles: {len(self.test_angles)}")
        print(f"{'='*70}\n")
    
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
    
    def evaluate_single_angle(self, sim_args, theta_target_deg):
        """単一の角度でシミュレーションを実行して評価"""
        try:
            sim = NARX_MPPI_Simulator(sim_args)
            
            # Initialize
            if self.args.context_csv:
                sim.initialize_from_context(self.args.context_csv)
            
            # Run simulation
            import math
            theta_target_rad = math.radians(theta_target_deg)
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
            
            # Control smoothness
            p1_cmd = np.array(sim.log_p1_cmd)
            p2_cmd = np.array(sim.log_p2_cmd)
            smoothness = np.mean(np.abs(np.diff(p1_cmd)) + np.abs(np.diff(p2_cmd)))
            
            # Composite score
            score = (
                1.0 * rms_error +
                0.3 * max_abs_error +
                0.5 * final_error +
                0.2 * settling_time +
                0.1 * overshoot +
                0.05 * smoothness
            )
            
            return {
                'success': True,
                'rms_error': rms_error,
                'max_abs_error': max_abs_error,
                'final_error': final_error,
                'settling_time': settling_time,
                'overshoot': overshoot,
                'smoothness': smoothness,
                'score': score
            }
            
        except Exception as e:
            print(f"    ❌ Failed at {theta_target_deg}°: {e}")
            return {
                'success': False,
                'score': float('inf')
            }
    
    def evaluate_params(self, params, trial_id=None):
        """パラメータセットを複数角度で評価"""
        sim_args = self.create_simulator_args(params)
        
        print(f"\n{'─'*70}")
        if trial_id is not None:
            print(f"Trial {trial_id} - Testing {len(self.test_angles)} angles...")
        else:
            print(f"Testing {len(self.test_angles)} angles...")
        
        # 各角度で評価
        angle_results = {}
        valid_scores = []
        
        for angle in self.test_angles:
            print(f"  Testing θ={angle:+5.0f}°...", end=' ', flush=True)
            result = self.evaluate_single_angle(sim_args, angle)
            angle_results[angle] = result
            
            if result['success']:
                valid_scores.append(result['score'])
                print(f"✓ Score={result['score']:.2f}, RMS={result['rms_error']:.2f}°")
            else:
                print(f"✗ Failed")
        
        # 平均スコアを計算
        if len(valid_scores) == 0:
            print(f"  ❌ All angles failed!")
            return float('inf'), None
        
        # 統計量を計算
        avg_score = np.mean(valid_scores)
        std_score = np.std(valid_scores)
        max_score = np.max(valid_scores)
        min_score = np.min(valid_scores)
        
        # 全体の平均メトリクス
        avg_metrics = {}
        for metric in ['rms_error', 'max_abs_error', 'final_error', 
                       'settling_time', 'overshoot', 'smoothness']:
            values = [r[metric] for r in angle_results.values() if r['success']]
            if values:
                avg_metrics[f'avg_{metric}'] = np.mean(values)
                avg_metrics[f'std_{metric}'] = np.std(values)
                avg_metrics[f'max_{metric}'] = np.max(values)
        
        # 総合評価スコア（平均スコア + 分散ペナルティ）
        # 分散が大きい = 角度によって性能が不安定
        composite_score = avg_score + 0.3 * std_score
        
        print(f"  📊 Average score: {avg_score:.2f} ± {std_score:.2f}")
        print(f"  📊 Composite score: {composite_score:.2f} (avg + 0.3×std)")
        print(f"  📊 Score range: [{min_score:.2f}, {max_score:.2f}]")
        print(f"  📊 Success rate: {len(valid_scores)}/{len(self.test_angles)}")
        
        result = {
            'trial_id': trial_id,
            'params': params.copy(),
            'composite_score': composite_score,
            'avg_score': avg_score,
            'std_score': std_score,
            'min_score': min_score,
            'max_score': max_score,
            'success_rate': len(valid_scores) / len(self.test_angles),
            'angle_results': angle_results,
            **avg_metrics
        }
        
        self.results.append(result)
        
        # Update best
        if composite_score < self.best_score:
            self.best_score = composite_score
            self.best_params = params.copy()
            print(f"\n  🎯 New best composite score: {composite_score:.3f}")
            print(f"     Average score: {avg_score:.2f} ± {std_score:.2f}°")
            print(f"     Average RMS error: {avg_metrics.get('avg_rms_error', 0):.2f}°")
            print(f"     Params: K={params['K']}, H={params['horizon']}, "
                  f"λ={params['lambda']:.2f}, σ={params['sigma_u']:.3f}")
        
        return composite_score, result
    
    def calculate_settling_time(self, errors_deg, threshold=2.0):
        """整定時間を計算"""
        try:
            settling_idx = np.where(np.abs(errors_deg) <= threshold)[0]
            if len(settling_idx) > 0:
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
                overshoot = np.max(theta_log_deg - theta_target_deg)
            else:
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
            
            composite_score, _ = self.evaluate_params(params, trial.number)
            return composite_score
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(
            objective,
            n_trials=self.args.n_trials,
            show_progress_bar=False  # 独自の進捗表示を使用
        )
        
        print(f"\n{'='*70}")
        print(f"Optimization completed!")
        print(f"Best composite score: {study.best_value:.3f}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"{'='*70}\n")
        
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
            
            print(f"\n{'='*70}")
            print(f"Trial {trial_id + 1}/{self.args.n_trials}")
            print(f"{'='*70}")
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
            print(f"\n{'='*70}")
            print(f"Trial {trial_id + 1}/{total}")
            print(f"{'='*70}")
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
                    'composite_score': r['composite_score'],
                    'avg_score': r['avg_score'],
                    'std_score': r['std_score'],
                    'min_score': r['min_score'],
                    'max_score': r['max_score'],
                    'success_rate': r['success_rate'],
                }
                # Add average metrics
                for key in r.keys():
                    if key.startswith('avg_') or key.startswith('std_') or key.startswith('max_'):
                        record[key] = r[key]
                # Add parameters
                record.update(r['params'])
                
                # Add per-angle scores
                for angle, result in r['angle_results'].items():
                    if result['success']:
                        record[f'score_{angle}deg'] = result['score']
                        record[f'rms_{angle}deg'] = result['rms_error']
                
                records.append(record)
            
            df = pd.DataFrame(records)
            csv_path = results_dir / f"optimization_results_multiangle_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n📊 Results saved to: {csv_path}")
            
            # Save best parameters as JSON
            if self.best_params:
                best_path = results_dir / f"best_params_multiangle_{timestamp}.json"
                
                # Get best result details
                best_result = min(self.results, key=lambda x: x['composite_score'])
                
                with open(best_path, 'w') as f:
                    json.dump({
                        'best_composite_score': float(self.best_score),
                        'best_avg_score': float(best_result['avg_score']),
                        'best_std_score': float(best_result['std_score']),
                        'best_params': self.best_params,
                        'test_angles': self.test_angles,
                        'method': self.args.method,
                        'timestamp': timestamp,
                        'per_angle_performance': {
                            str(angle): {
                                'score': float(result['score']),
                                'rms_error': float(result['rms_error']),
                                'final_error': float(result['final_error'])
                            }
                            for angle, result in best_result['angle_results'].items()
                            if result['success']
                        }
                    }, f, indent=2)
                print(f"🏆 Best parameters saved to: {best_path}")
                
                # Print best parameters
                print(f"\n{'='*70}")
                print(f"BEST PARAMETERS (Composite Score: {self.best_score:.3f})")
                print(f"{'='*70}")
                print(f"Performance:")
                print(f"  Average score: {best_result['avg_score']:.2f}")
                print(f"  Std score: {best_result['std_score']:.2f}")
                print(f"  Score range: [{best_result['min_score']:.2f}, {best_result['max_score']:.2f}]")
                print(f"  Success rate: {best_result['success_rate']*100:.0f}%")
                print(f"\nParameters:")
                for key, value in self.best_params.items():
                    print(f"  --{key.replace('_', '-')} {value}")
                print(f"{'='*70}\n")
    
    def plot_results(self):
        """結果を可視化"""
        if not self.results:
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Score history
        ax = fig.add_subplot(gs[0, :2])
        composite_scores = [r['composite_score'] for r in self.results]
        avg_scores = [r['avg_score'] for r in self.results]
        ax.plot(composite_scores, 'o-', alpha=0.6, label='Composite Score')
        ax.plot(avg_scores, 's-', alpha=0.6, label='Avg Score')
        ax.axhline(y=self.best_score, color='r', linestyle='--', 
                   label=f'Best: {self.best_score:.2f}')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Score')
        ax.set_title('Optimization Progress')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Score distribution
        ax = fig.add_subplot(gs[0, 2])
        ax.hist(composite_scores, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(composite_scores), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(composite_scores):.2f}')
        ax.set_xlabel('Composite Score')
        ax.set_ylabel('Count')
        ax.set_title('Score Distribution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. Per-angle performance (best trial)
        ax = fig.add_subplot(gs[1, :])
        best_result = min(self.results, key=lambda x: x['composite_score'])
        angles = []
        scores = []
        rms_errors = []
        for angle, result in best_result['angle_results'].items():
            if result['success']:
                angles.append(angle)
                scores.append(result['score'])
                rms_errors.append(result['rms_error'])
        
        ax2 = ax.twinx()
        width = 2
        ax.bar([a - width/2 for a in angles], scores, width=width, 
               alpha=0.7, label='Score', color='steelblue')
        ax2.bar([a + width/2 for a in angles], rms_errors, width=width, 
                alpha=0.7, label='RMS Error', color='coral')
        
        ax.set_xlabel('Target Angle [deg]')
        ax.set_ylabel('Score', color='steelblue')
        ax2.set_ylabel('RMS Error [deg]', color='coral')
        ax.set_title(f'Best Parameters Performance Across Angles (Trial {best_result["trial_id"]})')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='coral')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 4. Parameter correlation (K vs horizon)
        ax = fig.add_subplot(gs[2, 0])
        K_values = [r['params']['K'] for r in self.results]
        horizon_values = [r['params']['horizon'] for r in self.results]
        scatter = ax.scatter(K_values, horizon_values, c=composite_scores, 
                           cmap='viridis_r', alpha=0.6, s=100)
        ax.set_xlabel('K (samples)')
        ax.set_ylabel('Horizon')
        ax.set_title('Parameter Space (K vs Horizon)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Composite Score')
        
        # 5. Avg vs Std score
        ax = fig.add_subplot(gs[2, 1])
        avg_scores = [r['avg_score'] for r in self.results]
        std_scores = [r['std_score'] for r in self.results]
        scatter = ax.scatter(avg_scores, std_scores, c=composite_scores,
                           cmap='viridis_r', alpha=0.6, s=100)
        ax.set_xlabel('Average Score')
        ax.set_ylabel('Std Score')
        ax.set_title('Score Stability (Avg vs Std)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Composite Score')
        
        # 6. Success rate vs score
        ax = fig.add_subplot(gs[2, 2])
        success_rates = [r['success_rate'] * 100 for r in self.results]
        ax.scatter(success_rates, composite_scores, alpha=0.6, s=100)
        ax.set_xlabel('Success Rate [%]')
        ax.set_ylabel('Composite Score')
        ax.set_title('Success Rate vs Performance')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Multi-Angle Optimization Results (Test angles: {self.test_angles})', 
                    fontsize=14, fontweight='bold')
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = Path("optimization_results") / f"optimization_plot_multiangle_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📈 Plot saved to: {plot_path}")
        
        if self.args.plot:
            plt.show()
        else:
            plt.close()
    
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
        description='MPPI Parameter Optimizer with Multi-Angle Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing NARX model')
    
    # Test angles
    parser.add_argument('--test-angles-deg', type=float, nargs='+',
                        help='Target angles for evaluation (default: -30 -20 -10 0 10 20 30)')
    
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
    
    optimizer = MultiAngleParameterOptimizer(args)
    optimizer.run()


if __name__ == '__main__':
    main()