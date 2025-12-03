#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_comparison_results.py
モデル×制御手法の比較結果を分析・可視化

出力:
1. 性能比較表 (CSV, LaTeX)
2. 可視化プロット
3. 統計的分析
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.size'] = 10
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False
    print("[WARNING] matplotlib not available, skipping plots")

try:
    import seaborn as sns
    sns.set_style('whitegrid')
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

def load_model_metrics(models_dir, models):
    """モデルの訓練メトリクスをロード"""
    model_metrics = {}
    
    for model in models:
        metrics_path = os.path.join(models_dir, model, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                model_metrics[model] = json.load(f)
    
    return model_metrics

def load_control_results(results_dir, models, controllers, targets):
    """制御結果をロード"""
    control_results = {}
    
    for model in models:
        control_results[model] = {}
        
        for target in targets:
            target_str = f'target_{target:+.1f}deg'
            control_results[model][target] = {}
            
            for controller in controllers:
                summary_path = os.path.join(
                    results_dir, 'control_results', model,
                    target_str, 'summary.json'
                )
                
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                        if controller in summary:
                            control_results[model][target][controller] = summary[controller]
    
    return control_results

def create_model_comparison_table(model_metrics, output_dir):
    """モデル比較表を作成"""
    data = []
    
    for model, metrics in model_metrics.items():
        row = {
            'Model': model,
            'Train RMSE': metrics['train']['rmse'],
            'Val RMSE': metrics['val']['rmse'],
            'Test RMSE': metrics['test']['rmse'],
            'Train MAE': metrics['train']['mae'],
            'Val MAE': metrics['val']['mae'],
            'Test MAE': metrics['test']['mae']
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values('Test RMSE')
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.5f')
    print(f"[Saved] {csv_path}")
    
    # Save LaTeX
    latex_path = os.path.join(output_dir, 'model_comparison.tex')
    df.to_latex(latex_path, index=False, float_format='%.5f',
                caption='Model Performance Comparison',
                label='tab:model_comparison')
    print(f"[Saved] {latex_path}")
    
    return df

def create_control_comparison_table(control_results, models, controllers, targets, output_dir):
    """制御手法比較表を作成"""
    data = []
    
    for model in models:
        if model not in control_results:
            continue
        
        for controller in controllers:
            rmse_list = []
            mae_list = []
            max_err_list = []
            
            for target in targets:
                if target in control_results[model]:
                    if controller in control_results[model][target]:
                        result = control_results[model][target][controller]
                        rmse_list.append(result.get('rmse', np.nan))
                        mae_list.append(result.get('mae', np.nan))
                        max_err_list.append(result.get('max_abs_error', np.nan))
            
            if rmse_list:
                row = {
                    'Model': model,
                    'Controller': controller,
                    'Avg RMSE': np.mean(rmse_list),
                    'Avg MAE': np.mean(mae_list),
                    'Avg Max Error': np.mean(max_err_list),
                    'Std RMSE': np.std(rmse_list),
                    'Std MAE': np.std(mae_list)
                }
                data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'control_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.3f')
    print(f"[Saved] {csv_path}")
    
    # Save LaTeX
    latex_path = os.path.join(output_dir, 'control_comparison.tex')
    df.to_latex(latex_path, index=False, float_format='%.3f',
                caption='Controller Performance Comparison',
                label='tab:control_comparison')
    print(f"[Saved] {latex_path}")
    
    return df

def create_heatmap(control_results, models, controllers, metric='rmse', output_dir='.'):
    """ヒートマップ作成"""
    if not _HAS_PLT:
        return
    
    # Aggregate over targets
    data = np.zeros((len(models), len(controllers)))
    
    for i, model in enumerate(models):
        if model not in control_results:
            continue
        
        for j, controller in enumerate(controllers):
            values = []
            for target_data in control_results[model].values():
                if controller in target_data:
                    values.append(target_data[controller].get(metric, np.nan))
            
            if values:
                data[i, j] = np.nanmean(values)
            else:
                data[i, j] = np.nan
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(controllers)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(controllers, rotation=45, ha='right')
    ax.set_yticklabels(models)
    
    # Annotate
    for i in range(len(models)):
        for j in range(len(controllers)):
            if not np.isnan(data[i, j]):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha='center', va='center', color='black', fontsize=9)
    
    ax.set_title(f'Average {metric.upper()} [deg]')
    fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmap_{metric}.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'heatmap_{metric}.pdf'))
    plt.close()
    
    print(f"[Saved] {output_dir}/heatmap_{metric}.png")

def create_bar_plot(control_results, models, controllers, metric='rmse', output_dir='.'):
    """棒グラフ作成"""
    if not _HAS_PLT:
        return
    
    data = []
    labels = []
    
    for model in models:
        if model not in control_results:
            continue
        
        for controller in controllers:
            values = []
            for target_data in control_results[model].values():
                if controller in target_data:
                    values.append(target_data[controller].get(metric, np.nan))
            
            if values:
                data.append(np.nanmean(values))
                labels.append(f'{model}\n{controller}')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(labels))
    bars = ax.bar(x, data, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Color by controller
    colors = {'mppi': 'steelblue', 'cem': 'orange',
              'random_shooting': 'green', 'pid': 'red'}
    
    for i, label in enumerate(labels):
        for ctrl, color in colors.items():
            if ctrl in label:
                bars[i].set_color(color)
                break
    
    ax.set_ylabel(f'{metric.upper()} [deg]')
    ax.set_title(f'Controller Performance Comparison ({metric.upper()})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'barplot_{metric}.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'barplot_{metric}.pdf'))
    plt.close()
    
    print(f"[Saved] {output_dir}/barplot_{metric}.png")

def create_scatter_plot(model_metrics, control_results, models, controllers, output_dir):
    """散布図: モデル精度 vs 制御性能"""
    if not _HAS_PLT:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'mppi': 'steelblue', 'cem': 'orange',
              'random_shooting': 'green', 'pid': 'red'}
    
    for controller in controllers:
        x_vals = []  # Model test RMSE
        y_vals = []  # Control RMSE
        
        for model in models:
            if model not in model_metrics or model not in control_results:
                continue
            
            model_rmse = model_metrics[model]['test']['rmse']
            
            ctrl_rmse_list = []
            for target_data in control_results[model].values():
                if controller in target_data:
                    ctrl_rmse_list.append(target_data[controller].get('rmse', np.nan))
            
            if ctrl_rmse_list:
                x_vals.append(model_rmse)
                y_vals.append(np.nanmean(ctrl_rmse_list))
        
        if x_vals:
            ax.scatter(x_vals, y_vals, label=controller, 
                      color=colors.get(controller, 'gray'),
                      s=100, alpha=0.7, edgecolors='black')
    
    ax.set_xlabel('Model Test RMSE [rad]')
    ax.set_ylabel('Average Control RMSE [deg]')
    ax.set_title('Model Accuracy vs Control Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_model_vs_control.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'scatter_model_vs_control.pdf'))
    plt.close()
    
    print(f"[Saved] {output_dir}/scatter_model_vs_control.png")

def create_trajectory_comparison(results_dir, models, controllers, target, output_dir):
    """軌道比較プロット"""
    if not _HAS_PLT:
        return
    
    target_str = f'target_{target:+.1f}deg'
    
    fig, axes = plt.subplots(len(models), 1, figsize=(12, 3*len(models)), sharex=True)
    if len(models) == 1:
        axes = [axes]
    
    for i, model in enumerate(models):
        ax = axes[i]
        
        for controller in controllers:
            csv_path = os.path.join(
                results_dir, 'control_results', model,
                target_str, controller, 'simulation.csv'
            )
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                ax.plot(df['t[s]'], np.degrees(df['theta[rad]']),
                       label=controller, linewidth=1.5, alpha=0.8)
        
        # Reference line
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            ax.plot(df['t[s]'], np.degrees(df['theta_ref[rad]']),
                   'k--', label='reference', linewidth=1, alpha=0.5)
        
        ax.set_ylabel(f'{model}\nAngle [deg]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'Trajectory Comparison (Target: {target}°)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'trajectories_target_{target:+.1f}deg.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'trajectories_target_{target:+.1f}deg.pdf'))
    plt.close()
    
    print(f"[Saved] {output_dir}/trajectories_target_{target:+.1f}deg.png")

def generate_summary_report(model_df, control_df, output_dir):
    """サマリーレポート生成"""
    report_path = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(" COMPREHENSIVE COMPARISON SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("--- Model Performance ---\n")
        f.write(model_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("Best model (by Test RMSE):\n")
        best_model = model_df.iloc[0]
        f.write(f"  {best_model['Model']}: {best_model['Test RMSE']:.5f} rad\n\n")
        
        f.write("--- Controller Performance ---\n")
        f.write(control_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("Best combinations:\n")
        top5 = control_df.nsmallest(5, 'Avg RMSE')
        for idx, row in top5.iterrows():
            f.write(f"  {row['Model']} + {row['Controller']}: "
                   f"RMSE={row['Avg RMSE']:.3f}°, MAE={row['Avg MAE']:.3f}°\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"[Saved] {report_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--controllers', nargs='+', required=True)
    parser.add_argument('--targets', nargs='+', type=float, required=True)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f" Analyzing Comparison Results")
    print(f"{'='*70}")
    
    # Create output directory
    summary_dir = os.path.join(args.results_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Load data
    print("\n[Loading] Model metrics...")
    models_dir = os.path.join(args.results_dir, 'models')
    model_metrics = load_model_metrics(models_dir, args.models)
    
    print("[Loading] Control results...")
    control_results = load_control_results(
        args.results_dir, args.models, args.controllers, args.targets
    )
    
    # Create tables
    print("\n[Generating] Comparison tables...")
    model_df = create_model_comparison_table(model_metrics, summary_dir)
    control_df = create_control_comparison_table(
        control_results, args.models, args.controllers, args.targets, summary_dir
    )
    
    # Create plots
    if _HAS_PLT:
        print("\n[Generating] Plots...")
        
        create_heatmap(control_results, args.models, args.controllers,
                      metric='rmse', output_dir=summary_dir)
        
        create_heatmap(control_results, args.models, args.controllers,
                      metric='mae', output_dir=summary_dir)
        
        create_bar_plot(control_results, args.models, args.controllers,
                       metric='rmse', output_dir=summary_dir)
        
        create_scatter_plot(model_metrics, control_results,
                          args.models, args.controllers, summary_dir)
        
        # Trajectory plots for each target
        for target in args.targets:
            create_trajectory_comparison(args.results_dir, args.models,
                                        args.controllers, target, summary_dir)
    
    # Generate report
    print("\n[Generating] Summary report...")
    generate_summary_report(model_df, control_df, summary_dir)
    
    print(f"\n{'='*70}")
    print(f" Analysis Complete!")
    print(f"{'='*70}")
    print(f"Results saved in: {summary_dir}/")
    print(f"  - Comparison tables (CSV, LaTeX)")
    print(f"  - Visualization plots (PNG, PDF)")
    print(f"  - Summary report (TXT)")

if __name__ == '__main__':
    main()