#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_comprehensive_comparison.py
モデル×制御手法の包括的比較実験

このスクリプトは:
1. 複数のモデルを訓練
2. 各モデルに対して複数の制御手法を評価
3. 結果を統合して分析
"""
import os
import json
import subprocess
import argparse
from pathlib import Path

def run_command(cmd):
    """コマンド実行"""
    print(f"\n[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with code {result.returncode}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--dyn_csvs', nargs='+', required=True,
                       help='Dynamic training CSV files')
    
    # Models to train
    parser.add_argument('--models', nargs='+',
                       default=['linear_arx', 'narx', 'lstm', 'gru', 'transformer', 'cnn'],
                       help='Models to train and compare')
    
    # Controllers to evaluate
    parser.add_argument('--controllers', nargs='+',
                       default=['mppi', 'cem', 'random_shooting', 'pid'],
                       help='Controllers to evaluate')
    
    # Simulation parameters
    parser.add_argument('--theta_targets', nargs='+', type=float,
                       default=[15.0, 30.0, 45.0, -15.0, -30.0],
                       help='Target angles for evaluation [deg]')
    parser.add_argument('--steps', type=int, default=100,
                       help='Simulation steps')
    
    # Training parameters
    parser.add_argument('--lags', type=int, default=24)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=512)
    
    # Control parameters
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--horizon', type=int, default=15)
    
    # Output
    parser.add_argument('--output_root', type=str, default='comprehensive_comparison')
    
    # Options
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip model training (use existing models)')
    parser.add_argument('--skip_control', action='store_true',
                       help='Skip control evaluation')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f" Comprehensive Model × Controller Comparison")
    print(f"{'='*70}")
    print(f"Models: {args.models}")
    print(f"Controllers: {args.controllers}")
    print(f"Targets: {args.theta_targets}°")
    print(f"Output: {args.output_root}")
    
    os.makedirs(args.output_root, exist_ok=True)
    
    # ========== Step 1: Train Models ==========
    models_dir = os.path.join(args.output_root, 'models')
    
    if not args.skip_training:
        print(f"\n{'='*70}")
        print(f" STEP 1: Training Models")
        print(f"{'='*70}")
        
        cmd = [
            'python3', 'train_models_unified.py',
            '--dyn_csvs'] + args.dyn_csvs + [
            '--models'] + args.models + [
            '--out_dir', models_dir,
            '--lags', str(args.lags),
            '--hidden', str(args.hidden),
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size)
        ]
        
        if not run_command(cmd):
            print("[ERROR] Model training failed")
            return
    else:
        print(f"\n[SKIP] Model training (using existing models in {models_dir})")
    
    # ========== Step 2: Evaluate Controllers ==========
    if not args.skip_control:
        print(f"\n{'='*70}")
        print(f" STEP 2: Evaluating Controllers")
        print(f"{'='*70}")
        
        for model_type in args.models:
            model_dir = os.path.join(models_dir, model_type)
            
            if not os.path.exists(os.path.join(model_dir, 'model.pt')):
                print(f"[SKIP] Model not found: {model_dir}")
                continue
            
            print(f"\n[Model] {model_type}")
            
            for target_deg in args.theta_targets:
                print(f"  [Target] {target_deg}°")
                
                out_dir = os.path.join(
                    args.output_root, 'control_results',
                    model_type, f'target_{target_deg:+.1f}deg'
                )
                
                cmd = [
                    'python3', 'control_methods_unified.py',
                    '--model_dir', model_dir,
                    '--controllers'] + args.controllers + [
                    '--theta_target_deg', str(target_deg),
                    '--steps', str(args.steps),
                    '--K', str(args.K),
                    '--horizon', str(args.horizon),
                    '--out_dir', out_dir
                ]
                
                if not run_command(cmd):
                    print(f"[ERROR] Control evaluation failed for {model_type}, {target_deg}°")
    else:
        print(f"\n[SKIP] Controller evaluation")
    
    # ========== Step 3: Generate Summary ==========
    print(f"\n{'='*70}")
    print(f" STEP 3: Generating Summary")
    print(f"{'='*70}")
    
    cmd = [
        'python3', 'analyze_comparison_results.py',
        '--results_dir', args.output_root,
        '--models'] + args.models + [
        '--controllers'] + args.controllers + [
        '--targets'] + [str(t) for t in args.theta_targets]
    
    run_command(cmd)
    
    print(f"\n{'='*70}")
    print(f" Comparison Complete!")
    print(f"{'='*70}")
    print(f"Results saved in: {args.output_root}/")
    print(f"  - models/: Trained models")
    print(f"  - control_results/: Controller evaluations")
    print(f"  - summary/: Analysis and plots")

if __name__ == '__main__':
    main()