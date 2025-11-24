#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply Best Parameters Helper

Usage:
  # JSONファイルから最良パラメータを読み込んでシミュレーション実行
  python apply_best_params.py \
      --best-params optimization_results/best_params_20250124_143022.json \
      --model-dir models/narx_p1p2_production2 \
      --theta-target-deg 30 \
      --plot

  # 複数の角度でテスト
  python apply_best_params.py \
      --best-params optimization_results/best_params_20250124_143022.json \
      --model-dir models/narx_p1p2_production2 \
      --test-angles 10 20 30 -15 -25 \
      --plot
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


def load_best_params(json_path):
    """JSONファイルから最良パラメータを読み込む"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Display info about the parameters
    print(f"\n📊 Parameter Information:")
    if 'test_angles' in data:
        print(f"   Type: Multi-angle optimization")
        print(f"   Test angles: {data['test_angles']}")
        print(f"   Composite score: {data.get('best_composite_score', 'N/A'):.3f}")
        print(f"   Average score: {data.get('best_avg_score', 'N/A'):.3f}")
        print(f"   Std score: {data.get('best_std_score', 'N/A'):.3f}")
    else:
        print(f"   Type: Single-angle optimization")
        print(f"   Target angle: {data.get('target_deg', 'N/A')}°")
        print(f"   Best score: {data.get('best_score', 'N/A'):.3f}")
    
    return data['best_params']


def build_command(model_dir, theta_target_deg, best_params, additional_args):
    """シミュレーションコマンドを構築"""
    cmd = [
        'python', 'inverse7_2_narx_mppi_p1p2.py',
        '--model-dir', model_dir,
        '--theta-target-deg', str(theta_target_deg)
    ]
    
    # 最良パラメータを追加
    for key, value in best_params.items():
        arg_name = f"--{key.replace('_', '-')}"
        cmd.extend([arg_name, str(value)])
    
    # 追加引数
    if additional_args:
        cmd.extend(additional_args)
    
    return cmd


def run_simulation(cmd, verbose=True):
    """シミュレーションを実行"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*60}\n")
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Apply best parameters from optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--best-params', type=str, required=True,
                        help='Path to best_params.json file')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Model directory')
    
    # Single target or multiple test angles
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--theta-target-deg', type=float,
                       help='Single target angle')
    group.add_argument('--test-angles', type=float, nargs='+',
                       help='Multiple angles to test')
    
    # Additional options
    parser.add_argument('--plot', action='store_true',
                        help='Show plots')
    parser.add_argument('--save', action='store_true',
                        help='Save results to CSV')
    parser.add_argument('--steps', type=int,
                        help='Number of simulation steps')
    parser.add_argument('--context-csv', type=str,
                        help='Context CSV file')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.best_params):
        print(f"❌ Error: {args.best_params} not found")
        sys.exit(1)
    
    if not os.path.exists('inverse7_2_narx_mppi_p1p2.py'):
        print(f"❌ Error: inverse7_2_narx_mppi_p1p2.py not found in current directory")
        sys.exit(1)
    
    # Load best parameters
    print(f"\n📂 Loading best parameters from: {args.best_params}")
    best_params = load_best_params(args.best_params)
    
    print(f"\n✅ Best parameters loaded:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    # Build additional arguments
    additional_args = []
    if args.plot:
        additional_args.append('--plot')
    if args.steps:
        additional_args.extend(['--steps', str(args.steps)])
    if args.context_csv:
        additional_args.extend(['--context-csv', args.context_csv])
    
    # Run simulation(s)
    if args.theta_target_deg is not None:
        # Single target
        if args.save:
            save_path = f"results_best_params_{args.theta_target_deg:.0f}deg.csv"
            additional_args.extend(['--save', save_path])
        
        cmd = build_command(args.model_dir, args.theta_target_deg, 
                          best_params, additional_args)
        success = run_simulation(cmd)
        
        if success:
            print(f"\n✅ Simulation completed successfully!")
        else:
            print(f"\n❌ Simulation failed")
            sys.exit(1)
    
    else:
        # Multiple test angles
        print(f"\n🎯 Testing {len(args.test_angles)} different angles...")
        
        results = []
        for i, angle in enumerate(args.test_angles, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}/{len(args.test_angles)}: θ = {angle}°")
            print(f"{'='*60}")
            
            test_args = additional_args.copy()
            if args.save:
                save_path = f"results_best_params_{angle:.0f}deg.csv"
                test_args.extend(['--save', save_path])
            
            cmd = build_command(args.model_dir, angle, best_params, test_args)
            success = run_simulation(cmd, verbose=False)
            
            results.append({
                'angle': angle,
                'success': success
            })
        
        # Summary
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        for r in results:
            status = "✅ Success" if r['success'] else "❌ Failed"
            print(f"  θ = {r['angle']:6.1f}° : {status}")
        
        successes = sum(1 for r in results if r['success'])
        print(f"\nTotal: {successes}/{len(results)} succeeded")
        print(f"{'='*60}\n")
    
    print("\n🎉 All done!\n")


if __name__ == '__main__':
    main()