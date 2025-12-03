# Model and Controller Comparison Framework

人工筋肉拮抗駆動制御のための包括的なモデル・制御手法比較

## 概要

このフレームワークは以下の機能を提供します:

1. **複数のモデルアーキテクチャの訓練・評価**
   - Linear ARX (線形ベースライン)
   - NARX (非線形MLP)
   - LSTM (再帰型NN)
   - GRU (軽量再帰型NN)
   - Transformer (アテンション機構)
   - 1D-CNN + Dense (畳み込み)

2. **複数の制御手法の評価**
   - MPPI (Model Predictive Path Integral)
   - CEM (Cross-Entropy Method)
   - Random Shooting
   - PID (ベースライン)

3. **包括的な比較分析**
   - 性能比較表 (CSV, LaTeX)
   - 可視化 (ヒートマップ、棒グラフ、散布図、軌道プロット)
   - 統計分析

## ファイル構成

```
.
├── train_models_unified.py          # モデル訓練スクリプト
├── control_methods_unified.py       # 制御手法評価スクリプト
├── run_comprehensive_comparison.py  # 包括的比較実験マスタースクリプト
├── analyze_comparison_results.py    # 結果分析・可視化スクリプト
├── run_comparison.sh          # クイックスタート用シェルスクリプト
└── README.md                        # このファイル
```

## 使い方

### 準備

1. データの前処理 (

```bash
python3 preprocess_csvs_for_narx.py \
    --input_dir raw_data/ \
    --output_dir processed_data/
```

### 方法1:

```bash
python3 run_comprehensive_comparison.py \
    --dyn_csvs processed_data/*.csv \
    --models narx lstm gru transformer \
    --controllers mppi cem random_shooting pid \
    --theta_targets 15.0 30.0 45.0 -15.0 -30.0 \
    --steps 100 \
    --output_root comparison_results
```

### 方法2: ステップごとに実行

#### Step 1: モデル訓練

```bash
python3 train_models_unified.py \
    --dyn_csvs processed_data/session1.csv processed_data/session2.csv \
    --models linear_arx narx lstm gru transformer cnn \
    --out_dir models_comparison \
    --lags 24 \
    --hidden 128 \
    --epochs 300 \
    --batch_size 512
```

**出力:**
- `models_comparison/linear_arx/`: Linear ARXモデル
- `models_comparison/narx/`: NARXモデル
- ...
- `models_comparison/summary.json`: 全モデルの性能サマリー

#### Step 2: 制御手法評価

各モデルに対して制御手法を評価:

```bash
# NARXモデルで評価
python3 control_methods_unified.py \
    --model_dir models_comparison/narx \
    --controllers mppi cem random_shooting pid \
    --theta_target_deg 30.0 \
    --steps 100 \
    --K 32 \
    --horizon 15 \
    --out_dir control_results/narx/target_30deg

# LSTMモデルで評価
python3 control_methods_unified.py \
    --model_dir models_comparison/lstm \
    --controllers mppi cem random_shooting pid \
    --theta_target_deg 30.0 \
    --steps 100 \
    --out_dir control_results/lstm/target_30deg
```

**出力:**
- `control_results/MODEL/target_XXdeg/CONTROLLER/simulation.csv`
- `control_results/MODEL/target_XXdeg/summary.json`

#### Step 3: 結果分析

```bash
python3 analyze_comparison_results.py \
    --results_dir comparison_results \
    --models narx lstm gru transformer \
    --controllers mppi cem random_shooting pid \
    --targets 15.0 30.0 45.0 -15.0 -30.0
```

**出力:**
- `summary/model_comparison.csv`: モデル性能比較表
- `summary/control_comparison.csv`: 制御性能比較表
- `summary/model_comparison.tex`: LaTeX表
- `summary/control_comparison.tex`: LaTeX表
- `summary/heatmap_rmse.png`: ヒートマップ
- `summary/barplot_rmse.png`: 棒グラフ
- `summary/scatter_model_vs_control.png`: 散布図
- `summary/trajectories_target_XXdeg.png`: 軌道プロット
- `summary/summary_report.txt`: テキストレポート

## 論文用の推奨実験設定

### 実験1: モデルアーキテクチャ比較

```bash
python3 train_models_unified.py \
    --dyn_csvs data/train*.csv \
    --models linear_arx narx lstm gru transformer cnn \
    --lags 24 \
    --hidden 128 \
    --epochs 300 \
    --out_dir paper/exp1_models
```

**評価項目:**
- 訓練・検証・テストRMSE/MAE
- パラメータ数
- 訓練時間
- ロールアウト精度

### 実験2: 制御手法比較 (最良モデル使用)

```bash
# 最良モデルを選択 (例: NARX)
python3 control_methods_unified.py \
    --model_dir paper/exp1_models/narx \
    --controllers mppi cem random_shooting pid \
    --theta_target_deg 30.0 \
    --steps 200 \
    --K 32 \
    --horizon 15 \
    --out_dir paper/exp2_controllers
```

**評価項目:**
- 追従誤差 (RMSE, MAE, Max Error)
- 整定時間
- 制御コスト
- 計算時間

### 実験3: モデル×制御手法の組み合わせ評価

```bash
python3 run_comprehensive_comparison.py \
    --dyn_csvs data/*.csv \
    --models narx lstm gru \
    --controllers mppi cem random_shooting \
    --theta_targets 10 20 30 40 -10 -20 -30 -40 \
    --steps 150 \
    --output_root paper/exp3_comprehensive
```

**評価項目:**
- 各組み合わせの性能マトリクス
- ロバスト性評価 (複数目標値での性能)
- モデル精度と制御性能の相関

## パラメータ調整ガイド

### モデル訓練

- `--lags`: 過去の時刻数 (10-30推奨)
- `--hidden`: 隠れ層サイズ (64-256)
- `--dropout`: ドロップアウト率 (0.0-0.1)
- `--epochs`: エポック数 (200-500)
- `--batch_size`: バッチサイズ (256-1024)
- `--lr`: 学習率 (1e-4 ~ 1e-3)

### 制御パラメータ

**MPPI:**
- `--K`: サンプル数 (16-64, GPU使用時は大きめ)
- `--horizon`: 予測ホライゾン (10-20)
- `--lam`: 温度パラメータ (1.0-5.0, 小さいほど探索的)
- `--sigma_u`: 制御ノイズ (0.05-0.15 MPa)

**CEM:**
- `--K`: サンプル数 (32-128)
- `--elite_frac`: エリート率 (0.1-0.3)
- `--n_iter`: 反復回数 (2-5)

**コスト重み:**
- `--w_tracking`: 追従誤差重み (10-50)
- `--w_smooth`: 滑らかさ重み (0.01-0.1)
- `--w_effort`: 制御努力重み (0.001-0.05)

## 出力ファイルの説明

### モデル訓練

- `model.pt`: 訓練済みモデルの重み
- `meta.json`: モデルのメタ情報 (lags, features等)
- `metrics.json`: 訓練・検証・テストメトリクス

### 制御評価

- `simulation.csv`: シミュレーション軌道データ
  - `t[s]`: 時刻
  - `theta[rad]`: 角度
  - `theta_ref[rad]`: 目標角度
  - `error[rad]`: 追従誤差
  - `p1[MPa]`, `p2[MPa]`: 圧力指令
  - `cost`: コスト

- `summary.json`: 性能メトリクス
  - `rmse`: 二乗平均平方根誤差 [deg]
  - `mae`: 平均絶対誤差 [deg]
  - `max_abs_error`: 最大絶対誤差 [deg]
  - `final_error`: 最終誤差 [deg]
  - `mean_cost`: 平均コスト

### 分析結果

- **CSV/LaTeX表**: 論文にそのまま使える形式
- **PNG/PDF図**: 高解像度プロット (論文用)
- **TXT報告書**: 実験結果のサマリー

## 論文での使用例

### セクション構成案

1. **Introduction**
   - 人工筋肉制御の課題
   - モデルベース制御の重要性

2. **Methods**
   - 2.1 System Model
   - 2.2 Model Architectures (表: `model_comparison.tex`)
   - 2.3 Control Methods
   - 2.4 Evaluation Metrics

3. **Results**
   - 3.1 Model Performance (図: `heatmap_rmse.png`)
   - 3.2 Controller Performance (表: `control_comparison.tex`)
   - 3.3 Trajectory Comparison (図: `trajectories_*.png`)
   - 3.4 Correlation Analysis (図: `scatter_model_vs_control.png`)

4. **Discussion**
   - モデル選択の指針
   - 制御手法の適用可能性
   - 計算コストとの兼ね合い

## トラブルシューティング

### GPUが使われない
- PyTorchのCUDAサポートを確認: `python3 -c "import torch; print(torch.cuda.is_available())"`
- `--cpu` フラグを外す

### メモリ不足
- `--batch_size` を小さく (256, 128)
- `--K` を小さく (16, 8)
- `--hidden` を小さく (64)

### 訓練が収束しない
- `--lr` を小さく (1e-4)
- `--epochs` を増やす
- データの正規化を確認

### 制御が発散する
- `--horizon` を短く (10)
- `--sigma_u` を小さく (0.05)
- `--w_constraint` を大きく (1000.0)

## 依存パッケージ

```bash
pip install torch numpy pandas matplotlib seaborn
```
