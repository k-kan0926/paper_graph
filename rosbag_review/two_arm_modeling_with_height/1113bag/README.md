# MPPI Parameter Tuning Guide

## パラメータの詳細解説

### 1. MPPI制御パラメータ

#### K (サンプル数)
```
範囲: 16～128
デフォルト: 32
```
**効果:**
- サンプル軌道の数を制御
- 多いほど：より広い制御空間を探索、最適解に近づく
- 少ないほど：計算が速い、局所解に陥りやすい

**選び方:**
- リアルタイム制御が必要: K=16～32
- 高精度が必要: K=64～128
- バランス型: K=32～48

**他のパラメータとの関係:**
- horizonが長い場合はKも増やすべき
- sigma_uが大きい場合はKを増やして探索を安定化

---

#### horizon (予測ホライズン)
```
範囲: 8～30ステップ
デフォルト: 15
```
**効果:**
- 何ステップ先まで予測するか
- 長いほど：先を見越した滑らかな制御
- 短いほど：反応的、計算が速い

**選び方:**
- システム応答が速い: horizon=8～12
- システム応答が遅い: horizon=20～30
- 標準的: horizon=15

**注意点:**
- 長すぎるとモデル誤差が累積
- 制御周期 × horizon が目標到達時間の1/3～1/2程度が目安

---

#### lambda (温度パラメータ)
```
範囲: 0.5～10.0
デフォルト: 2.0
```
**効果:**
- MPPI重み計算の温度パラメータ
- 小さいほど：最良サンプルに集中（exploitation）
- 大きいほど：多様なサンプルを考慮（exploration）

**選び方:**
- 環境が決定的: lambda=0.5～1.0
- ノイズが多い環境: lambda=3.0～10.0
- 標準的: lambda=1.0～2.0

**ヒント:**
```
exp(-(cost - min_cost) / lambda)
```
- lambdaが小さい → 最小コストのサンプルが支配的
- lambdaが大きい → より民主的な平均

---

#### sigma_u (制御ノイズ)
```
範囲: 0.03～0.3 MPa
デフォルト: 0.10 MPa
```
**効果:**
- 制御入力のランダムノイズの標準偏差
- 大きいほど：大胆な制御を試す、探索的
- 小さいほど：安定、保守的

**選び方:**
```
sigma_u ≈ 0.05～0.10 × 制御範囲
```
- 制御範囲が0.7MPaなら: 0.05～0.10 MPa

**他のパラメータとの関係:**
- Kが小さい → sigma_uを小さく（安定性重視）
- Kが大きい → sigma_uを大きく（探索強化）

---

### 2. コスト重み

#### w_tracking (追従コスト重み)
```
範囲: 10.0～100.0
デフォルト: 30.0
```
**効果:**
- 目標値への追従誤差のコスト
- 大きいほど：素早く目標に到達しようとする
- 小さいほど：他の要素（滑らかさなど）を優先

**選び方:**
- 高速応答が必要: 50～100
- 滑らかな動作が重要: 10～30
- バランス型: 30～50

**副作用:**
- 大きすぎると振動的になる
- ターミナルコストも併用されるので過度に大きくしない

---

#### w_smooth (滑らかさコスト重み)
```
範囲: 0.01～1.0
デフォルト: 0.05
```
**効果:**
- 制御入力の変化率(dp/dt)を抑制
- 大きいほど：滑らかな制御
- 小さいほど：急激な制御変化を許容

**選び方:**
- 機械的負荷軽減が重要: 0.1～1.0
- 高速応答が必要: 0.01～0.05
- 標準的: 0.05

**ヒント:**
```
cost += w_smooth × (dp1² + dp2²)
```
- アクチュエータの寿命延長に寄与
- 振動抑制効果

---

#### w_effort (制御努力コスト重み)
```
範囲: 0.001～0.1
デフォルト: 0.01
```
**効果:**
- 制御入力の大きさを抑制
- 大きいほど：エネルギー効率的
- 小さいほど：強い制御を許容

**選び方:**
- エネルギー効率重視: 0.05～0.1
- 性能重視: 0.001～0.01
- 標準的: 0.01

**注意:**
- 大きすぎると制御が弱くなり目標に到達しない

---

#### w_constraint (制約違反コスト重み)
```
範囲: 100.0～1000.0
デフォルト: 500.0
```
**効果:**
- 圧力範囲、角度範囲の制約違反ペナルティ
- 大きいほど：制約を厳守
- 小さいほど：制約を柔軟に扱う

**選び方:**
- ハード制約として扱いたい: 500～1000
- ソフト制約として扱いたい: 100～300

**ヒント:**
- enforce_constraints()で物理的にクリップされるが
- コスト関数でも評価することで事前に回避

---

## 最適化手法の選択

### 1. Optuna (Bayesian Optimization) 【推奨】
```bash
python mppi_parameter_optimizer.py \
    --model-dir models/narx_p1p2_production2 \
    --method optuna \
    --n-trials 50 \
    --theta-target-deg 30
```

**特徴:**
- ベイズ最適化により効率的に探索
- 過去の試行結果を活用
- 50試行で良い結果が得られることが多い

**推奨ケース:**
- 時間が限られている
- 最良の結果を得たい
- **最も推奨**

**必要なインストール:**
```bash
pip install optuna
```

---

### 2. Random Search
```bash
python mppi_parameter_optimizer.py \
    --model-dir models/narx_p1p2_production2 \
    --method random \
    --n-trials 30 \
    --theta-target-deg 30
```

**特徴:**
- ランダムにパラメータをサンプル
- シンプルで実装が容易
- 意外と効果的

**推奨ケース:**
- 初期探索として使いたい
- Optunaがインストールできない
- パラメータ空間の概要を把握したい

---

### 3. Grid Search
```bash
python mppi_parameter_optimizer.py \
    --model-dir models/narx_p1p2_production2 \
    --method grid \
    --theta-target-deg 30
```

**特徴:**
- 格子状にパラメータを探索
- 徹底的だが時間がかかる
- 再現性が高い

**推奨ケース:**
- 時間に余裕がある
- パラメータの影響を体系的に調べたい
- 論文などで使用

**注意:**
- デフォルトで96通りの組み合わせ
- 1試行30秒として約48分かかる

---

## 典型的なワークフロー

### Phase 1: 初期探索
```bash
# 1. Random searchで大まかな範囲を把握（速い）
python mppi_parameter_optimizer.py \
    --model-dir models/narx_p1p2_production2 \
    --method random \
    --n-trials 20 \
    --theta-target-deg 30
```

### Phase 2: 最適化
```bash
# 2. Optunaで詳細最適化（効率的）
python mppi_parameter_optimizer.py \
    --model-dir models/narx_p1p2_production2 \
    --method optuna \
    --n-trials 50 \
    --theta-target-deg 30 \
    --plot
```

### Phase 3: 検証
```bash
# 3. 最良パラメータで実際にシミュレーション
python inverse7_2_narx_mppi_p1p2.py \
    --model-dir models/narx_p1p2_production2 \
    --theta-target-deg 30 \
    --K 48 \
    --horizon 18 \
    --lambda 1.5 \
    --sigma-u 0.08 \
    --w-tracking 45.0 \
    --w-smooth 0.08 \
    --w-effort 0.02 \
    --w-constraint 600.0 \
    --plot
```

---

## 評価メトリクス

最適化スクリプトは以下のメトリクスを計算します：

### 1. RMS Error (主要指標)
```
RMS = sqrt(mean(error²))
```
- 全体的な追従性能
- **最も重要な指標**

### 2. Max Absolute Error
```
max(|error|)
```
- 最大逸脱量
- 安全性の評価

### 3. Final Error
```
|error[-1]|
```
- 定常偏差
- 最終的な精度

### 4. Settling Time
```
最後に誤差が2°を超えた時点
```
- 応答速度の指標
- 短いほど速い

### 5. Overshoot
```
目標を超えた最大量
```
- 振動的な動作の指標
- 小さいほど良い

### 6. Smoothness
```
mean(|Δp1| + |Δp2|)
```
- 制御の滑らかさ
- 機械的負荷の指標

### 総合スコア
```
score = 1.0×RMS + 0.3×MaxErr + 0.5×FinalErr 
        + 0.2×SettlingTime + 0.1×Overshoot + 0.05×Smoothness
```

---

## トラブルシューティング

### 問題1: 振動が発生する
**症状:** 目標値周りで振動する

**解決策:**
1. `w_tracking` を小さく (30 → 20)
2. `w_smooth` を大きく (0.05 → 0.1)
3. `sigma_u` を小さく (0.10 → 0.05)
4. `lambda` を大きく (2.0 → 5.0)

### 問題2: 応答が遅い
**症状:** 目標到達に時間がかかる

**解決策:**
1. `w_tracking` を大きく (30 → 50)
2. `horizon` を長く (15 → 20)
3. `K` を増やす (32 → 64)
4. `w_effort` を小さく (0.01 → 0.005)

### 問題3: 制約違反が多い
**症状:** 圧力制約を頻繁に超える

**解決策:**
1. `w_constraint` を大きく (500 → 1000)
2. `sigma_u` を小さく (0.10 → 0.05)
3. `p_max` や `dp_max` を見直す

### 問題4: 計算が遅い
**症状:** 最適化に時間がかかりすぎる

**解決策:**
1. `K` を減らす (64 → 32)
2. `horizon` を短く (20 → 12)
3. `--steps` を減らす (100 → 50)
4. Random searchを使う

---

## 実践的なパラメータセット

### セット1: 高速応答型
```bash
--K 32 --horizon 12 --lambda 1.0 --sigma-u 0.12 \
--w-tracking 60.0 --w-smooth 0.03 --w-effort 0.005 --w-constraint 500.0
```
- 用途: 素早い応答が必要な場合
- 特徴: 多少振動的だが高速

### セット2: 安定重視型
```bash
--K 64 --horizon 20 --lambda 3.0 --sigma-u 0.05 \
--w-tracking 25.0 --w-smooth 0.15 --w-effort 0.02 --w-constraint 700.0
```
- 用途: 滑らかで安定した動作が必要
- 特徴: 遅いが振動が少ない

### セット3: バランス型（推奨開始点）
```bash
--K 48 --horizon 15 --lambda 2.0 --sigma-u 0.08 \
--w-tracking 35.0 --w-smooth 0.08 --w-effort 0.01 --w-constraint 500.0
```
- 用途: 標準的な用途
- 特徴: 速度と安定性のバランス

### セット4: 高精度型
```bash
--K 96 --horizon 25 --lambda 1.5 --sigma-u 0.06 \
--w-tracking 40.0 --w-smooth 0.10 --w-effort 0.015 --w-constraint 800.0
```
- 用途: 高精度な追従が必要
- 特徴: 計算コスト高いが高精度

---

## よくある質問

### Q1: どのくらいの試行回数が必要？
**A:** Optunaの場合:
- 初期探索: 20～30試行
- 詳細最適化: 50～100試行
- 十分な最適化: 100～200試行

### Q2: 異なる目標角度で再チューニングが必要？
**A:** 基本的には不要ですが、大きく異なる場合（±10°以上）は再チューニング推奨

### Q3: 実機とシミュレーションで同じパラメータで良い？
**A:** シミュレーションで最適化したパラメータを実機の初期値として使用し、実機で微調整することを推奨

### Q4: 最適化に何時間かかる？
**A:** 
- Optuna 50試行: 約25分（1試行30秒として）
- Random 30試行: 約15分
- Grid search: 約1～2時間

### Q5: 複数の目標角度でテストすべき？
**A:** はい。以下の角度でテスト推奨:
```bash
--theta-target-deg 20
--theta-target-deg 30
--theta-target-deg -25
```

---

## 出力ファイル

最適化後、以下のファイルが生成されます：

```
optimization_results/
├── optimization_results_YYYYMMDD_HHMMSS.csv  # 全試行の結果
├── best_params_YYYYMMDD_HHMMSS.json          # 最良パラメータ
└── optimization_plot_YYYYMMDD_HHMMSS.png     # 可視化
```

### best_params.json の使用例
```bash
# JSONファイルから最良パラメータを読み取って実行
python inverse7_2_narx_mppi_p1p2.py \
    --model-dir models/narx_p1p2_production2 \
    --theta-target-deg 30 \
    $(python -c "import json; params=json.load(open('optimization_results/best_params_20250124_143022.json'))['best_params']; print(' '.join([f'--{k.replace(\"_\", \"-\")} {v}' for k,v in params.items()]))")
```

---

## まとめ

1. **まず Optuna で最適化** (推奨)
2. **バランス型から開始** してチューニング
3. **複数の目標角度** でテスト
4. **実機で微調整**

Good luck with your parameter tuning! 🎯s