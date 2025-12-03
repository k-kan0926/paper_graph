#!/bin/bash
# run_quick_comparison.sh
# クイックスタート用の実行スクリプト

set -e  # エラーで停止

echo "========================================"
echo " Quick Comparison Experiment"
echo "========================================"

# ========== Configuration ==========
# データファイル (環境に合わせて変更)
DYN_CSVS="processed_data/*.csv"

# 比較するモデル
MODELS="linear_arx narx lstm gru"

# 比較する制御手法
CONTROLLERS="mppi cem random_shooting pid"

# 評価する目標角度
TARGETS="15.0 30.0 -15.0 -30.0"

# 出力ディレクトリ
OUTPUT_ROOT="quick_comparison_results"

# 実験パラメータ
LAGS=24
HIDDEN=128
EPOCHS=200
BATCH_SIZE=512

# 制御パラメータ
K=32
HORIZON=15
STEPS=100

# ========== Step 1: モデル訓練 ==========
echo ""
echo "Step 1/3: Training models..."
echo "Models: $MODELS"

python3 train_models_unified.py \
    --dyn_csvs $DYN_CSVS \
    --models $MODELS \
    --out_dir ${OUTPUT_ROOT}/models \
    --lags $LAGS \
    --hidden $HIDDEN \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --patience 30

if [ $? -ne 0 ]; then
    echo "Error: Model training failed"
    exit 1
fi

# ========== Step 2: 制御評価 ==========
echo ""
echo "Step 2/3: Evaluating controllers..."
echo "Controllers: $CONTROLLERS"

for MODEL in $MODELS; do
    echo ""
    echo "--- Model: $MODEL ---"
    
    MODEL_DIR="${OUTPUT_ROOT}/models/${MODEL}"
    
    if [ ! -f "${MODEL_DIR}/model.pt" ]; then
        echo "Warning: Model not found, skipping: $MODEL_DIR"
        continue
    fi
    
    for TARGET in $TARGETS; do
        echo "  Target: ${TARGET}°"
        
        OUT_DIR="${OUTPUT_ROOT}/control_results/${MODEL}/target_${TARGET}deg"
        
        python3 control_methods_unified.py \
            --model_dir $MODEL_DIR \
            --controllers $CONTROLLERS \
            --theta_target_deg $TARGET \
            --steps $STEPS \
            --K $K \
            --horizon $HORIZON \
            --out_dir $OUT_DIR \
            --dt 0.01
    done
done

if [ $? -ne 0 ]; then
    echo "Error: Controller evaluation failed"
    exit 1
fi

# ========== Step 3: 結果分析 ==========
echo ""
echo "Step 3/3: Analyzing results..."

python3 analyze_comparison_results.py \
    --results_dir $OUTPUT_ROOT \
    --models $MODELS \
    --controllers $CONTROLLERS \
    --targets $TARGETS

if [ $? -ne 0 ]; then
    echo "Error: Analysis failed"
    exit 1
fi

# ========== Complete ==========
echo ""
echo "========================================"
echo " Experiment Complete!"
echo "========================================"
echo "Results saved in: ${OUTPUT_ROOT}/"
echo ""
echo "Key outputs:"
echo "  - ${OUTPUT_ROOT}/models/summary.json"
echo "  - ${OUTPUT_ROOT}/summary/model_comparison.csv"
echo "  - ${OUTPUT_ROOT}/summary/control_comparison.csv"
echo "  - ${OUTPUT_ROOT}/summary/*.png"
echo "  - ${OUTPUT_ROOT}/summary/summary_report.txt"
echo ""
echo "View the summary report:"
echo "  cat ${OUTPUT_ROOT}/summary/summary_report.txt"
echo ""