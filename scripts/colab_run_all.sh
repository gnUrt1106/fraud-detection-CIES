#!/usr/bin/env bash
# ==============================================================================
# scripts/colab_run_all.sh — Dứt điểm toàn bộ 25 tổ hợp còn lại trên Colab
# Tự động điều phối: LR -> XGBoost -> ANN -> CatBoost (Approximate SHAP) + RF ngầm
# ==============================================================================
set -u

REPO_DIR="/content/fraud-detection-CIES"
DRIVE_DIR="/content/drive/MyDrive/CIES_colab"
LOGS_DIR="${DRIVE_DIR}/logs"
mkdir -p "${LOGS_DIR}"

cd "${REPO_DIR}" || exit 1

echo "=========================================================="
echo "🚀 BẮT ĐẦU CHẠY TOÀN BỘ CÁC TỔ HỢP CÒN LẠI TRONG ĐÊM NAY"
echo "Thời gian bắt đầu: $(date)"
echo "=========================================================="

# 1. Đợi tiến trình GPU cũ (PID 37104 nếu còn) kết thúc sạch sẽ
if ps aux | grep -v grep | grep "37104" > /dev/null 2>&1; then
    echo "⏳ Đang đợi tiến trình ANN cũ (PID 37104) kết thúc..."
    while ps aux | grep -v grep | grep "37104" > /dev/null 2>&1; do
        sleep 20
    done
    echo "✅ Tiến trình ANN cũ đã kết thúc!"
fi

# 2. Chạy LR với smote_enn (chỉ mất ~2 phút trên CPU)
echo "⚡ [1/3] Đang chạy Logistic Regression + SMOTE-ENN (CPU, ~2 phút)..."
python scripts/run_rerun.py train \
    --models logistic_regression \
    --techniques smote_enn \
    --shap-rows all \
    --shap-batch 100000 \
    --force-cpu \
    --seeds 0-9 \
    --name colab_full_cpu

echo "✅ Đã xong trọn vẹn 5/5 kỹ thuật cho Logistic Regression!"

# 3. Kích hoạt RF Watcher ngầm: đợi PID 49287 xong thì tự chạy tiếp RF + SMOTE-ENN
echo "🌲 [2/3] Bật watcher ngầm cho Random Forest (chờ PID 49287 xong sẽ tự chạy nốt smote_enn)..."
nohup bash -c '
    while ps aux | grep -v grep | grep "49287" > /dev/null 2>&1; do
        sleep 60
    done
    echo "[$(date)] PID 49287 đã xong! Bắt đầu chạy random_forest smote_enn..."
    python scripts/run_rerun.py train \
        --models random_forest \
        --techniques smote_enn \
        --shap-rows all \
        --shap-batch 100000 \
        --shap-backend native \
        --seeds 0-9 \
        --name colab_full_rf
    echo "[$(date)] Hoàn tất trọn vẹn 5/5 kỹ thuật cho Random Forest!"
' > "${LOGS_DIR}/rf_watcher.log" 2>&1 &

# 4. Chạy chuỗi GPU chính: XGBoost (smote_enn) -> ANN (smote_enn) -> CatBoost (5 kỹ thuật)
echo "🔥 [3/3] Bắt đầu chuỗi GPU: XGBoost (smote_enn) -> ANN (smote_enn) -> CatBoost (5 kỹ thuật, GPU train + Approx SHAP)..."
python scripts/run_rerun.py train \
    --models xgboost ann catboost \
    --techniques class_weighting smote adasyn borderline_smote smote_enn \
    --shap-rows all \
    --shap-batch 100000 \
    --shap-backend native \
    --catboost-shap-calc-type Approximate \
    --seeds 0-9 \
    --name colab_full_gpu

echo "=========================================================="
echo "🎉 HOÀN THÀNH TẤT CẢ 25 TỔ HỢP!"
echo "Thời gian kết thúc: $(date)"
echo "=========================================================="
