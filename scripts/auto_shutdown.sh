#!/usr/bin/env bash
# ==============================================================================
# scripts/auto_shutdown.sh — Tự động tắt Colab runtime ngay khi xong để không hao CCU
# ==============================================================================
set -u

REPO_DIR="/content/fraud-detection-CIES"
DRIVE_DIR="/content/drive/MyDrive/CIES_colab"
LOGS_DIR="${DRIVE_DIR}/logs"
LOG_FILE="${LOGS_DIR}/auto_shutdown.log"
mkdir -p "${LOGS_DIR}"

echo "[$(date)] 🛡️ Auto-Shutdown Watcher đã được kích hoạt." > "${LOG_FILE}"
echo "[$(date)] Đang theo dõi các tiến trình CIES (GPU và RF)..." >> "${LOG_FILE}"

# Chờ 30s ban đầu để các tiến trình ổn định
sleep 30

while true; do
    # Kiểm tra xem còn tiến trình run_rerun.py hay watcher nào không
    if ps aux | grep -v grep | grep -E "run_rerun.py|rf_watcher" > /dev/null 2>&1; then
        sleep 60
    else
        echo "[$(date)] ✅ TẤT CẢ TIẾN TRÌNH CIES ĐÃ KẾT THÚC HOÀN TOÀN!" >> "${LOG_FILE}"
        echo "[$(date)] 📊 Đang tự động gộp kết quả (merge) 25 tổ hợp..." >> "${LOG_FILE}"
        
        cd "${REPO_DIR}" || cd /content
        python scripts/run_rerun.py merge >> "${LOG_FILE}" 2>&1
        
        echo "[$(date)] 🔌 Đang giải phóng Colab VM (runtime.unassign()) để ngưng trừ CCU..." >> "${LOG_FILE}"
        python3 -c "from google.colab import runtime; runtime.unassign()" >> "${LOG_FILE}" 2>&1
        
        echo "[$(date)] Hoàn tất yêu cầu tắt máy." >> "${LOG_FILE}"
        break
    fi
done
