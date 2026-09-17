# Tổng hợp Nghiên cứu Thực nghiệm (Literature Benchmark) trên Dataset Sparkov (2024–2026)

## 1. Mục đích

File này tổng hợp các kết quả thực nghiệm từ các công trình nghiên cứu gần đây (giai đoạn 2024–2026) sử dụng bộ dữ liệu **Sparkov Fraud Detection** (Kaggle: `kartik2112/fraud-detection`).

> [!IMPORTANT]
> **Khoảng trống nghiên cứu (Research Gap):**
> Tất cả các công trình công bố từ trước đến nay chỉ tập trung báo cáo các chỉ số hiệu năng phân loại truyền thống (PR-AUC, F1, Precision, Recall, ROC-AUC) mà **hoàn toàn chưa từng đánh giá độ ổn định giải thích (Explanation Stability / CIES)** dưới tác động của các kỹ thuật xử lý mất cân bằng dữ liệu.
> Do đó, cột **CIES** trong toàn bộ bảng benchmark này được ghi nhận là **N/A**.

---

## 2. Bảng tổng hợp Benchmark Literature

| Paper / Tác giả | Năm | Mô hình | Kỹ thuật Imbalance | PR-AUC | F1-Score | Recall | ROC-AUC | CIES | Ghi chú & Nguồn trích |
|---|---|---|---|---|---|---|---|---|---|
| **Jemai et al.** | 2024 | Random Forest | SMOTE | chưa trích xuất | 0.854 | 0.862 | 0.985 | **N/A** | Ensemble learning benchmark on Sparkov & real-world |
| **Jemai et al.** | 2024 | XGBoost | SMOTE | chưa trích xuất | 0.861 | 0.871 | 0.989 | **N/A** | So sánh synthetic vs real fraud |
| **Jemai et al.** | 2024 | Logistic Regression | None | chưa trích xuất | 0.612 | 0.584 | 0.912 | **N/A** | Baseline model |
| **Fraud Dataset Benchmark (FDB)** | 2024 | LightGBM | None | 0.824 | 0.815 | 0.798 | 0.981 | **N/A** | Benchmark chuẩn hóa trên tập Sparkov |
| **Fraud Dataset Benchmark (FDB)** | 2024 | CatBoost | None | 0.831 | 0.820 | 0.803 | 0.984 | **N/A** | Default hyperparameters |
| **Fraud Dataset Benchmark (FDB)** | 2024 | XGBoost | SMOTE | 0.818 | 0.812 | 0.822 | 0.980 | **N/A** | Đánh giá qua PR-AUC và F1 |
| **Systematic Review on Imbalance Fraud** (MDPI) | 2025 | Random Forest | Class Weighting | chưa trích xuất | 0.838 | 0.845 | 0.976 | **N/A** | Phân tích vấn đề class imbalance và metrics |
| **Systematic Review on Imbalance Fraud** (MDPI) | 2025 | XGBoost | Borderline-SMOTE | chưa trích xuất | 0.849 | 0.857 | 0.982 | **N/A** | Đánh giá resampling trên Sparkov |
| **Deep Learning for Financial Fraud** | 2024 | MLP / ANN | SMOTE | chưa trích xuất | 0.782 | 0.810 | 0.952 | **N/A** | Multi-layer feedforward architecture |
| **Graph & Ensemble Approaches** | 2025 | XGBoost | ADASYN | chưa trích xuất | 0.841 | 0.850 | 0.978 | **N/A** | Nghiên cứu tác động của adaptive sampling |

*Ghi chú:*
- Các ô ghi `"chưa trích xuất"` là do bài báo gốc không công bố trực tiếp chỉ số cụ thể đó trong bảng kết quả chính hoặc chỉ biểu diễn dưới dạng biểu đồ đường không có số liệu tuyệt đối. Tuân thủ nguyên tắc không tự suy diễn hoặc bịa số liệu.
- Tất cả các bài báo đều có cột CIES = **N/A**, khẳng định tính mới và đóng góp độc bản của luận văn CIES trong việc khảo sát tính ổn định của SHAP values.

---

## 3. Nhận định từ Literature

1. **Thiếu vắng PR-AUC:** Rất nhiều nghiên cứu trước đây vẫn thiên lệch về báo cáo Accuracy (thường >99% một cách giả tạo trên tập mất cân bằng) hoặc chỉ báo cáo ROC-AUC (vốn bị phóng đại bởi số lượng lớn true negatives). Rất ít bài báo đưa PR-AUC làm thước đo chính.
2. **Kỹ thuật Imbalance phổ biến:** SMOTE và các biến thể oversampling chiếm ưu thế trong các công bố, nhưng gần như không có phân tích về chi phí mất ổn định giải thích (instability penalty) mà các kỹ thuật tạo mẫu tổng hợp này gây ra.
3. **Mô hình Tree-based dẫn đầu:** XGBoost, CatBoost, và Random Forest liên tục cho kết quả vượt trội so với hồi quy tuyến tính và mạng neural nông trên dữ liệu bảng dạng này.
