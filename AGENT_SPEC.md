# AGENT_SPEC.md — Đặc tả kỹ thuật cho pipeline nghiên cứu CIES

> File này dành cho coding agent (Antigravity/Claude Code) đọc và thực thi khi xây dựng project.
> Đây KHÔNG phải tài liệu học thuật để đọc hiểu — đây là spec kỹ thuật với quyết định đã chốt, ràng buộc bắt buộc, và lý do (để agent không tự "sửa cho tối ưu hơn" và vô tình phá vỡ tính hợp lệ của thí nghiệm.

---

## 0. MỤC TIÊU HỆ THỐNG

Xây dựng pipeline đo **CIES (Credibility Index via Explanation Stability)** — độ ổn định của SHAP values — qua tổ hợp (5 model × 5 kỹ thuật xử lý mất cân bằng), trên 2 dataset (1 chính, 1 phụ để kiểm chứng khái quát hóa).

**Nguyên tắc thiết kế tối thượng — áp dụng cho MỌI quyết định implement:**
> Mọi bước trong pipeline phải loại bỏ được các nguồn bất định (confound) không kiểm soát được, để dao động SHAP đo được trong CIES chỉ phản ánh đúng 1 biến độc lập: kỹ thuật xử lý mất cân bằng (và model, khi so sánh cross-model).

Nếu agent cần thêm 1 kỹ thuật/thư viện mới không có trong spec này, **phải dừng lại và hỏi**, không tự quyết định — vì rất có thể nó vi phạm nguyên tắc trên.

---

## 1. DATASET

### 1.1. Dataset chính: Sparkov

- Nguồn: `kaggle.com/datasets/kartik2112/fraud-detection`
- Tải qua `kagglehub.dataset_download("kartik2112/fraud-detection")`
- ~1.85M dòng, 23 cột, fraud rate ~0.52%
- Vai trò: phân tích chính, bao gồm feature-level CIES

### 1.2. Dataset phụ: ULB (ĐÃ CHỐT)

- Nguồn: `kaggle.com/datasets/mlg-ulb/creditcardfraud`
- Tải qua `kagglehub.dataset_download("mlg-ulb/creditcardfraud")`
- ~284K giao dịch, 30 features (28 PCA + Amount + Time), fraud rate ~0.17%
- Vai trò: kiểm chứng xu hướng CIES có khái quát hóa không (chỉ cần aggregate CIES, không cần feature-level)
- **Lưu ý:** Features đã PCA nên SHAP giải thích theo V1..V28, không có ý nghĩa business — chấp nhận được vì chỉ đo aggregate stability

### 1.3. Phân bổ công sức 2 dataset (KHÔNG đối xứng — có chủ đích)

| | Sparkov (chính) | ULB (phụ) |
|---|---|---|
| Số model chạy | Cả 5 | **Cả 5** (so sánh 1:1 chặt chẽ hơn) |
| Số kỹ thuật imbalance | Cả 5 | Cả 5 (không rút gọn — đây là biến chính cần khái quát hóa) |
| Feature-level CIES | Có | Không cần (chỉ cần aggregate CIES) |
| Mục đích | Đóng góp chính | Kiểm chứng xu hướng RQ1/RQ2 có khái quát hóa không |

### 1.4. Bảng benchmark literature (bắt buộc build trước khi viết kết quả)

Tạo file `reports/benchmark_literature.md` liệt kê kết quả PR-AUC/F1 từ các paper đã dùng Sparkov (2024-2026), cột CIES để trống "N/A" cho tất cả — dùng để minh họa khoảng trống nghiên cứu. Không tự bịa số — nếu chưa trích được số cụ thể từ paper, để trống và ghi "chưa trích xuất", không suy diễn.

---

## 2. CẤU TRÚC PROJECT

```
fraud-detection-thesis/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train_models.ipynb
│   └── 04_cies_experiment.ipynb
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── download.py
│   │   └── encoding.py          # Stratified K-fold Target Encoding, one-hot
│   ├── imbalance/
│   │   └── resamplers.py        # wrapper cho 5 kỹ thuật, KHÔNG thêm kỹ thuật khác
│   ├── models/
│   │   └── train.py             # 5 model, cùng bộ feature input cho tất cả (xem mục 5.2)
│   ├── explainability/
│   │   ├── shap_utils.py        # TreeSHAP cho LR/RF/XGB/CatBoost, KernelSHAP/DeepSHAP cho ANN
│   │   └── cies.py              # thuật toán tính CIES
│   └── evaluation/
│       └── metrics.py           # PR-AUC chính, không dùng ROC-AUC làm metric quyết định
├── reports/
│   ├── figures/
│   ├── profiling/
│   └── benchmark_literature.md
├── requirements.txt
└── README.md
```

---

## 3. TIỀN XỬ LÝ — ENCODING

### 3.1. Bảng quyết định encoding theo cardinality (KHÔNG one-hot toàn bộ)

| Feature | Cardinality | Encoding bắt buộc |
|---|---|---|
| gender | 2 | One-hot |
| category | ~14 | One-hot |
| state | ~51 | One-hot |
| merchant | ~600+ | **Stratified K-fold Target Encoding hoặc WOE** |
| city | vài trăm | **Stratified K-fold Target Encoding hoặc WOE** |
| job | vài trăm | **Stratified K-fold Target Encoding hoặc WOE** |

### 3.2. Thuật toán Stratified K-fold Target Encoding (bắt buộc implement đúng thứ tự)

```python
# Pseudo-code bắt buộc tuân thủ — KHÔNG dùng random KFold, PHẢI dùng StratifiedKFold
from sklearn.model_selection import StratifiedKFold

def stratified_kfold_target_encode(df_train, col, target_col, n_splits=5, smoothing=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    encoded = pd.Series(index=df_train.index, dtype=float)
    global_mean = df_train[target_col].mean()

    for train_idx, val_idx in skf.split(df_train, df_train[target_col]):
        fold_train = df_train.iloc[train_idx]
        stats = fold_train.groupby(col)[target_col].agg(['mean', 'count'])
        # Smoothing: category hiếm bị kéo về global_mean
        smoothed = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        encoded.iloc[val_idx] = df_train[col].iloc[val_idx].map(smoothed).fillna(global_mean)

    return encoded
```

**RÀNG BUỘC BẮT BUỘC:**
- `StratifiedKFold`, KHÔNG `KFold` thường
- FIT chỉ trên train; với test set, dùng thống kê tính trên TOÀN BỘ train (không k-fold) để transform — vì test không cần chống leakage nội bộ giữa các fold, chỉ cần không dùng thông tin test
- Lưu lại encoding map (dict/Series) để đảm bảo tái lập được ở bước CIES (mỗi lần resample train phải tính lại encoding riêng — xem mục 6)

### 3.3. Thứ tự pipeline BẮT BUỘC — agent không được đảo thứ tự này

```
1. Split train/test (stratified theo target)
2. Encoding: fit trên train (Stratified K-fold nội bộ) → transform test
3. Imbalance handling: CHỈ áp dụng lên train, SAU khi encode
4. Train model
5. Evaluate trên test (test KHÔNG BAO GIỜ qua resample)
```

### 3.4. Danh sách CẤM (không implement dù có thư viện sẵn)

| Kỹ thuật | Lý do cấm |
|---|---|
| Gray encoding | Chỉ hợp ordinal; categorical ở đây nominal → tạo thứ tự giả, phá SHAP interpretability |
| One-hot cho merchant/city/job | Bùng nổ chiều → CIES không khả thi về compute |
| Fit encoding trên toàn dataset trước khi split | Leakage |
| Entity embedding (deep-learned) | Chỉ áp dụng ANN → bất đối xứng cross-model |

---

## 4. XỬ LÝ MẤT CÂN BẰNG

### 4.1. Danh sách 5 kỹ thuật — CHÍNH XÁC 5, không thêm không bớt

```python
# src/imbalance/resamplers.py — interface bắt buộc
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from sklearn.utils.class_weight import compute_class_weight

IMBALANCE_TECHNIQUES = {
    "smote": SMOTE(random_state=SEED),
    "smote_enn": SMOTEENN(random_state=SEED),
    "adasyn": ADASYN(random_state=SEED),
    "borderline_smote": BorderlineSMOTE(random_state=SEED),
    "class_weighting": None,  # xử lý riêng — truyền class_weight vào model.fit, KHÔNG resample data
}
```

**Ràng buộc quan trọng cho `class_weighting`:** đây là algorithm-level, KHÔNG resample dữ liệu. Agent phải code riêng nhánh này (truyền `class_weight='balanced'` hoặc tính bằng `compute_class_weight` vào model), không dùng chung code path với 4 kỹ thuật resampling kia.

### 4.2. Danh sách CẤM — KHÔNG implement dù user có thể vô tình yêu cầu thêm sau

| Kỹ thuật | Lý do cấm (nhắc lại nếu bị hỏi thêm) |
|---|---|
| GAN-based oversampling (SMOTE-GAN, GANified-SMOTE...) | Thêm lớp bất định hội tụ đối kháng → confound CIES, không tách được nguyên nhân dao động SHAP |
| Focal loss / cost-sensitive nâng cao | Chỉ tự nhiên cho DL → bất đối xứng giữa 5 model |
| Random undersampling đơn thuần | Không có trong scope 5 kỹ thuật đã chốt — nếu muốn thêm phải sửa lại toàn bộ thiết kế RQ, KHÔNG tự thêm |

---

## 5. MODEL

### 5.1. Danh sách 5 model

```python
models = {
    "logistic_regression": LogisticRegression(...),
    "random_forest": RandomForestClassifier(...),
    "xgboost": XGBClassifier(...),
    "catboost": CatBoostClassifier(...),
    "ann": <PyTorch/Keras model>,
}
```

### 5.2. RÀNG BUỘC QUAN TRỌNG NHẤT VỀ CATBOOST

**CatBoost PHẢI dùng chung bộ feature đã encode sẵn (mục 3) như 4 model kia — KHÔNG dùng cơ chế Ordered Target Statistics nội tại của CatBoost.**

Cụ thể: KHÔNG truyền categorical columns thô vào CatBoost qua tham số `cat_features`. Phải truyền feature đã qua Stratified K-fold Target Encoding/one-hot giống hệt input của LR/RF/XGBoost/ANN.

Lý do (đừng "tối ưu" bỏ ràng buộc này): nếu CatBoost dùng encoding riêng, sự khác biệt CIES giữa CatBoost và các model khác không tách bạch được là do bản chất model hay do input feature khác nhau — phá vỡ RQ3.

---

## 6. EXPLAINABILITY — SHAP

### 6.1. Mapping explainer bắt buộc theo model

| Model | Explainer | Ghi chú |
|---|---|---|
| Logistic Regression | `shap.LinearExplainer` | Exact |
| Random Forest | `shap.TreeExplainer` | Exact |
| XGBoost | `shap.TreeExplainer` | Exact |
| CatBoost | `shap.TreeExplainer` | Exact |
| ANN | `shap.KernelExplainer` hoặc `shap.DeepExplainer` | **Xấp xỉ — ghi log riêng, không so sánh thô với TreeSHAP** |

### 6.2. Thí nghiệm đối chứng bắt buộc (nếu đủ thời gian)

Chạy thêm `KernelExplainer` trên 1 model tree-based (ví dụ Random Forest) để đo mức nhiễu riêng của KernelSHAP, tách khỏi nhiễu do model. Lưu kết quả riêng vào `reports/kernelshap_control_experiment.json` — dùng để trả lời câu hỏi "CIES thấp của ANN là do model hay do explainer xấp xỉ".

---

## 7. CIES — THIẾT KẾ THÍ NGHIỆM

### 7.1. Pseudo-code bắt buộc

```python
N_RUNS = 20  # tối thiểu, có thể tăng lên 30 nếu compute cho phép

def run_cies_experiment(model_name, imbalance_technique, df_train, df_test_fixed_eval):
    shap_values_runs = []
    for seed in range(N_RUNS):
        # 1. Bootstrap resample train (KHÔNG đổi df_test_fixed_eval)
        train_resample = df_train.sample(frac=1.0, replace=True, random_state=seed)

        # 2. Encoding — FIT LẠI trên resample này (không dùng encoding đã fit từ lần trước)
        encoded = stratified_kfold_target_encode(train_resample, ..., random_state=seed)

        # 3. Imbalance handling
        X_res, y_res = apply_imbalance_technique(imbalance_technique, encoded, seed=seed)

        # 4. Train model
        model = train_model(model_name, X_res, y_res, seed=seed)

        # 5. SHAP trên tập eval CỐ ĐỊNH — KHÔNG đổi qua các lần lặp
        shap_vals = compute_shap(model, df_test_fixed_eval)
        shap_values_runs.append(shap_vals)

    cies_score = compute_stability_metric(shap_values_runs)  # theo công thức paper CIES gốc
    return cies_score
```

**RÀNG BUỘC KHÔNG ĐƯỢC VI PHẠM:**
- `df_test_fixed_eval` phải là **1 tập cố định duy nhất**, dùng lại y hệt qua toàn bộ N_RUNS — KHÔNG lấy sample eval khác nhau mỗi lần
- Encoding phải fit lại từ đầu ở mỗi run trên chính resample đó — KHÔNG dùng lại encoding map đã tính từ lần trước (nếu không, sẽ không đo đúng dao động do resample)
- Ghi log seed của từng run để tái lập được

### 7.2. Chi phí tính toán — cảnh báo cho agent

Tổng số lần train = 5 model × 5 kỹ thuật × N_RUNS = 500–750 lần (với dataset chính). Agent cần:
- Ưu tiên chạy trên Kaggle Notebook/Colab (GPU), không chạy N_RUNS đầy đủ trên máy local
- Nếu code chạy quá lâu, báo lại cho user để cân nhắc giảm N_RUNS hoặc subsample train trước khi bootstrap — KHÔNG tự ý giảm số kỹ thuật hoặc số model để tiết kiệm thời gian

---

## 8. METRICS

- Metric chính: **PR-AUC** (KHÔNG dùng ROC-AUC làm metric quyết định, có thể báo cáo thêm để tham khảo)
- Báo cáo kèm: Precision@Recall cố định, F1, F2
- Test set giữ nguyên phân phối gốc (fraud rate thật), KHÔNG resample test
- **Dùng stratified random split** (ĐÃ CHỐT) — không dùng time-based split để tránh confound từ concept drift, đảm bảo dao động CIES chỉ phản ánh biến imbalance technique

---

## 9. MÔI TRƯỜNG THỰC THI

| Việc | Ở đâu |
|---|---|
| Viết code, debug, refactor | Local (Antigravity) — dùng subsample nhỏ (vài nghìn dòng) |
| Train thật, chạy CIES experiment đầy đủ | Kaggle Notebook (GPU miễn phí, dataset mount `/kaggle/input/`) |
| Dự phòng khi hết quota Kaggle | Google Colab |

Tải dataset qua `kagglehub`, KHÔNG hardcode đường dẫn tải thủ công.

---

## 10. BẢNG TRA CẤM TUYỆT ĐỐI — agent đọc lại trước khi thêm bất kỳ kỹ thuật mới

| Cấm | Không cấm (đã chốt dùng) |
|---|---|
| GNN, GNNExplainer | TreeSHAP / KernelSHAP / DeepSHAP |
| Gray encoding | One-hot / Stratified K-fold Target Encoding / WOE |
| GAN-based oversampling | SMOTE / SMOTE-ENN / ADASYN / Borderline-SMOTE / Class-weighting |
| Focal loss, entity embedding | Class-weighting (algorithm-level, đã đủ) |
| CatBoost dùng `cat_features` nội tại | CatBoost dùng feature đã encode chung với các model khác |
| Random KFold cho target encoding | StratifiedKFold |
| Eval set thay đổi giữa các run CIES | Eval set cố định xuyên suốt N_RUNS |

**Nếu trong quá trình code, agent thấy có lý do kỹ thuật để làm khác đi so với bảng trên — DỪNG LẠI, không tự sửa, hỏi user trước.**
