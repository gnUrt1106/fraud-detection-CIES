# AGENT_SPEC.md — Đặc tả kỹ thuật cho pipeline nghiên cứu CIES

> File này dành cho coding agent (Antigravity/Claude Code) đọc và thực thi khi xây dựng project.
> **Cập nhật 2026-09-21:** các mục 2, 3.2 và 6.1 đã chỉnh cho khớp code; mục **11** (cuối file) liệt kê mọi chỗ code hiện tại lệch bản gốc và trạng thái từng chỗ. Các ràng buộc bắt buộc ở những mục còn lại giữ nguyên.
> **Thiết kế chạy lại theo seed:** mục **12** mô tả cách đo kết quả chính (yêu cầu của giảng viên): mẫu SHAP, backend GPU/native; các dòng tương ứng ở mục 11. Mục 7 (bootstrap) là thiết kế đầu tiên, không còn là cách đo kết quả chính.
> Đây KHÔNG phải tài liệu học thuật để đọc hiểu — đây là spec kỹ thuật với quyết định đã chốt, ràng buộc bắt buộc, và lý do (để agent không tự "sửa cho tối ưu hơn" và vô tình phá vỡ tính hợp lệ của thí nghiệm.

---

## 0. MỤC TIÊU HỆ THỐNG

Xây dựng pipeline đo **CIES (Credibility Index via Explanation Stability)** — độ ổn định của SHAP values — qua tổ hợp (5 model × 5 kỹ thuật xử lý mất cân bằng), trên 2 dataset (1 chính, 1 phụ để kiểm chứng khái quát hóa).

**Nguyên tắc thiết kế tối thượng — áp dụng cho MỌI quyết định implement:**
> Mọi bước trong pipeline phải loại bỏ được các nguồn bất định (confound) không kiểm soát được, để dao động SHAP đo được trong CIES chỉ phản ánh đúng 1 biến độc lập: kỹ thuật xử lý mất cân bằng (và model, khi so sánh cross-model).

Nếu agent cần thêm 1 kỹ thuật/thư viện mới không có trong spec này, **phải dừng lại và hỏi**, không tự quyết định — vì rất có thể nó vi phạm nguyên tắc trên.

---

## 1. DATASET

### 1.1. Dataset: Sparkov

- Nguồn: `kaggle.com/datasets/kartik2112/fraud-detection`
- Tải qua `kagglehub.dataset_download("kartik2112/fraud-detection")`
- ~1.85M dòng, 23 cột, fraud rate ~0.52%
- Vai trò: một trong 2 dataset ĐỐI XỬ NGANG NHAU (theo yêu cầu giảng viên, không phân chính/phụ); đo CIES ổn định giữa các lần chạy lại

### 1.2. Dataset: ULB (ĐÃ CHỐT)

- Nguồn: `kaggle.com/datasets/mlg-ulb/creditcardfraud`
- Tải qua `kagglehub.dataset_download("mlg-ulb/creditcardfraud")`
- ~284K giao dịch, 30 features (28 PCA + Amount + Time), fraud rate ~0.17%
- Vai trò: dataset thứ hai, đối xử NGANG Sparkov (cùng thiết kế chạy lại theo seed, cùng công thức CIES, SHAP trên toàn bộ test; tham số Optuna riêng)
- **Lưu ý:** Features đã PCA nên SHAP giải thích theo V1..V28, không có ý nghĩa business — chấp nhận được vì thước đo chính là CIES ổn định giữa các lần chạy (không cần phân tích feature-level)

### 1.3. Phân bổ công sức 2 dataset (ĐỐI XỨNG — theo yêu cầu giảng viên)

| | Sparkov | ULB |
|---|---|---|
| Số model chạy | Cả 5 | **Cả 5** (so sánh 1:1 chặt chẽ hơn) |
| Số kỹ thuật imbalance | Cả 5 | Cả 5 (không rút gọn — đây là biến chính cần khái quát hóa) |
| Thước đo | CIES ổn định giữa các lần chạy lại (aggregate) | CIES ổn định giữa các lần chạy lại (aggregate) |
| Mục đích | Trả lời RQ1/RQ2 | Trả lời RQ1/RQ2 (kết quả hai dataset đặt cạnh nhau, không phân chính/phụ) |

### 1.4. Bảng benchmark literature (bắt buộc build trước khi viết kết quả)

Tạo file `reports/benchmark_literature.md` liệt kê kết quả PR-AUC/F1 từ các paper đã dùng Sparkov (2024-2026), cột CIES để trống "N/A" cho tất cả — dùng để minh họa khoảng trống nghiên cứu. Không tự bịa số — nếu chưa trích được số cụ thể từ paper, để trống và ghi "chưa trích xuất", không suy diễn.

---

## 2. CẤU TRÚC PROJECT

```
fraud-detection-cies/
├── data/                        # KHÔNG nằm trong git (.gitignore: /data/)
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── kaggle_pipeline.ipynb        # tune → benchmark → CIES trên Kaggle (xem mục 11)
│   ├── 03_train_models.ipynb
│   ├── 04_cies_experiment.ipynb
│   ├── 05_cies_experiment_ulb.ipynb # CIES trên ULB
│   └── 06_visualizations.ipynb
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── download.py
│   │   ├── preprocess.py        # sắp/chia theo thời gian, bỏ cột định danh khách hàng (xem mục 8)
│   │   └── encoding.py          # Stratified K-fold Target Encoding, one-hot
│   ├── imbalance/
│   │   ├── resamplers.py        # wrapper cho 5 kỹ thuật, KHÔNG thêm kỹ thuật khác
│   │   └── gpu_enn.py           # bước ENN của SMOTE-ENN trên GPU (chỉ dùng khi check trùng bản CPU)
│   ├── models/
│   │   ├── train.py             # 5 model, cùng bộ feature input cho tất cả (xem mục 5.2)
│   │   └── tune.py              # Optuna HPO (ngoài spec gốc — xem mục 11)
│   ├── explainability/
│   │   ├── shap_utils.py        # Linear/Tree SHAP; ANN dùng DeepExplainer (Kernel dự phòng)
│   │   ├── cies.py              # thuật toán tính CIES (bootstrap) + công thức dùng chung
│   │   ├── rerun_stability.py   # chạy lại theo seed (cách đo chính, mục 12)
│   │   ├── deeplift_native.py   # DeepLIFT theo lô cho ANN (GPU), khớp shap.DeepExplainer
│   │   └── rf_gpu_shap.py       # TreeSHAP của Random Forest qua XGBoost (GPU), khớp shap.TreeExplainer
│   ├── evaluation/
│   │   └── metrics.py           # PR-AUC chính, không dùng ROC-AUC làm metric quyết định
│   ├── visualization/           # dataset.py, explain.py
│   └── utils/
│       ├── isolation.py         # chạy mỗi model/tổ hợp trong subprocess riêng
│       └── jsonio.py            # ghi file kết quả dùng chung: khoá + ghi atomic
├── results/                     # best_params.json, kết quả benchmark/CIES
├── reports/
│   ├── figures/
│   ├── profiling/
│   ├── benchmark_literature.md
│   ├── pipeline_report.md
│   └── system_architecture.html
├── scripts/                     # run_rerun.py (CLI chạy lại theo seed), gpu_smote_enn.py, setup_gpu_env.py
├── tests/                       # test_pipeline.py; test của phần chạy lại theo seed, GPU và backend native
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
| merchant | ~600+ | **Stratified K-fold Target Encoding hoặc WOE** |
| state, city, job, lat/long, city_pop, merch_lat/long, zip | — | **Không đưa vào model** — định danh khách hàng (cập nhật 2026-09-26, xem §8 và §11) |

### 3.2. Thuật toán Stratified K-fold Target Encoding (bắt buộc implement đúng thứ tự)

```python
# Pseudo-code bắt buộc tuân thủ — KHÔNG dùng random KFold, PHẢI dùng StratifiedKFold
from sklearn.model_selection import StratifiedKFold

def stratified_kfold_target_encode(df_train, col, target_col, n_splits=5, smoothing=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    encoded = pd.Series(index=df_train.index, dtype=float)

    for train_idx, val_idx in skf.split(df_train, df_train[target_col]):
        fold_train = df_train.iloc[train_idx]
        # CẬP NHẬT: prior tính CHỈ từ fold train (bản gốc dùng trung bình toàn bộ train,
        # làm nhãn của fold validation lọt vào prior). Test set vẫn dùng thống kê toàn train.
        fold_mean = fold_train[target_col].mean()
        stats = fold_train.groupby(col)[target_col].agg(['mean', 'count'])
        # Smoothing: category hiếm bị kéo về fold_mean
        smoothed = (stats['count'] * stats['mean'] + smoothing * fold_mean) / (stats['count'] + smoothing)
        encoded.iloc[val_idx] = df_train[col].iloc[val_idx].map(smoothed).fillna(fold_mean)

    return encoded
```

**RÀNG BUỘC BẮT BUỘC:**
- `StratifiedKFold`, KHÔNG `KFold` thường
- FIT chỉ trên train; với test set, dùng thống kê tính trên TOÀN BỘ train (không k-fold) để transform — vì test không cần chống leakage nội bộ giữa các fold, chỉ cần không dùng thông tin test
- Lưu lại encoding map (dict/Series) để đảm bảo tái lập được ở bước CIES (mỗi lần resample train phải tính lại encoding riêng — xem mục 6)

### 3.3. Thứ tự pipeline BẮT BUỘC — agent không được đảo thứ tự này

```
1. Split train/test THEO THỜI GIAN (Sparkov: fraudTrain.csv → fraudTest.csv; ULB: sắp theo Time, 20% cuối) — xem §8
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
| ANN | `shap.KernelExplainer` hoặc `shap.DeepExplainer` (**hiện dùng `DeepExplainer`**, tự lùi về Kernel nếu lỗi) | **Xấp xỉ — ghi log riêng, không so sánh thô với TreeSHAP** |

### 6.2. Thí nghiệm đối chứng bắt buộc (nếu đủ thời gian)

Chạy thêm `KernelExplainer` trên 1 model tree-based (ví dụ Random Forest) để đo mức nhiễu riêng của KernelSHAP, tách khỏi nhiễu do model. Lưu kết quả riêng vào `reports/kernelshap_control_experiment.json` — dùng để trả lời câu hỏi "CIES thấp của ANN là do model hay do explainer xấp xỉ".

---

## 7. CIES — THIẾT KẾ THÍ NGHIỆM

> **Từ 2026-10 đây là thiết kế đầu tiên (bootstrap). Cách đo kết quả chính là chạy lại theo seed — xem §12.**

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

    cies_score = compute_stability_metric(shap_values_runs)  # biến thể của CIES gốc (xem §11)
    return cies_score
```

**RÀNG BUỘC KHÔNG ĐƯỢC VI PHẠM:**
- `df_test_fixed_eval` phải là **1 tập cố định duy nhất**, dùng lại y hệt qua toàn bộ N_RUNS — KHÔNG lấy sample eval khác nhau mỗi lần
- Encoding phải fit lại từ đầu ở mỗi run trên chính resample đó — KHÔNG dùng lại encoding map đã tính từ lần trước (nếu không, sẽ không đo đúng dao động do resample)
- Ghi log seed của từng run để tái lập được

### 7.2. Chi phí tính toán — cảnh báo cho agent

Tổng số lần train = 5 model × 5 kỹ thuật × N_RUNS = 500–750 lần (với mỗi dataset). Agent cần:
- Ưu tiên chạy trên Kaggle Notebook/Colab (GPU), không chạy N_RUNS đầy đủ trên máy local
- Nếu code chạy quá lâu, báo lại cho user để cân nhắc giảm N_RUNS hoặc subsample train trước khi bootstrap — KHÔNG tự ý giảm số kỹ thuật hoặc số model để tiết kiệm thời gian

---

## 8. METRICS

- Metric chính: **PR-AUC** (KHÔNG dùng ROC-AUC làm metric quyết định, có thể báo cáo thêm để tham khảo)
- Báo cáo kèm: Precision@Recall cố định, F1, F2
- Test set giữ nguyên phân phối gốc (fraud rate thật), KHÔNG resample test
- **Chia train/test THEO THỜI GIAN** (người dùng chốt 2026-09-26, thay cho "stratified random split" của bản gốc): train là quá khứ, test là tương lai, như literature fraud (Fraud Detection Handbook, Le Borgne et al.; Dal Pozzolo et al., TNNLS 2018) và như cách Sparkov được phát hành. Lý do bỏ chia ngẫu nhiên: fraud đến theo đợt trên cùng thẻ, chia theo dòng khiến 100% thẻ có fraud ở test cũng có fraud ở train. Lo ngại gốc về concept drift được chấp nhận vì đó là điều kiện thật của bài toán.
- **Không đưa cột định danh khách hàng vào model** (`config.CUSTOMER_IDENTITY_COLS`): khi chia theo thời gian, model học thuộc "khách nào từng bị hack" và sụp (XGBoost PR-AUC 0,24 khi giữ, 0,88 khi bỏ).
- **Tune validate theo thời gian** (cửa sổ mở rộng 3 fold, `tune.py::time_series_folds`), cùng logic với cách chia train/test.

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
| Mẫu SHAP khác nhau giữa các tổ hợp của một dataset | Một mẫu SHAP cố định duy nhất cho mỗi dataset (§12) |

**Nếu trong quá trình code, agent thấy có lý do kỹ thuật để làm khác đi so với bảng trên — DỪNG LẠI, không tự sửa, hỏi user trước.**

---

## 11. CẬP NHẬT SO VỚI BẢN GỐC

Mục này ghi lại những chỗ code hiện tại **khác hoặc vượt** so với spec ở trên. Cột "Trạng thái" cho biết đã được người dùng yêu cầu/chấp thuận hay còn cần xác nhận. Ràng buộc bắt buộc ở các mục 0–10 không bị sửa ngoài các chỗ ghi trong bảng.

| Mục | Bản gốc | Hiện tại | Trạng thái |
|---|---|---|---|
| §3.2 Target encoding | Prior làm mịn = trung bình toàn bộ train | Prior = trung bình **fold train**; test vẫn dùng thống kê toàn train | Người dùng yêu cầu. Đã sửa pseudo-code ở §3.2 |
| §7.1 Bootstrap + encoding | `stratified_kfold_target_encode(train_resample)` | Bootstrap có dòng trùng nên CIES truyền `groups` (id dòng gốc) và dùng `StratifiedGroupKFold` để bản sao cùng fold; đường tune/benchmark vẫn `StratifiedKFold` | Sửa rò rỉ nhãn; cần xác nhận nếu đưa vào luận văn |
| §4.1 Kỹ thuật imbalance | SMOTE/ADASYN/Borderline/SMOTE-ENN thuần | Mặc định SMOTE thuần (`config.SNAP_SYNTHETIC_ONEHOT=False`); đặt `True` để ép dòng tổng hợp về one-hot hợp lệ (argmax). Đánh đổi đã đo: PR-AUC XGBoost 0.887 → 0.766, Random Forest 0.719 → 0.573 (Sparkov 300k) | Đã chọn `False` (khớp spec); `True` giữ làm ablation. Chi tiết ở `reports/pipeline_report.md` |
| §5 Model | Không nói về scale | Logistic Regression và ANN có `StandardScaler` (cây bất biến với scale). ANN không scale cho PR-AUC 0.16 so với 0.74 (Sparkov) | Chi tiết kỹ thuật, không đổi bộ feature |
| §5/§7 Hyperparameter | Không có | Optuna HPO (`src/models/tune.py`): tune **trước**, trên dữ liệu chưa xử lý imbalance, rồi đóng băng cho mọi kỹ thuật để không thêm biến gây nhiễu | **Ngoài spec gốc** — đã dùng xuyên suốt; cần xác nhận chính thức nếu đưa vào luận văn |
| §6.1 Explainer ANN | Kernel hoặc Deep | `DeepExplainer` (Kernel tự chạy lại nhiễu: 2 lần chạy cùng model chỉ khớp Spearman ~0.85, và chậm hơn ~5×) | Trong phạm vi spec cho phép |
| §6.2 Đối chứng KernelSHAP | Bắt buộc nếu đủ thời gian | **Chưa làm** (`reports/kernelshap_control_experiment.json` chưa tồn tại) | Còn nợ |
| §7.1 Công thức CIES | "Theo công thức paper CIES gốc" | Khoảng cách thứ hạng có trọng số `1/min(hạng_a, hạng_b)` trên từng feature; `cies_score = 1 − khoảng cách TB`; hoà hạng = hạng trung bình; run có SHAP toàn 0 bị loại | **Đã đối chiếu (arXiv:2603.05024): khác công thức gốc** — gốc nhiễu hoá đầu vào, so độ lớn SHAP, chuẩn hoá theo `‖φ(x)‖_w`; repo nhiễu hoá dữ liệu huấn luyện, so thứ hạng. Xem `reports/literature_support.md` |
| §1.2 ULB | 30 feature gồm `Time` | Bỏ cột `Time` (`ULB_FEATURE_COLS`: V1..V28 + Amount) | Quyết định trong `config.py`, nay được áp dụng nhất quán |
| §1.3 / §7.2 Phạm vi chạy | Cả 5 model × 5 kỹ thuật ở cả 2 dataset; không tự giảm để tiết kiệm thời gian | Notebook 04 và 05 chạy đủ 5×5; Sparkov subsample phân tầng 100.000 dòng (`SUBSAMPLE_N`), ULB dùng toàn bộ | **Lệch nhẹ** — chỉ subsample Sparkov; đã kiểm tra độ nhạy 10k–100k trên dữ liệu hiện tại (`xgboost × class_weighting` 0,943–0,970, `random_forest × smote` 0,894–0,961): CIES tăng theo cỡ mẫu nên chỉ so trong cùng dataset (xem `pipeline_report.md`) |
| §8 Chia train/test | Stratified random split, cấm chia theo thời gian | Chia theo thời gian (Sparkov: file gốc; ULB: theo `Time`) | **Người dùng chốt 2026-09-26** sau khi đo: chia ngẫu nhiên rò rỉ theo đợt hack thẻ. Chi tiết và số đo ở `reports/pipeline_report.md` |
| §3.1 Feature/encoding | One-hot `state`; target encoding `city`, `job` | Bỏ `state`, `city`, `job`, `lat`/`long`, `city_pop`, `merch_lat`/`merch_long`, `zip` (định danh khách hàng); còn `amt`, `category`, `merchant`, `hour`, `day_of_week`, `age`, `gender` | **Người dùng chốt 2026-09-26** (XGBoost, chia theo thời gian: PR-AUC 0,24 khi giữ, 0,88 khi bỏ) |
| §5/§7 Validation khi tune | (không quy định) | Validation theo thời gian, cửa sổ mở rộng 3 fold, thay 5-fold xáo trộn; mỗi fold fit encoding riêng trên dòng trước khối validation | **Người dùng chốt 2026-09-26** — nhất quán với cách chia train/test |
| §9 Môi trường | Local dùng subsample nhỏ; train thật trên Kaggle | LR, RF, XGBoost, CatBoost chạy đầy đủ trên máy local (tune, benchmark, CIES); ANN chạy trên Kaggle GPU qua `kaggle_pipeline.ipynb` (`MODELS_SCOPE = ["ann"]`, `RUN_ULB = True`). Mỗi tổ hợp chạy trong subprocess `spawn`; `timeout=None` cho tune và benchmark | Đúng tinh thần spec |
| §7 Thiết kế đo CIES (kết quả chính) | Bootstrap train 20 run, encode lại mỗi run | Chạy lại 10 seed (0–9) trên toàn bộ train, không bootstrap, encoding fit một lần; PR-AUC và CIES từ cùng các run; tham số Optuna đóng băng; hai dataset đối xử ngang nhau | **Giảng viên yêu cầu.** Xem §12. Bootstrap giữ làm thiết kế đầu tiên (kết quả tham khảo) |
| §7.1 Tập eval SHAP | Một tập eval cố định | Sparkov: mẫu cố định phân tầng theo nhãn = 20% tập test (111.144/555.719 dòng, 429/2.145 fraud); ULB: toàn bộ test (56.961 dòng) | **Giảng viên ban đầu yêu cầu toàn bộ test; mẫu 20% chưa được giảng viên xác nhận.** Bằng chứng: 15 tổ hợp có cả bản test đầy đủ lẫn bản 20% (PR-AUC giống hệt, \|ΔCIES\| ≤ 0,0022); CatBoost và Random Forest chưa đối chiếu |
| §6.1 Cách tính SHAP | `shap.*Explainer` | Thêm backend `native` (cùng thuật toán, nhanh hơn, dùng GPU): XGBoost `pred_contribs` / CatBoost `ShapValues`; ANN DeepLIFT Rescale theo lô; Random Forest chuyển sang XGBoost rồi TreeSHAP của XGBoost. Tên explainer ghi trong từng kết quả | Đã kiểm khớp bản `shap` đến sai số float32 (tests). ULB dùng `shap` cho Random Forest và ANN |
| §4.1 SMOTE-ENN | Chạy CPU | Bước ENN chạy trên GPU (`gpu_enn.py`), chính xác (float64), chỉ được dùng sau khi `gpu_smote_enn.py check` xác nhận trùng khít file CPU | Đã check seed 0: trùng khít (2.557.107 dòng, 1.288.931 fraud); ~44 phút/seed so với ~145 phút trên CPU |
| §9 Môi trường | Local subsample nhỏ; Kaggle GPU | Windows, CPU hoặc GPU NVIDIA tùy tổ hợp; mỗi (model, kỹ thuật) trọn 10 seed trong một lần chạy | Do thời gian và giới hạn Kaggle; thiết bị ghi trong `config` của từng kết quả |
| §3.3 Chia train/test | Một tỉ lệ cho cả hai dataset | Sparkov 70/30 (hai file gốc, theo thời gian); ULB 80/20 (theo thời gian) | Mỗi dataset giữ chia theo thời gian chuẩn của nó; "20%" ở Sparkov là 20% của tập test, ở ULB là 20% của cả dataset — báo cáo ghi số dòng tuyệt đối |

### Quy ước kỹ thuật thêm
- Tham số tối ưu lưu ở `results/best_params.json` (Sparkov, `build_model` tự nạp) và `results/best_params_ulb.json` (ULB — benchmark/CIES ULB truyền tường minh `params=`, báo lỗi nếu thiếu, không bao giờ dùng tham số Sparkov cho ULB).
- File kết quả luôn sắp theo `MODEL_NAMES` × `IMBALANCE_TECHNIQUES`; mỗi run CIES ghi explainer thực sự đã dùng (`run_logs[i]["explainer"]`).
- Kết quả benchmark/CIES lưu ngay sau mỗi tổ hợp và bỏ qua tổ hợp đã có khi chạy lại.
- Không nạp torch và xgboost trong cùng một process (xem `src/utils/isolation.py`).

---

## 12. THIẾT KẾ CHẠY LẠI THEO SEED (yêu cầu giảng viên) — cách đo kết quả chính

Thay §7 cho kết quả chính. Mã: `src/explainability/rerun_stability.py`; CLI: `scripts/run_rerun.py` (pilot / cache / train / status / merge).

**Ràng buộc bắt buộc:**
- 10 seed (0–9). Train trên TOÀN BỘ train, phân phối giữ nguyên; encoding fit MỘT lần (`train_encoded.parquet`); mỗi run chỉ đổi seed của resampler và model — nguồn dao động duy nhất là tính ngẫu nhiên của thuật toán.
- PR-AUC (trên TOÀN BỘ test) và SHAP lấy từ CÙNG model của mỗi run — không tách nhánh hiệu năng / nhánh ổn định.
- Tham số model giữ nguyên từ Optuna (`results/best_params.json`, ULB: `best_params_ulb.json`), không tune lại. Thiếu file tham số thì dừng, không âm thầm dùng tham số mặc định.
- CIES = công thức hiện có (`compute_stability_metric`) trên 10 bảng hạng (45 cặp). KHÔNG đổi công thức.
- Sparkov và ULB đối xử NGANG nhau: cùng thiết kế, cùng 5 model × 5 kỹ thuật, cùng 10 seed (không chính/phụ).
- Mỗi (model, kỹ thuật) chạy trọn 10 seed trong MỘT lần chạy, cùng thiết bị: khác thiết bị hay kiến trúc thì phép tính số thực khác, lẫn vào chính thứ cần đo. Dữ liệu đã resample (cache theo kỹ thuật + seed, không phụ thuộc model) dùng chung cho mọi model. `merge` báo: trùng tổ hợp, thiếu seed, lệch dấu vân tay dữ liệu, lệch mẫu SHAP.
- **Tập eval SHAP:** Sparkov = mẫu CỐ ĐỊNH, phân tầng theo nhãn, bằng 20% tập test (111.144/555.719 dòng; 429/2.145 fraud); ULB = toàn bộ test (56.961 dòng, 75 fraud). MỘT mẫu duy nhất cho mọi model, kỹ thuật và seed (`config.RERUN_SHAP_TEST_FRACTION`, `resolve_shap_rows`, `stratified_eval_indices`). KHÔNG lấy mẫu khác nhau giữa các lần chạy. KHÔNG đổi cỡ mẫu giữa chừng: kết quả chỉ lưu mean|SHAP| từng seed nên đổi mẫu là train lại. Giảng viên ban đầu yêu cầu toàn bộ test — mẫu 20% **cần giảng viên xác nhận**.
- Run có SHAP toàn 0 bị loại; tổ hợp cần ≥ 2 seed thành công. Model tất định theo seed (Logistic Regression + class_weighting) cho CIES = 1: hệ quả của thiết kế, `diagnostics.identical_mean_abs_shap` đánh dấu.
- CIES chế độ này KHÔNG so sánh trực tiếp với CIES bootstrap (45 cặp thay vì 190; dao động chỉ do thuật toán).

**Chi tiết kỹ thuật (đã kiểm khớp bản chuẩn, không đổi ý nghĩa thí nghiệm):**
- `--shap-backend`: `shap` (thư viện) | `native` | `auto` (native cho XGBoost/CatBoost, `shap` cho phần còn lại). `native` chỉ có cho xgboost, catboost, ann, random_forest; Logistic Regression luôn là `LinearExplainer`.
- Random Forest `native`: mỗi cây sklearn → cây XGBoost (cấu trúc, đặc trưng tách giữ nguyên; ngưỡng đổi sang float32 liền sau để `x ≤ thr` ⇔ `x < s`; cover = `weighted_n_node_samples`; giá trị lá = P(lớp 1)/số cây), tính `pred_contribs` (GPUTreeShap nếu có GPU). Cộng dồn theo cây nên chia nhóm cây không đổi kết quả.
- ANN `native`: DeepLIFT Rescale theo lô trên GPU, cùng thuật toán `shap.DeepExplainer`; chỉ hỗ trợ MLP Linear/ReLU/BatchNorm1d/Dropout 1 logit, kiến trúc khác báo lỗi, không lùi âm thầm.
- torch và xgboost không nạp chung một tiến trình (`run_isolated`). `--force-cpu` đặt `CUDA_VISIBLE_DEVICES=-1`.
- `train` ghi `results/rerun/<name>.json` (`--name`, mặc định `main`) sau MỖI tổ hợp hoàn tất (tổ hợp bị ngắt giữa chừng mất toàn bộ seed của nó); chạy lại cùng lệnh sẽ bỏ qua tổ hợp đã xong.
- Kết quả chạy SHAP trên toàn bộ test (đối chiếu mẫu) đặt ở `results/full_test_reference/`, KHÔNG để trong `results/rerun/`.
