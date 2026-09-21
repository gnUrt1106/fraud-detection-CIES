---
title: "Pipeline Fraud Detection CIES — báo cáo trạng thái"
---

# Pipeline Fraud Detection CIES — báo cáo trạng thái

Tài liệu phản ánh **code hiện tại** (cập nhật 2026-09-21), không phải thiết kế trong `AGENT_SPEC.md` (mục 11 của file đó liệt kê các chỗ lệch spec). Sơ đồ kiến trúc: [`system_architecture.html`](system_architecture.html).

## Bảng giải thích từng bước

| Bước | Kỹ thuật dùng | File / Function liên quan |
|---|---|---|
| **Dataset chính** | Sparkov (`kartik2112/fraud-detection`), tải qua `kagglehub`, ~1.85M dòng, fraud rate ~0.52% | `src/data/download.py::download_dataset`; load ở `notebooks/02_preprocessing.ipynb` |
| **Dataset phụ** | ULB (`mlg-ulb/creditcardfraud`), ~284K giao dịch, 28 feature PCA sẵn (V1..V28) + Amount | `notebooks/02_preprocessing.ipynb` (§6) — tải + split + lưu parquet; **CIES đã chạy trên ULB** qua `notebooks/05_cies_experiment_ulb.ipynb` (chưa có benchmark PR-AUC riêng cho ULB) |
| **Feature engineering & cleaning** | Bỏ cột ID/định danh cá nhân (`trans_num`, `cc_num`, `first`, `last`, `street`); derive `hour`, `day_of_week` từ `trans_date_trans_time`; derive `age` từ `dob` | `notebooks/02_preprocessing.ipynb::preprocess_features` |
| **Train/Test Split** | Stratified random split 80/20 theo `is_fraud` (`sklearn.train_test_split`, seed=42) | `notebooks/02_preprocessing.ipynb` (§3) |
| **Encoding — One-hot** | One-hot cho low-cardinality: `gender`, `category`, `state` | `src/data/encoding.py::encode_train / encode_test` |
| **Encoding — Target Encoding** | Stratified K-Fold Target Encoding cho high-cardinality: `merchant`, `city`, `job` (`StratifiedKFold n_splits=5`, smoothing=10) | `src/data/encoding.py::stratified_kfold_target_encode` |
| **Lưu dữ liệu đã encode** | `train_encoded.parquet`, `test_encoded.parquet`, `encoding_maps.joblib` — dùng cho benchmark (notebook 03) | `notebooks/02_preprocessing.ipynb` (§5) |
| **Lưu dữ liệu raw (chưa encode)** | `train_raw.parquet`, `test_raw.parquet` — bắt buộc cho CIES vì mỗi run phải re-encode từ đầu | `notebooks/02_preprocessing.ipynb` (§5) |
| **Xử lý mất cân bằng** | Đủ 5 kỹ thuật: SMOTE, SMOTE-ENN, ADASYN, Borderline-SMOTE (resample), Class Weighting (`compute_class_weight('balanced')`, không resample data) | `src/imbalance/resamplers.py::apply_imbalance / get_class_weights` |
| **Models** | 5 model: Logistic Regression, Random Forest (CPU), XGBoost, CatBoost (KHÔNG dùng `cat_features` — dùng chung feature đã encode), ANN (PyTorch MLP 128→64→32→1) — XGBoost/CatBoost/ANN **tự động dùng GPU nếu có** (phát hiện qua `nvidia-smi`, không import torch để tránh xung đột OpenMP) | `src/models/train.py::build_model / train_model / _train_ann / _has_gpu` |
| **Hyperparameter tuning (tùy chọn)** | Optuna HPO, tham số tốt nhất đóng băng ở `results/best_params.json`, nạp lại qua `build_model` nếu có | `src/models/tune.py::load_best_params` |
| **Cách ly subprocess** | Mỗi tổ hợp (model × technique) chạy trong 1 subprocess `spawn` riêng — tránh torch (ANN) + xgboost cùng process gây segfault/hang do xung đột OpenMP. Timeout có escalation SIGTERM → SIGKILL (trước đó `process.join()` có thể treo vô hạn nếu process con bỏ qua SIGTERM — đã tái hiện trên 1 GPU driver hang thực tế trên Kaggle) | `src/utils/isolation.py::run_isolated / _terminate_hard` |
| **Benchmark 25 tổ hợp** | Vòng lặp 5 model × 5 kỹ thuật, đánh giá PR-AUC (chính), F1, F2, ROC-AUC, Precision@Recall | `notebooks/03_train_models.ipynb`; `src/models/train.py::train_and_evaluate_combo`; `src/evaluation/metrics.py::evaluate_model` |
| **SHAP — Linear** | `LinearExplainer` (exact) cho Logistic Regression | `src/explainability/shap_utils.py::_compute_shap_linear` |
| **SHAP — Tree** | `TreeExplainer` (exact) cho Random Forest, XGBoost, CatBoost. Tự nhận diện cả 2 dạng output binary-classification của shap: `list` (API cũ) và ndarray 3 chiều `(n_samples, n_features, n_classes)` — dạng sau đã xác nhận là output thật của `RandomForestClassifier` trên `shap==0.52.0`, trước đây bị bỏ sót gây lỗi shape khi chạy CIES cho Random Forest | `src/explainability/shap_utils.py::_compute_shap_tree` |
| **SHAP — Deep (ANN)** | `DeepExplainer` (xấp xỉ DeepLIFT, tất định, giải thích logit) cho ANN; tự lùi về `KernelExplainer` nếu Deep lỗi. Đã đo trên ANN thật: Kernel chạy lặp 2 lần cùng model chỉ khớp nhau Spearman ~0.85 (nhiễu lấy mẫu riêng, làm điểm CIES của ANN thấp giả) và chậm hơn ~5x; Deep giống hệt qua các lần chạy, cộng dồn đúng. Vẫn là xấp xỉ — không so thô với TreeSHAP | `src/explainability/shap_utils.py::_compute_shap_deep` |
| **CIES — ĐÃ IMPLEMENT** | Bootstrap resample train → re-encode từ đầu → xử lý mất cân bằng → train → SHAP trên eval set cố định, lặp N_RUNS lần | `src/explainability/cies.py::run_cies_experiment` |
| **CIES — Stability metric** | Rank-weighted distance (trọng số `1/min(rank_a, rank_b)` theo THỨ HẠNG từng feature, chuẩn hoá tổng = 1; trước đây bị gán theo vị trí cột nên phụ thuộc thứ tự cột) trung bình trên mọi cặp run, `cies_score = 1 − khoảng cách trung bình`; kèm Spearman trung bình để đối chiếu | `src/explainability/cies.py::compute_stability_metric / compute_rank_weighted_distance / shap_to_ranks` |
| **CIES — Feature-level** | Stability cấp từng feature (mean\|SHAP\|, hệ số biến thiên, rank_std) — chỉ cho Sparkov | `src/explainability/cies.py::compute_feature_level_cies` |
| **CIES — chạy cách ly** | Wrap `run_cies_experiment` trong subprocess riêng | `src/explainability/cies.py::run_cies_experiment_isolated` |
| **Trực quan hoá** | Insight dataset (giờ đêm, category×giờ, số tiền, nhịp giao dịch, tính mùa vụ, ULB top-feature), SHAP (beeswarm, so sánh model, đồng thuận thứ hạng, dependence), CIES (heatmap, ổn định thứ hạng theo feature, top-k, Spearman giữa các run, Sparkov vs ULB, trade-off PR-AUC) | `notebooks/06_visualizations.ipynb`; `src/visualization/dataset.py`, `explain.py`; ảnh ở `reports/figures/` |
| **Lưu kết quả CIES** | `results/cies_summary_results.json` | `src/explainability/cies.py::save_cies_results` |

## Trạng thái chạy thực nghiệm

| Hạng mục | Trạng thái |
|---|---|
| Tune Logistic Regression / XGBoost / CatBoost / ANN | Xong — PR-AUC CV 0.319 / 0.929 / 0.926 / 0.892 |
| Tune Random Forest | **Chưa xong** (chạy riêng bằng `MODELS_TO_TUNE = ["random_forest"]`) |
| Benchmark 25 tổ hợp (03), CIES Sparkov (04), CIES ULB (05) | **Chưa có kết quả** trên code/dữ liệu hiện tại |
| Đối chứng KernelSHAP (`AGENT_SPEC.md` §6.2) | Chưa làm |

## Lệch so với spec và giới hạn

1. **Notebook 04/05 chạy đủ 5×5.** Sparkov subsample **phân tầng 100.000 dòng** (`SUBSAMPLE_N`, ≈521 fraud, giữ tỷ lệ 0,52%) vì train đầy đủ 1.48M dòng quá nặng cho 20 run/tổ hợp; mẫu 10.000 cũ chỉ có ~59 fraud (bootstrap còn ~37 fraud khác nhau) nên CIES sẽ đo nhiễu mẫu nhỏ. Cỡ 100k chưa được kiểm chứng bằng thực nghiệm: notebook 04 mục 8 (`RUN_SENSITIVITY`) chạy CIES `xgboost × class_weighting` ở 10k/30k/50k/100k để xem điểm có bão hoà không. ULB dùng toàn bộ 227.845 dòng train; SMOTE-ENN ở cỡ này chậm, chạy đủ 25 tổ hợp mất vài giờ.
2. **ULB chưa có benchmark PR-AUC/F1 riêng**, chỉ có CIES; không cần encoding vì feature đã là số, và bỏ cột `Time`.
3. **Tune tách khỏi imbalance**: tham số tối ưu cho trường hợp không xử lý mất cân bằng, dùng chung cho cả 5 kỹ thuật.
4. **CIES của repo là biến thể của CIES gốc** (arXiv:2603.05024): gốc nhiễu hoá đầu vào lúc suy luận, so độ lớn SHAP, chuẩn hoá theo độ lớn giải thích gốc; repo nhiễu hoá dữ liệu huấn luyện và so thứ hạng. Bảng so sánh chi tiết và nguồn cho các quyết định khác: [`literature_support.md`](literature_support.md).
5. **`timeout=None`** cho tune/benchmark: không còn lưới an toàn nếu tiến trình treo thật.
6. **Sparkov là dữ liệu mô phỏng**; nhiều hình dạng "quá sạch" là dấu vết của bộ sinh dữ liệu.
7. **Thời gian resample SMOTE-ENN** tăng ~4× mỗi lần gấp đôi dữ liệu (25k→1.4s, 50k→5.1s, 100k→19.7s ⇒ ≈1.2 giờ cho 1.48M dòng, chưa tính train).

## Đánh đổi đã đo

**Ép one-hot hợp lệ (`config.SNAP_SYNTHETIC_ONEHOT`)**: SMOTE sinh cột one-hot phân số (Sparkov 300k: 90% dòng tổng hợp có ≥1 cột phân số, 81% "bật" >1 category). Nay mỗi nhóm one-hot được ép về 1 category (argmax) cho SMOTE/SMOTE-ENN/ADASYN/Borderline-SMOTE (không áp cho class_weighting), nhưng PR-AUC tụt mạnh (1 seed):

| | Không resample | SMOTE gốc | Snap argmax | Lấy mẫu theo trọng số | SMOTENC |
|---|---|---|---|---|---|
| XGBoost | 0.905 | 0.887 | 0.766 | 0.736 | 0.733 |
| Random Forest | 0.815 | 0.719 | 0.573 | 0.547 | 0.487 |

Ba cách hợp lệ độc lập cho kết quả gần nhau nên không phải lỗi của một cách làm; giả thuyết: giá trị phân số vô tình cô lập dòng tổng hợp khỏi vùng dữ liệu thật. Hệ quả: các kỹ thuật resample sẽ trông kém hơn rõ so với trước và so với class_weighting. `False` = SMOTE thuần.

**Prior của target encoding** (nay chỉ từ fold train): giá trị encode lệch tối đa 5.6e-7 (thang ~0.005), `test_encoded` không đổi ⇒ kết quả Optuna đã có vẫn hợp lệ.

**Explainer của ANN**: KernelExplainer chạy lặp 2 lần chỉ khớp Spearman ~0.85 và chậm ~5×; DeepExplainer tất định, cộng dồn đúng, nên là mặc định. Vẫn là xấp xỉ, không so thô với TreeSHAP.

## Nhật ký sửa lỗi (mỗi lỗi có test hồi quy)

- **`.gitignore` nuốt `src/data/`** (`data/` trần) → clone mới không import được CIES. Nay chỉ ignore `/data/`.
- **ANN không scale input** (feature cỡ 1e9, 1e6): PR-AUC 0.16 so với 0.74 khi scale. Nay `StandardScaler` (`input_scaler`) áp cho cả predict và SHAP. Mọi số ANN cũ không hợp lệ.
- **ANN crash khi `len(train) % batch_size == 1`** (`BatchNorm1d`) → `drop_last`.
- **`scale_pos_weight` của XGBoost tính ngược** (~0.0017 thay vì ~577): PR-AUC `xgboost × class_weighting` trên ULB 0.877 → 0.704 khi sai.
- **Trọng số khoảng cách CIES gán theo chỉ số cột** thay vì thứ hạng: cùng một cú đổi chỗ top-2 cho 0.122 hoặc 0.030. Mọi `cies_score` cũ không hợp lệ.
- **Model hằng số → SHAP toàn 0 → CIES giả = 1.0**: nay run như vậy bị loại.
- **`TreeExplainer` trả mảng 3 chiều cho RandomForest** (shap 0.52) → chuẩn hoá về `(n_mẫu, n_feature)`.
- **`encode_train` lệch hàng khi index không mặc định** (19% NaN) → gán theo vị trí.
- **Bootstrap có dòng trùng gây rò nhãn trong target encoding** → `groups` + `StratifiedGroupKFold` (chỉ cho CIES).
- **Hoà hạng xếp theo thứ tự cột** → hạng trung bình.
- **ULB đưa nhầm cột `Time`** → chỉ V1..V28 + Amount.
- **Ghép PR-AUC với CIES sai khoá** (`imbalance_technique` vs `technique`) → tích Descartes.
- **Thiếu khoá `mean_spearman`** ở đường lỗi CIES → `KeyError` sập vòng lặp.
- **`pip install optuna>=...` không có ngoặc kép** → `>` thành chuyển hướng file.
- **`run_isolated`: `join()` không timeout sau `terminate()`** → treo hơn 5 giờ trên Kaggle; nay leo thang SIGKILL.
- **`daemon=True` ép joblib về `n_jobs=1`** → RF chậm; nay `daemon=False`.
- **`tune_all_models` chỉ ghi file cuối vòng lặp** → nay lưu/merge từng model.
- **Logistic Regression không hội tụ** vì feature chưa scale → thêm `StandardScaler`.

**Cần chạy lại:** notebook 03, 04, 05 (mọi kết quả cũ không còn hợp lệ).

## Việc tiếp theo

1. Tune Random Forest trên Kaggle.
2. Chạy 03, 04, 05; mở rộng 04/05 lên 5×5 trước khi báo cáo.
3. Quyết định giữ hay tắt `SNAP_SYNTHETIC_ONEHOT` sau khi có số benchmark thật.
4. Quyết định có tính thêm CIES đúng công thức gốc (nhiễu đầu vào) làm chỉ số phụ hay không; làm đối chứng KernelSHAP.
5. Chạy lại notebook 06 với kết quả thật.
