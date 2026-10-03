---
title: "Pipeline Fraud Detection CIES — báo cáo trạng thái"
---

# Pipeline Fraud Detection CIES — báo cáo trạng thái

Tài liệu phản ánh **code hiện tại** (cập nhật 2026-10-03), không phải thiết kế trong `AGENT_SPEC.md` (mục 11 của file đó liệt kê các chỗ lệch spec). Sơ đồ kiến trúc: [`system_architecture.html`](system_architecture.html).

## Bảng giải thích từng bước

| Bước | Kỹ thuật dùng | File / Function liên quan |
|---|---|---|
| **Dataset chính** | Sparkov (`kartik2112/fraud-detection`), tải qua `kagglehub`, ~1.85M dòng, fraud rate ~0.52% | `src/data/download.py::download_dataset`; load ở `notebooks/02_preprocessing.ipynb` |
| **Dataset phụ** | ULB (`mlg-ulb/creditcardfraud`), ~284K giao dịch, 28 feature PCA sẵn (V1..V28) + Amount | `notebooks/02_preprocessing.ipynb` (§6) — tải + split + lưu parquet; **CIES đã chạy trên ULB** qua `notebooks/05_cies_experiment_ulb.ipynb` (chưa có benchmark PR-AUC riêng cho ULB) |
| **Feature engineering & cleaning** | Sắp theo thời gian; thêm `hour`, `day_of_week`, `age` (năm giao dịch − năm sinh); bỏ cột ID (`trans_num`, `cc_num`, `first`, `last`, `street`, `zip`, `unix_time`) và **mọi cột định danh khách hàng** (`lat`, `long`, `city`, `state`, `job`, `city_pop`, `merch_lat`, `merch_long`). Feature còn lại: `amt`, `category`, `merchant`, `hour`, `day_of_week`, `age`, `gender` | `src/data/preprocess.py::preprocess_sparkov`; `config.CUSTOMER_IDENTITY_COLS` |
| **Train/Test Split** | **Theo thời gian.** Sparkov: file gốc `fraudTrain.csv` (2019-01-01 → 2020-06-21, 1.296.675 dòng, fraud 0,579%) / `fraudTest.csv` (2020-06-21 → 2020-12-31, 555.719 dòng, fraud 0,386%). ULB: sắp theo `Time`, 20% cuối làm test (fraud 0,183% / 0,132%) | `notebooks/02_preprocessing.ipynb`; `src/data/preprocess.py::split_ulb_by_time` |
| **Encoding — One-hot** | `gender`, `category` | `src/data/encoding.py::encode_train / encode_test` |
| **Encoding — Target Encoding** | Stratified K-Fold Target Encoding cho `merchant` (`StratifiedKFold n_splits=5`, smoothing=10) | `src/data/encoding.py::stratified_kfold_target_encode` |
| **Lưu dữ liệu đã encode** | `train_encoded.parquet`, `test_encoded.parquet`, `encoding_maps.joblib` — dùng cho benchmark (notebook 03) | `notebooks/02_preprocessing.ipynb` (§5) |
| **Lưu dữ liệu raw (chưa encode)** | `train_raw.parquet`, `test_raw.parquet` — bắt buộc cho CIES vì mỗi run phải re-encode từ đầu | `notebooks/02_preprocessing.ipynb` (§5) |
| **Xử lý mất cân bằng** | Đủ 5 kỹ thuật: SMOTE, SMOTE-ENN, ADASYN, Borderline-SMOTE (resample), Class Weighting (`compute_class_weight('balanced')`, không resample data) | `src/imbalance/resamplers.py::apply_imbalance / get_class_weights` |
| **Models** | 5 model: Logistic Regression, Random Forest (CPU), XGBoost, CatBoost (KHÔNG dùng `cat_features` — dùng chung feature đã encode), ANN (PyTorch MLP 128→64→32→1) — XGBoost/CatBoost/ANN **tự động dùng GPU nếu có** (phát hiện qua `nvidia-smi`, không import torch để tránh xung đột OpenMP) | `src/models/train.py::build_model / train_model / _train_ann / _has_gpu` |
| **Hyperparameter tuning** | Optuna HPO (TPE + MedianPruner), tối ưu PR-AUC, validation **theo thời gian**, mỗi fold encode lại chỉ bằng dòng trước khối validation (không để target encoding thấy nhãn tương lai): cửa sổ mở rộng 3 fold, 3 khối validation liên tiếp ở 1/3 cuối tập train (Sparkov: 12/2019→02/2020, 02→04/2020, 04→06/2020). Tham số tốt nhất đóng băng ở `results/best_params.json`, `build_model` tự nạp | `src/models/tune.py::time_series_folds / encode_time_folds / tune_model / load_best_params` |
| **Cách ly subprocess** | Mỗi tổ hợp (model × technique) chạy trong 1 subprocess `spawn` riêng — tránh torch (ANN) + xgboost cùng process gây segfault/hang do xung đột OpenMP. Timeout có escalation SIGTERM → SIGKILL (trước đó `process.join()` có thể treo vô hạn nếu process con bỏ qua SIGTERM — đã tái hiện trên 1 GPU driver hang thực tế trên Kaggle) | `src/utils/isolation.py::run_isolated / _terminate_hard` |
| **Benchmark 25 tổ hợp** | Vòng lặp 5 model × 5 kỹ thuật, đánh giá PR-AUC (chính), F1, F2, ROC-AUC, Precision@Recall | `notebooks/03_train_models.ipynb`; `src/models/train.py::train_and_evaluate_combo`; `src/evaluation/metrics.py::evaluate_model` |
| **SHAP — Linear** | `LinearExplainer` (exact) cho Logistic Regression | `src/explainability/shap_utils.py::_compute_shap_linear` |
| **SHAP — Tree** | `TreeExplainer` (exact) cho Random Forest, XGBoost, CatBoost. Tự nhận diện cả 2 dạng output binary-classification của shap: `list` (API cũ) và ndarray 3 chiều `(n_samples, n_features, n_classes)` — dạng sau đã xác nhận là output thật của `RandomForestClassifier` trên `shap==0.52.0`, trước đây bị bỏ sót gây lỗi shape khi chạy CIES cho Random Forest | `src/explainability/shap_utils.py::_compute_shap_tree` |
| **SHAP — Deep (ANN)** | `DeepExplainer` (xấp xỉ DeepLIFT, tất định, giải thích logit) cho ANN; tự lùi về `KernelExplainer` nếu Deep lỗi. Đã đo trên ANN thật: Kernel chạy lặp 2 lần cùng model chỉ khớp nhau Spearman ~0.85 (nhiễu lấy mẫu riêng, làm điểm CIES của ANN thấp giả) và chậm hơn ~5x; Deep giống hệt qua các lần chạy, cộng dồn đúng. Vẫn là xấp xỉ — không so thô với TreeSHAP | `src/explainability/shap_utils.py::_compute_shap_deep` |
| **CIES — ĐÃ IMPLEMENT** | Bootstrap resample train → re-encode từ đầu → xử lý mất cân bằng → train → SHAP trên eval set cố định, lặp N_RUNS lần. Seed của run thứ i (= i) đặt cho cả bootstrap, resampler và model, nên CIES đo dao động khi **train lại từ đầu** (dữ liệu + tính ngẫu nhiên của thuật toán), không chỉ của dữ liệu. Dữ liệu nền SHAP (LR, ANN) là cùng 100 dòng gốc ở mọi run, encode theo mapping của run | `src/explainability/cies.py::run_cies_experiment` |
| **CIES — Stability metric** | Rank-weighted distance (trọng số `1/min(rank_a, rank_b)` theo THỨ HẠNG từng feature, chuẩn hoá tổng = 1; trước đây bị gán theo vị trí cột nên phụ thuộc thứ tự cột) trung bình trên mọi cặp run, `cies_score = 1 − khoảng cách trung bình`; kèm Spearman trung bình để đối chiếu | `src/explainability/cies.py::compute_stability_metric / compute_rank_weighted_distance / shap_to_ranks` |
| **CIES — Feature-level** | Stability cấp từng feature (mean\|SHAP\|, hệ số biến thiên, rank_std) — chỉ cho Sparkov | `src/explainability/cies.py::compute_feature_level_cies` |
| **CIES — chạy cách ly** | Wrap `run_cies_experiment` trong subprocess riêng | `src/explainability/cies.py::run_cies_experiment_isolated` |
| **Trực quan hoá** | Insight dataset (giờ đêm, category×giờ, số tiền, nhịp giao dịch, tính mùa vụ, ULB top-feature), SHAP (beeswarm, so sánh model, đồng thuận thứ hạng, dependence), CIES (heatmap, ổn định thứ hạng theo feature, top-k, Spearman giữa các run, Sparkov vs ULB, trade-off PR-AUC) | `notebooks/06_visualizations.ipynb`; `src/visualization/dataset.py`, `explain.py`; ảnh ở `reports/figures/` |
| **Lưu kết quả CIES** | `results/cies_summary_results.json`, `cies_summary_results_ulb.json` — ghi từng tổ hợp qua khoá file + ghi atomic, an toàn khi nhiều tiến trình cùng ghi | `src/explainability/cies.py::merge_cies_result`; `src/utils/jsonio.py::update_json` |

## Trạng thái chạy thực nghiệm

| Hạng mục | Trạng thái |
|---|---|
| _(thiết kế trước)_ Tune lần 1 (30 trial, vùng hẹp) | LR / XGBoost / CatBoost / ANN xong (PR-AUC CV 0.319 / 0.929 / 0.926 / 0.892), RF xong 0.857 nhưng `max_depth` chạm biên 18. Nhiều tham số chạm biên (CatBoost `depth`, `iterations`; ANN `epochs`; LR `C`) |
| _(thiết kế trước)_ Tune lần 2 (100 trial, vùng đã nới, cùng ngân sách cho mọi model) | Xong: LR 0.319, XGBoost 0.933, CatBoost 0.9271, RF 0.8872 (70 xong / 30 cắt). **ANN còn lại** (chạy nhiều phiên Kaggle với checkpoint SQLite, ~30/100 trial). `best_params.json` hiện là hỗn hợp hai giao thức cho riêng ANN (30 trial cũ), chưa dùng cho benchmark/CIES của ANN |
| **Thiết kế hiện tại (2026-09-26)** | Chia theo thời gian + bỏ cột định danh khách hàng + tune validate theo thời gian (mục "Lệch so với spec" 9–10). `data/processed/` đã tạo lại theo thiết kế này, kèm 4 tập train đã resample (`data/processed/resampled/train_encoded_<kỹ thuật>.parquet`). Kết quả của thiết kế trước: bảng "thiết kế trước" bên dưới; file gốc xem bằng `git show d562d24:results/cies_summary_results.json`, `git show 0b86295:results/cies_summary_results_ulb.json`, `git show 074fc1f:results/model_benchmark_results.csv` |
| Tune (50 trial, 3 fold thời gian, vùng tìm ban đầu) | **Xong 5 model × 2 dataset.** PR-AUC CV Sparkov: LR 0,2460 · RF 0,9122 · XGBoost 0,9214 · CatBoost 0,9242 · ANN 0,9190; ULB: LR 0,8085 · RF 0,7938 · XGBoost 0,8095 · CatBoost 0,8118 · ANN 0,8028. LR/RF/XGBoost/CatBoost chạy local, ANN trên Kaggle. Kiểm tra biên: `design_decisions.md` mục 8–9 |
| Benchmark Sparkov + ULB | **Xong 25/25 mỗi dataset** — `model_benchmark_results.csv`, `model_benchmark_results_ulb.csv` |
| CIES Sparkov (subsample 100k, N_RUNS=20) + ULB (toàn bộ 227.846 dòng, N_RUNS=20) | **Xong 25/25 mỗi dataset** — `cies_summary_results.json`, `cies_summary_results_ulb.json`; ANN dùng DeepExplainer ở mọi run |
| Đối chứng KernelSHAP (`AGENT_SPEC.md` §6.2) | Chưa làm |

## Kết quả — thiết kế hiện tại (5 model × 5 kỹ thuật × 2 dataset)

**Cách chạy.** Chia theo thời gian, không có cột định danh khách hàng, tham số tune 50 trial trên 3 fold thời gian (vùng tìm ban đầu, `design_decisions.md` mục 8–9), 1 seed (42).
- **Benchmark:** train trên toàn bộ tập train (Sparkov 1.296.675 dòng; ULB 227.846), chấm trên test không resample (Sparkov 555.719 dòng, 2.145 fraud; ULB 56.961 dòng, **75 fraud**).
- **CIES:** 20 lần bootstrap; Sparkov lấy mẫu phân tầng 100.000 dòng (21 feature), ULB toàn bộ train (29 feature); SHAP trên 250 mẫu đánh giá cố định (50 fraud). ANN dùng DeepExplainer ở cả 200 run (`run_logs[i]["explainer"]`).
- Tune và benchmark: LR, RF, XGBoost, CatBoost chạy local; ANN trên Kaggle (GPU T4). CIES: mọi model chạy local (CPU).

Nguồn: `results/model_benchmark_results(_ulb).csv`, `results/cies_summary_results(_ulb).json`. **In đậm**: cao nhất trong 5 kỹ thuật của model đó (PR-AUC, CIES); _nghiêng_: CIES thấp nhất. FP = số giao dịch thật bị báo là fraud ở ngưỡng 0,5.

### Sparkov

| Model | Kỹ thuật | PR-AUC | F1 | FP (ngưỡng 0,5) | CIES | Spearman |
|---|---|---|---|---|---|---|
| Logistic Regression | smote | **0,1215** | 0,0426 | 70.996 | 0,9237 | 0,7876 |
| Logistic Regression | smote_enn | 0,1191 | 0,0417 | 72.817 | 0,9225 | 0,7827 |
| Logistic Regression | adasyn | 0,0958 | 0,0280 | 121.316 | **0,9404** | 0,8587 |
| Logistic Regression | borderline_smote | 0,1208 | 0,0329 | 98.881 | _0,9113_ | 0,7405 |
| Logistic Regression | class_weighting | 0,1206 | 0,0449 | 67.074 | 0,9339 | 0,8238 |
| Random Forest | smote | 0,8465 | 0,7613 | 730 | 0,9609 | 0,9572 |
| Random Forest | smote_enn | 0,8414 | 0,7227 | 1.100 | **0,9637** | 0,9611 |
| Random Forest | adasyn | 0,8378 | 0,7380 | 885 | 0,9565 | 0,9440 |
| Random Forest | borderline_smote | 0,8454 | 0,7833 | 522 | _0,9446_ | 0,9189 |
| Random Forest | class_weighting | **0,8720** | 0,6927 | 1.450 | 0,9559 | 0,9195 |
| XGBoost | smote | **0,8747** | 0,7010 | 1.365 | 0,9563 | 0,9272 |
| XGBoost | smote_enn | 0,8743 | 0,6487 | 1.915 | 0,9583 | 0,9340 |
| XGBoost | adasyn | 0,8597 | 0,6685 | 1.635 | 0,9640 | 0,9539 |
| XGBoost | borderline_smote | 0,8566 | 0,7270 | 1.110 | _0,9525_ | 0,9147 |
| XGBoost | class_weighting | 0,8738 | 0,4594 | 4.606 | **0,9695** | 0,9392 |
| CatBoost | smote | **0,8869** | 0,7319 | 1.109 | 0,9561 | 0,9521 |
| CatBoost | smote_enn | 0,8864 | 0,6833 | 1.578 | 0,9565 | 0,9553 |
| CatBoost | adasyn | 0,8821 | 0,7060 | 1.323 | 0,9515 | 0,9540 |
| CatBoost | borderline_smote | 0,8804 | 0,7356 | 1.054 | _0,9510_ | 0,9475 |
| CatBoost | class_weighting | 0,8809 | 0,4905 | 4.063 | **0,9730** | 0,9763 |
| ANN | smote | **0,8803** | 0,4619 | 4.461 | _0,8690_ | 0,7176 |
| ANN | smote_enn | 0,8774 | 0,4180 | 5.463 | 0,8759 | 0,7610 |
| ANN | adasyn | 0,8751 | 0,3959 | 6.043 | 0,8994 | 0,7880 |
| ANN | borderline_smote | 0,8700 | 0,5262 | 3.303 | 0,8810 | 0,7717 |
| ANN | class_weighting | 0,8655 | 0,2413 | 13.045 | **0,9205** | 0,8642 |

### ULB

| Model | Kỹ thuật | PR-AUC | F1 | FP (ngưỡng 0,5) | CIES | Spearman |
|---|---|---|---|---|---|---|
| Logistic Regression | smote | 0,6865 | 0,1680 | 634 | 0,9011 | 0,7513 |
| Logistic Regression | smote_enn | 0,6859 | 0,1650 | 648 | **0,9016** | 0,7541 |
| Logistic Regression | adasyn | 0,6778 | 0,0423 | 3.116 | 0,8792 | 0,6734 |
| Logistic Regression | borderline_smote | 0,6991 | 0,2630 | 341 | _0,8367_ | 0,5251 |
| Logistic Regression | class_weighting | **0,7007** | 0,0904 | 1.340 | 0,8742 | 0,6287 |
| Random Forest | smote | **0,8194** | 0,5970 | 66 | **0,9507** | 0,9266 |
| Random Forest | smote_enn | 0,8190 | 0,5782 | 75 | 0,9502 | 0,9258 |
| Random Forest | adasyn | 0,8115 | 0,2813 | 316 | 0,9403 | 0,8724 |
| Random Forest | borderline_smote | 0,8096 | 0,7421 | 25 | _0,9277_ | 0,8246 |
| Random Forest | class_weighting | 0,8142 | 0,7468 | 24 | 0,9489 | 0,9101 |
| XGBoost | smote | 0,7907 | 0,6203 | 54 | **0,9204** | 0,8154 |
| XGBoost | smote_enn | 0,7960 | 0,6413 | 50 | 0,9176 | 0,8130 |
| XGBoost | adasyn | 0,7803 | 0,5455 | 85 | 0,9175 | 0,8027 |
| XGBoost | borderline_smote | **0,8072** | 0,7792 | 19 | _0,8992_ | 0,7326 |
| XGBoost | class_weighting | 0,7964 | 0,7755 | 15 | 0,9114 | 0,7565 |
| CatBoost | smote | 0,7874 | 0,4357 | 144 | 0,9255 | 0,8299 |
| CatBoost | smote_enn | 0,7853 | 0,4413 | 144 | 0,9291 | 0,8454 |
| CatBoost | adasyn | 0,7691 | 0,3289 | 240 | 0,9178 | 0,7839 |
| CatBoost | borderline_smote | 0,7777 | 0,7229 | 31 | _0,9033_ | 0,7448 |
| CatBoost | class_weighting | **0,7990** | 0,5622 | 81 | **0,9316** | 0,8430 |
| ANN | smote | 0,8026 | 0,6040 | 66 | **0,8923** | 0,7731 |
| ANN | smote_enn | 0,7894 | 0,5894 | 71 | 0,8866 | 0,7511 |
| ANN | adasyn | 0,7927 | 0,6105 | 57 | 0,8880 | 0,7577 |
| ANN | borderline_smote | **0,8143** | 0,7160 | 29 | _0,8649_ | 0,6715 |
| ANN | class_weighting | 0,7939 | 0,0586 | 2.177 | 0,8841 | 0,6805 |

### Nhận xét (1 seed — là quan sát, chưa kiểm định thống kê)

Hình minh hoạ (`reports/figures/`, notebook 06): thứ hạng kỹ thuật trong từng model — `cies_technique_ranks.png` (nhận xét 1); CIES theo
model × kỹ thuật — `cies_heatmap_sparkov.png`, `cies_heatmap_ulb.png` (2); CIES theo cỡ mẫu — `cies_sensitivity_subsample.png` (3);
số báo nhầm ở ngưỡng 0,5 — `benchmark_false_positives.png` (4); PR-AUC theo kỹ thuật — `benchmark_prauc_by_technique.png` (5, 6);
CIES vs PR-AUC — `cies_vs_prauc.png` (7); đồng thuận feature giữa kỹ thuật — `technique_agreement_summary.png` (8); thứ hạng feature
qua các run của tổ hợp ổn định nhất/kém nhất — `cies_rank_stability_*.png` (9).

1. **Borderline-SMOTE cho CIES thấp nhất ở 9/10 cặp (model, dataset)** — mọi model trên ULB, 4/5 model trên Sparkov (ngoại lệ: ANN Sparkov, nơi SMOTE thấp nhất và Borderline-SMOTE đứng 3/5). Đây là xu hướng nhất quán nhất của thí nghiệm. Về độ lớn: chênh với kỹ thuật kế tiếp 0,0005–0,011 trên Sparkov, 0,012–0,038 trên ULB. So với sai số jackknife của 1 điểm CIES (bỏ lần lượt 1 trong 20 run: 0,002–0,006; ANN Sparkov 0,013): ở cả 9 cặp, Borderline-SMOTE thua kỹ thuật tốt nhất của cùng model hơn 2 lần sai số; nhưng so với kỹ thuật kế tiếp thì 4/9 cặp chưa vượt 2 lần sai số (LR, XGBoost, CatBoost Sparkov; ANN ULB). Chưa chạy nhiều seed.
2. **Model quyết định mức CIES nhiều hơn kỹ thuật trên Sparkov, không rõ bằng trên ULB.** Sparkov: 3 model cây 0,945–0,973, LR 0,911–0,940, **ANN thấp nhất 0,869–0,920**; chênh giữa 5 kỹ thuật trong cùng model chỉ 0,017–0,029 (riêng ANN 0,051). Tách phương sai của 25 điểm CIES (2 chiều, không lặp): model giải thích 86% trên Sparkov, kỹ thuật 7%. ULB: RF cao nhất (0,928–0,951), nhưng chênh giữa kỹ thuật của LR (0,065) ngang khoảng cách giữa các model; model 77%, kỹ thuật 17%. Thứ hạng CIES của 25 tổ hợp tương quan Spearman 0,76 giữa 2 dataset.
3. **CIES trên ULB thấp hơn Sparkov ở 23/25 tổ hợp** (chênh TB 0,030; ngoại lệ: ANN × SMOTE và ANN × SMOTE-ENN) — nhưng **không kết luận được "ULB kém ổn định hơn"**: CIES tăng theo số fraud trong mẫu bootstrap (kiểm tra độ nhạy, mục "Lệch so với spec" 1: RF × SMOTE 0,945 ở 289 fraud → 0,961 ở 579), và mẫu ULB có 417 fraud so với 579 của Sparkov. Chênh lệch giữa 2 dataset vì vậy lẫn giữa đặc tính dữ liệu và cỡ mẫu. Điều chắc chắn: giả thuyết ban đầu "ULB có vài feature PCA gánh tín hiệu nên SHAP ổn định hơn" không được dữ liệu ủng hộ.
4. **`class_weighting`: CIES cao nhất ở XGBoost, CatBoost, ANN trên Sparkov** (không có sinh dòng tổng hợp nên các lần bootstrap chỉ khác nhau ở dữ liệu thật) — **nhưng trên ULB chỉ đúng với CatBoost**, nên không khái quát. Ở ngưỡng 0,5 nó báo nhầm nhiều nhất trên Sparkov: FP của RF 1.450, XGBoost 4.606, CatBoost 4.063, ANN 13.045 (so với 522–6.043 ở các kỹ thuật resample của cùng model) → F1 thấp nhất ở 4/5 model. PR-AUC không phụ thuộc ngưỡng nên ít bị ảnh hưởng (cao nhất ở RF Sparkov, LR và CatBoost ULB). Chưa dò ngưỡng tối ưu.
5. **Hiệu năng.** Model tốt nhất: Sparkov CatBoost × SMOTE (PR-AUC 0,887), ULB RF × SMOTE (0,819). SMOTE cho PR-AUC cao nhất ở 4/5 model trên Sparkov. ADASYN cho PR-AUC thấp nhất ở 2/5 model trên Sparkov (LR, RF) và 3/5 trên ULB (LR, XGBoost, CatBoost). LR rất yếu trên Sparkov (0,10–0,12) nhưng khá trên ULB (0,68–0,70): feature PCA của ULB gần tuyến tính hơn; trên Sparkov tín hiệu `hour` có dạng "cao ban đêm" (22h–3h: 1,4–2,9% fraud; giờ khác ~0,1%) mà 1 hệ số tuyến tính không mô tả được (notebook 06, mục 3).
6. **SMOTE-ENN gần như trùng SMOTE:** chênh PR-AUC ≤ 0,005 trên Sparkov (≤ 0,013 trên ULB), chênh CIES ≤ 0,007 — ENN chỉ bỏ 0,82% số dòng sau SMOTE (2.578.338 → 2.557.109 trên Sparkov) nhưng là bước tốn thời gian nhất (resample Sparkov 145 phút; CIES ULB 25–41 phút/tổ hợp, trong khi phần lớn tổ hợp khác dưới 20 phút).
7. **Hiệu năng và độ ổn định giải thích: độc lập trên Sparkov, cùng chiều trên ULB.** Spearman(CIES, PR-AUC) trên Sparkov 0,15 (3 model cây: −0,07) — vd. CatBoost × SMOTE có PR-AUC cao nhất nhưng CIES chỉ trung bình của CatBoost (0,956). Trên ULB 0,49 (3 model cây: 0,66), nhưng PR-AUC ULB dựa trên 75 fraud nên dao động nhiều. Kết luận "chọn theo PR-AUC không đảm bảo giải thích ổn định" chỉ đứng được trên Sparkov.
8. **Đổi kỹ thuật imbalance ít đổi feature được coi là quan trọng — trừ khi đổi sang `class_weighting`.** Đồng thuận thứ hạng feature (cùng công thức rank-weighted của CIES, trên mean\|SHAP\| trung bình 20 run) giữa các kỹ thuật resample với nhau: 0,947–0,996 trên Sparkov, 0,909–1,000 trên ULB; giữa `class_weighting` và các kỹ thuật resample thấp hơn ở model cây và ANN (Sparkov: RF 0,854–0,901, XGBoost 0,902–0,930, CatBoost 0,922–0,939, ANN 0,903–0,922), riêng LR Sparkov không (0,966–0,980). Hình: `reports/figures/technique_agreement_summary.png`.
9. **CIES và Spearman lệch nhau** ở ANN Sparkov (CIES 0,87–0,92, Spearman 0,72–0,86) và LR ULB (0,84–0,90 / 0,53–0,75): top feature giữ vị trí, phần đuôi xáo trộn — CIES phạt nhẹ phần đuôi (trọng số 1/rank), Spearman thì không.

**Giới hạn cần nói kèm:** 1 seed; test ULB chỉ 75 fraud và mỗi khối validation lúc tune chỉ 24–36 fraud; vài tham số sát biên (`design_decisions.md` mục 8–9: ULB XGBoost `max_depth=3`, CatBoost `depth=4`, ANN `epochs=5`; Sparkov ANN `dropout=0,124`, RF `max_depth=27`); ANN chỉ tái lập được về mặt thống kê giữa các máy (giới hạn 17).

## Tham khảo: kết quả thiết kế trước (chia ngẫu nhiên, còn cột định danh — không so sánh trực tiếp được)



> Cột "CIES ULB" trong bảng dưới được chạy bằng script đọc `ulb_train.parquet` mà **không bỏ cột `Time`** (log: 31 cột = V1..V28 + Amount + Time + Class), tức `Time` bị dùng làm feature — trái với `config.ULB_FEATURE_COLS` và notebook 05. Kết quả đã xoá; lần chạy lại chọn đúng V1..V28 + Amount.

Benchmark (chỉ Sparkov): train trên toàn bộ 1.481.915 dòng, đánh giá trên test cố định (370.479 dòng, không resample). CIES Sparkov: bootstrap từ mẫu phân tầng 100.000 dòng, N_RUNS=20, `feature_level=True`. CIES ULB: bootstrap từ toàn bộ 227.845 dòng train, N_RUNS=20.

| Model | Kỹ thuật | PR-AUC | F1 | CIES Sparkov | Spearman Sparkov | CIES ULB | Spearman ULB |
|---|---|---|---|---|---|---|---|
| Logistic Regression | smote | 0.2334 | 0.0901 | 0.8470 | 0.6017 | 0.8940 | 0.7586 |
| Logistic Regression | smote_enn | 0.2268 | 0.0860 | 0.8534 | 0.6145 | 0.8827 | 0.7347 |
| Logistic Regression | adasyn | 0.2184 | 0.0585 | 0.8475 | 0.6010 | 0.8904 | 0.7766 |
| Logistic Regression | borderline_smote | 0.2577 | 0.1087 | 0.8408 | 0.5491 | 0.8731 | 0.6944 |
| Logistic Regression | class_weighting | 0.2244 | 0.0797 | 0.8575 | 0.6098 | 0.8852 | 0.7841 |
| Random Forest | smote | 0.8773 | 0.8343 | 0.9695 | 0.9500 | 0.9598 | 0.9291 |
| Random Forest | smote_enn | 0.8536 | 0.7772 | 0.9677 | 0.9467 | 0.9580 | 0.9233 |
| Random Forest | adasyn | 0.8786 | 0.8317 | 0.9695 | 0.9498 | 0.9591 | 0.9291 |
| Random Forest | borderline_smote | 0.8692 | 0.8252 | 0.9643 | 0.9344 | 0.9452 | 0.8873 |
| Random Forest | class_weighting | **0.9024** | **0.8432** | **0.9784** | 0.9673 | 0.9531 | 0.9138 |
| XGBoost | smote | 0.9268 | 0.8565 | 0.9588 | 0.9269 | 0.9406 | 0.8714 |
| XGBoost | smote_enn | 0.9074 | 0.7798 | 0.9570 | 0.9234 | 0.9375 | 0.8586 |
| XGBoost | adasyn | 0.9240 | 0.8582 | 0.9595 | 0.9274 | 0.9420 | 0.8724 |
| XGBoost | borderline_smote | 0.9260 | 0.8527 | 0.9554 | 0.9097 | 0.9273 | 0.8381 |
| XGBoost | class_weighting | **0.9328** | 0.6803 | **0.9645** | 0.9237 | 0.9384 | 0.8521 |
| CatBoost | smote | 0.9251 | 0.8466 | 0.9558 | 0.8852 | 0.9243 | 0.8134 |
| CatBoost | smote_enn | 0.9012 | 0.7438 | 0.9545 | 0.8758 | 0.9218 | 0.8025 |
| CatBoost | adasyn | 0.9268 | 0.8507 | 0.9553 | 0.8772 | 0.9187 | 0.8010 |
| CatBoost | borderline_smote | 0.9251 | 0.8428 | 0.9545 | 0.8875 | 0.9117 | 0.7675 |
| CatBoost | class_weighting | **0.9339** | 0.7461 | **0.9571** | 0.8663 | **0.9291** | 0.8465 |

Nguồn: `results/model_benchmark_results.csv`, `results/cies_summary_results.json`, `results/cies_summary_results_ulb.json` (sắp theo `MODEL_NAMES` × `IMBALANCE_TECHNIQUES`). In đậm: cao nhất trong 5 kỹ thuật của model đó (chỉ đánh dấu ở cột có nhắc tới trong nhận xét).

## Lệch so với spec và giới hạn

1. **CIES Sparkov bootstrap từ mẫu phân tầng 100.000 dòng** (`SUBSAMPLE_N`, 579 fraud, giữ tỷ lệ 0,579%) vì train đầy đủ 1,3 triệu dòng quá nặng cho 20 run × 25 tổ hợp; ULB dùng toàn bộ 227.846 dòng (417 fraud). **Kiểm tra độ nhạy trên dữ liệu hiện tại** (notebook 04 mục 8; `results/cies_sensitivity_subsample.json`):

    | Cỡ mẫu (fraud) | 10k (58) | 30k (174) | 50k (289) | 100k (579) |
    |---|---|---|---|---|
    | XGBoost × class_weighting — CIES (std khoảng cách hạng) | 0,9434 (0,0173) | 0,9506 (0,0116) | 0,9571 (0,0109) | 0,9695 (0,0090) |
    | RF × SMOTE | 0,8938 (0,0244) | 0,9362 (0,0160) | 0,9448 (0,0181) | 0,9609 (0,0111) |

    Mốc 100k tính lại độc lập **trùng từng chữ số** với CIES đã lưu (0,9694726194 và 0,9608966982). CIES **tăng theo cỡ mẫu và chưa bão hoà ở 100k**, nhiễu giữa các run nhìn chung giảm. Hệ quả: so sánh **trong cùng dataset** (giữa kỹ thuật, giữa model — cùng cỡ mẫu) là công bằng; so sánh **mức CIES giữa 2 dataset** thì bị lẫn với số fraud trong mẫu bootstrap (Sparkov 579, ULB 417) — xem nhận xét 3 ở mục Kết quả. Mới kiểm 2 tổ hợp; chưa kiểm SMOTE-ENN, CatBoost, ANN.
2. **ULB có tune, benchmark và CIES riêng** (notebook 05): tham số `best_params_ulb.json`, benchmark `model_benchmark_results_ulb.csv`. Test ULB chỉ có ~75 fraud nên PR-AUC trên ULB dao động nhiều hơn Sparkov; mỗi khối validation lúc tune chỉ có 24–36 fraud. Không cần encoding vì feature đã là số, và bỏ cột `Time`.
3. **Tune tách khỏi imbalance**: tham số tối ưu cho trường hợp không xử lý mất cân bằng, dùng chung cho cả 5 kỹ thuật.
4. **CIES của repo là biến thể của CIES gốc** (arXiv:2603.05024): gốc nhiễu hoá đầu vào lúc suy luận, so độ lớn SHAP, chuẩn hoá theo độ lớn giải thích gốc; repo nhiễu hoá dữ liệu huấn luyện và so thứ hạng. Bảng so sánh chi tiết và nguồn cho các quyết định khác: [`literature_support.md`](literature_support.md).
5. **`timeout=None`** cho tune/benchmark: không còn lưới an toàn nếu tiến trình treo thật.
6. **Sparkov là dữ liệu mô phỏng**; nhiều hình dạng "quá sạch" là dấu vết của bộ sinh dữ liệu.
7. **Thời gian resample SMOTE-ENN** tăng nhanh hơn tuyến tính theo cỡ dữ liệu (bước ENN tìm láng giềng trên toàn bộ tập sau SMOTE): 1 lần resample toàn bộ train Sparkov (1.296.675 dòng) mất 145 phút, nên benchmark đọc tập đã resample từ cache (`data/processed/resampled/`); CIES resample lại ở mỗi run nhưng trên mẫu 100.000 dòng.
8. **Vùng tìm Optuna giữ như ban đầu** dù vài tham số sát biên (XGBoost Sparkov `n_estimators=550`/600, `learning_rate=0,0137`; RF Sparkov `max_depth=27`/30; LR ULB `C` ở 12% thang log). Đã thử nới biên: PR-AUC CV không đổi (XGBoost Sparkov 0,9214 → 0,9212 với 1850 cây, lr 0,0043; LR ULB 0,8085 → 0,8089) mà mỗi trial chậm ~4×. Chi tiết: [`design_decisions.md`](design_decisions.md) mục 8. RF chưa được kiểm tra bằng vùng nới.
9. **Không đưa cột định danh khách hàng vào model** (`config.CUSTOMER_IDENTITY_COLS`, cùng `cc_num`, `zip`). Gộp lại chúng chỉ ra đúng 1 người: 98,6% cặp (lat, long) và 90,1% giá trị `city_pop` chỉ thuộc 1 thẻ, 92,2% city chỉ có 1 thẻ, `zip` ứng đúng 1 cặp tọa độ, tọa độ cửa hàng luôn trong ~1,4° quanh nhà khách. Model dùng chúng để học thuộc "khách nào từng bị hack" (762/983 thẻ bị hack trong giai đoạn train). `age`, `gender` giữ lại vì là hồ sơ nhân khẩu học (Sparkov sinh fraud theo hồ sơ), không định danh.
10. **Chia train/test theo thời gian** (thay chia ngẫu nhiên theo dòng). Chia theo dòng khiến 100% thẻ có fraud ở test (845/845) cũng có dòng fraud ở train. Đo trên XGBoost (tham số tune cũ, `class_weighting`, PR-AUC; 1 lần chia, 1 seed):

    | Cách chia | Đủ feature | Bỏ nhóm định danh (lat/long, city, state, job) | Bỏ thêm `city_pop` + tọa độ cửa hàng (thiết kế hiện tại) | Bỏ cả `age`, `gender`, `city_pop` |
    |---|---|---|---|---|
    | Ngẫu nhiên theo dòng (trước) | 0,932 | — | — | 0,828 |
    | Theo thẻ (`cc_num`) | 0,914 | — | — | 0,820 |
    | **Theo thời gian (hiện tại)** | **0,240** | 0,881 | **0,882** (pipeline thật: 0,883) | 0,765 |

    Chia theo thẻ bị loại vì không phải cách đánh giá chuẩn trong literature và không áp dụng được cho ULB (không có mã khách hàng). Luật của Fraud Detection Handbook "bỏ thẻ đã bị lộ khỏi tập test" không dùng: trên Sparkov nó bỏ 87% tập test và đẩy tỷ lệ fraud 0,39% → 2,9%, vì bộ mô phỏng không bao giờ khoá thẻ bị hack. Tham chiếu: Wu (arXiv:2607.14686, 2026) dùng đúng tập test gốc của Sparkov, không định danh, đạt AP ≈ 0,93 với feature số tiền tự thiết kế.

11. **ULB có giao dịch trùng giữa train và test nhưng không có fraud trùng.** 1.722 dòng test giống hệt 1 dòng train trên V1..V28 + Amount — **cả 1.722 đều hợp lệ** (0/75 fraud của test), và nhãn luôn giống nhau → không có rò rỉ nhãn fraud, PR-AUC ULB không bị thổi phồng bởi việc nhớ đáp án. Giữ nguyên (không khử trùng lặp), như phần lớn các nghiên cứu dùng ULB. Sparkov (7 feature) chỉ có 26 dòng test như vậy, cũng không có fraud.
12. **Thang đo của SHAP khác nhau giữa các model:** RF theo xác suất; LR, XGBoost, CatBoost, ANN theo log-odds (đã kiểm: tổng SHAP + giá trị nền tái tạo đúng xác suất của RF, log-odds của CatBoost; LR khớp `coef·(x − mean nền)` trong không gian đã scale). CIES chỉ so **thứ hạng trong cùng 1 model**, và biểu đồ so sánh giữa các model (notebook 06) chuẩn hoá theo % tổng của từng model, nên không bị ảnh hưởng.
13. **`shap` 0.52 đọc sai giá trị nền (`expected_value`) của XGBoost 3.x** — lệch 1 hằng số (0,063 trong phép thử). Giá trị SHAP từng feature thì **trùng khít** với TreeSHAP do chính XGBoost tính (`pred_contribs`, sai số 0), nên CIES và mọi biểu đồ (chỉ dùng SHAP từng feature) không bị ảnh hưởng. Không dùng `expected_value` của XGBoost cho việc gì.
14. **Target encoding của tập train dùng 5-fold xáo trộn (out-of-fold)**: giá trị encode của 1 dòng train có thể dùng nhãn của các dòng train xảy ra sau nó. Tập test và các khối validation lúc tune không bao giờ thấy nhãn tương lai (test dùng thống kê của toàn bộ train; tune encode riêng từng fold theo thời gian). Đây là cách target encoding chuẩn (Micci-Barreca 2001); Fraud Detection Handbook thì tính đặc trưng rủi ro theo cửa sổ quá khứ có độ trễ — chỉ ảnh hưởng feature lúc train, không ảnh hưởng tính hợp lệ của đánh giá.
15. **F1 và FP đo ở ngưỡng 0,5**: resample về 1:1 hoặc đánh trọng số lớp làm xác suất dự đoán bị đẩy lên so với tỷ lệ fraud thật, nên F1 ở ngưỡng cố định không so sánh công bằng giữa các kỹ thuật (xem nhận xét 4 ở mục Kết quả). PR-AUC không phụ thuộc ngưỡng là chỉ số chính.
16. **CIES đo dao động khi train lại từ đầu, không chỉ dao động của dữ liệu.** Run thứ i dùng seed i cho bootstrap, cho resampler (SMOTE…) và cho model (khởi tạo ANN, lấy mẫu của RF/XGBoost). Vậy CIES gộp 2 nguồn: dữ liệu train đổi (bootstrap) và tính ngẫu nhiên của thuật toán. Các yếu tố cố định giữa 20 run: tập eval, tham số model, kỹ thuật imbalance, dữ liệu nền SHAP. 5 kỹ thuật dùng chung seed nên chung 20 mẫu bootstrap (so sánh theo cặp).
17. **ANN chỉ tái lập được về mặt thống kê giữa các máy.** Cùng code, cùng seed, chạy trên GPU Kaggle và CPU local: mean\|SHAP\| từng run lệch tới 40–140% ở vài feature (khác phần cứng → khác thứ tự phép tính số thực → khác trọng số sau train), nhưng CIES gần như trùng (ANN × SMOTE 0,843 / 0,838; ANN × class_weighting 0,906 / 0,907). Model còn lại tái lập từng chữ số trên cùng máy (đã kiểm LR × SMOTE: 20/20 run trùng tuyệt đối).
18. **PR-AUC và CIES đến từ 2 cách train khác nhau.** PR-AUC: model train 1 lần trên toàn bộ train (Sparkov 1,3 triệu dòng). CIES: 20 model train trên bootstrap của mẫu 100.000 dòng (Sparkov), cùng tham số đã tune trên toàn bộ train. Tương quan PR-AUC–CIES (nhận xét 7) vì vậy so 2 đại lượng đo trên 2 cỡ dữ liệu khác nhau; ULB không bị vì CIES dùng toàn bộ train.

## Đánh đổi đã đo

**Ép one-hot hợp lệ (`config.SNAP_SYNTHETIC_ONEHOT`)**: SMOTE sinh cột one-hot phân số (Sparkov 300k: 90% dòng tổng hợp có ≥1 cột phân số, 81% "bật" >1 category). Công tắc này ép mỗi nhóm one-hot về 1 category (argmax) cho SMOTE/SMOTE-ENN/ADASYN/Borderline-SMOTE (không áp cho class_weighting), nhưng PR-AUC tụt mạnh (1 seed):

| | Không resample | SMOTE gốc | Snap argmax | Lấy mẫu theo trọng số | SMOTENC |
|---|---|---|---|---|---|
| XGBoost | 0.905 | 0.887 | 0.766 | 0.736 | 0.733 |
| Random Forest | 0.815 | 0.719 | 0.573 | 0.547 | 0.487 |

Ba cách hợp lệ độc lập cho kết quả gần nhau nên không phải lỗi của một cách làm; giả thuyết: giá trị phân số vô tình cô lập dòng tổng hợp khỏi vùng dữ liệu thật. Hệ quả: các kỹ thuật resample sẽ trông kém hơn rõ so với trước và so với class_weighting. **Quyết định: mặc định `False` (SMOTE thuần).** Lý do: (1) khớp spec và cách làm chuẩn trong literature nên so sánh được; (2) trên `xgboost × smote`, 100k dòng, CIES gần như không đổi giữa hai chế độ (0,957 khi `False`, 0,959 khi `True`; std hạng 0,0052 và 0,0042) trong khi PR-AUC tụt mạnh khi `True`; (3) ép hợp lệ làm các kỹ thuật resample bị thiệt so với class_weighting vì lý do chưa được giải thích chắc chắn. Hạn chế: dòng tổng hợp có one-hot phân số (không hợp lệ về ngữ nghĩa). `True` giữ làm ablation. So sánh CIES chỉ có một tổ hợp, chưa kiểm ở tổ hợp khác.

**Prior của target encoding** (nay chỉ từ fold train): giá trị encode lệch tối đa 5.6e-7 (thang ~0.005), `test_encoded` không đổi ⇒ kết quả Optuna đã có vẫn hợp lệ.

**Explainer của ANN**: KernelExplainer chạy lặp 2 lần chỉ khớp Spearman ~0.85 và chậm ~5×; DeepExplainer tất định, cộng dồn đúng, nên là mặc định. Vẫn là xấp xỉ, không so thô với TreeSHAP.

## Nhật ký sửa lỗi (mỗi lỗi có test hồi quy)

- **`.gitignore` nuốt `src/data/`** (`data/` trần) → clone mới không import được CIES. Nay chỉ ignore `/data/`.
- **ANN không scale input** (feature cỡ 1e9, 1e6): PR-AUC 0.16 so với 0.74 khi scale. Nay `StandardScaler` (`input_scaler`) áp cho cả predict và SHAP. Mọi số ANN cũ không hợp lệ.
- **ANN crash khi `len(train) % batch_size == 1`** (`BatchNorm1d`) → bỏ batch lẻ 1 mẫu.
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
- **`save_cies_results()` ghi đè cả file bằng list trong bộ nhớ riêng của tiến trình gọi nó** — an toàn với 1 tiến trình tuần tự, nhưng khi chạy nhiều tiến trình CIES song song thì tiến trình ghi sau xoá mất kết quả tiến trình khác vừa thêm — **đã xảy ra thật**, mất 1 tổ hợp đã xong. Thêm `merge_cies_result()` (chỉ thay đúng 1 tổ hợp); notebook 04/05 nay cũng ghi qua hàm này.
- **`age` tính bằng `2020 − năm sinh` (hardcode)** thay vì năm giao dịch: dữ liệu trải 2019–2020, ~50% dòng (năm 2019) bị thừa 1 tuổi. `age` xếp hạng 4–20/79 theo SHAP nên không bỏ qua được. Đã sửa, chạy lại benchmark + CIES Sparkov (lệch ≤ 0,005 so với trước). ULB không bị ảnh hưởng.
- **Ghi file kết quả dùng chung không khoá** (`tune._merge_write`, `merge_cies_result`): `_merge_write` xoá trắng file rồi mới ghi, và cả hai coi JSON đọc lỗi là file rỗng — tiến trình đọc trúng lúc file đang ghi dở sẽ ghi lại file chỉ còn mục của nó. Stress test 8 tiến trình: bản cũ còn 24/200 mục, bản mới đủ 200/200. Nay đi qua `src/utils/jsonio.py::update_json` (khoá `fcntl`, ghi file tạm + `os.replace`, báo lỗi khi JSON hỏng).
- **Notebook Kaggle sẽ bỏ qua gần hết việc** (phát hiện trước khi chạy): `git clone` mang theo kết quả đã commit nên benchmark/CIES tưởng 20/20 tổ hợp đã xong, ANN sẽ dùng tham số 30 trial cũ, và bước khôi phục không bao giờ chép tiến độ phiên trước. Nay mỗi phiên chỉ bỏ kết quả clone về của các model trong `MODELS_SCOPE` rồi khôi phục tiến độ phiên trước cho chúng, còn model ngoài phạm vi giữ bản trong repo (để chạy được nửa local nửa Kaggle); thêm ngân sách thời gian chung cho cả 3 giai đoạn.
- **Notebook 05 chạy CIES ULB bằng tham số của Sparkov**: `run_cies_experiment_isolated` không truyền `params`, nên `build_model` tự nạp `best_params.json` (Sparkov). Nay notebook 05 tune ULB riêng, và cả benchmark lẫn CIES truyền `params=ulb_params(model_name)` (báo lỗi nếu thiếu, không âm thầm dùng tham số Sparkov). Cache resample của ULB ở thư mục riêng `resampled_ulb/` — chung thư mục thì ghi đè cache của Sparkov.
- **`encode_test` ghép lệch hàng khi `df_test` có index khác 0..n-1** (phát hiện khi rà soát, chưa ảnh hưởng kết quả nào): cột one-hot dựng từ `pd.Categorical` có index 0..n-1 được ghép theo nhãn index → số dòng nhân đôi, nửa là NaN. Mọi nơi gọi hiện tại đều truyền index mặc định (đã kiểm: `test_encoded.parquet` tái tạo khớp từng giá trị, không NaN). Nay `encode_test` reset index như `encode_train`.
- **CIES không ghi explainer thực sự đã dùng**: ANN tự lùi DeepExplainer → KernelExplainer khi Deep lỗi (vd. version `shap` khác trên Kaggle), mà Kernel có nhiễu lấy mẫu riêng làm CIES thấp giả tạo — trước đây không để lại dấu vết trong kết quả, chỉ có dòng cảnh báo trong log. Nay mỗi run ghi `run_logs[i]["explainer"]` (`linear`/`tree`/`deep`/`kernel`).
- **Dữ liệu nền SHAP của LR và ANN đổi theo từng run CIES**: nền là 100 dòng đầu của tập bootstrap của chính run đó. LinearExplainer và DeepExplainer đo SHAP so với nền (LR: `coef·(x − trung bình nền)`), nên mốc so sánh dịch theo run và thứ hạng dao động thêm vì nền chứ không vì model — nhất là các cột one-hot `category_*` hiếm (trung bình trên 100 dòng của 1 category ~3% dao động mạnh) và `amt` (đuôi dài). Kiểm chứng: giữ nguyên các model đã train, chỉ cố định nền → CIES LR × SMOTE Sparkov 0,912 → 0,924, ANN × SMOTE 0,838 → 0,867. TreeExplainer không dùng nền nên RF/XGBoost/CatBoost không bị ảnh hưởng. Nay nền là cùng 100 dòng gốc của tập train ở mọi run, encode theo mapping của run. Đã chạy lại CIES của LR và ANN trên 2 dataset: Sparkov LR tăng 0,005–0,013, ANN tăng 0,014–0,043; ULB đổi từ −0,011 đến +0,008 (feature PCA liên tục, chỉ `Amount` hưởng lợi). Kết luận định tính giữ nguyên (Borderline-SMOTE thấp nhất 9/10 cặp; cây > LR > ANN trên Sparkov), khoảng cách cây với LR/ANN hẹp lại.

**Trạng thái kết quả:** xem bảng "Trạng thái chạy thực nghiệm" — 4 model × 2 dataset đã có tune, benchmark, CIES theo thiết kế hiện tại; còn ANN.

## Dọn dẹp code (rà soát dư thừa, không đổi hành vi)

Không sửa kết quả nào ở trên — chỉ bỏ phần chết/trùng lặp, xác nhận bằng `pyflakes` (0 cảnh báo trên `src` và `tests`) và `pytest` (21 test vẫn pass):
- `format_metrics_table()` trong `metrics.py` — chưa từng được gọi ở notebook/test nào, đã xoá (34 dòng).
- 7 import không dùng (`typing.Any`/`Tuple`, `SEED`, `predict_proba`, `classification_report`, `os`...) và 1 biến cục bộ chết (`n_runs` trong `compute_feature_level_cies`).
- `config.MODELS_DIR` — tạo thư mục `models/` ở gốc repo nhưng không nơi nào dùng (không model nào được lưu ra đĩa); đã xoá hằng số và thư mục rỗng.
- `config.KAGGLE_DATASET` — alias trùng với `KAGGLE_DATASET_SPARKOV`, chỉ `download.py` dùng; gộp lại dùng thẳng `KAGGLE_DATASET_SPARKOV`.
- `config.ULB_FILE` được định nghĩa nhưng notebook 02 hardcode `'creditcard.csv'` thay vì dùng — sửa notebook dùng hằng số, đúng nguyên tắc "không hardcode" ghi ngay trong docstring `config.py`.
- `tune_all_models`: bỏ đoạn đọc `best_params.json` đầu hàm mà không dùng tới (mỗi lần ghi đã tự đọc lại qua `_merge_write`, đọc trước đó chỉ tốn công vô ích).
- 6 import test không dùng trong `tests/test_pipeline.py` (`ONEHOT_COLS`, `TARGET_ENCODE_COLS`, `evaluate_model`, `sample_hyperparameters`, và `train_model`/`predict_proba` ở top-level — đã có bản import cục bộ riêng trong 2 test dùng chúng).
- **Nhãn `model_kind` thay cho đoán qua key dict.** `train.py`/`shap_utils.py` từng phân biệt LR/ANN/model thường bằng 6 chỗ lặp lại `isinstance(model, dict) and "scaler"/"model" in model` — suy luận ngầm, dễ vỡ nếu thêm key trùng tên. Nay `build_model()` gắn `"kind": "lr"`/`"ann"` tường minh lúc tạo; `model_kind()` (public, `src/models/train.py`) là điểm đọc duy nhất, raise `ValueError` nếu dict thiếu nhãn thay vì đoán bừa. Đã kiểm lại đủ 5 model (build → train → predict_proba → SHAP) qua process riêng biệt, không gộp torch+xgboost+catboost vào 1 process (đúng ràng buộc cô lập của `isolation.py`).
- **Dọn dẹp 2026-09-27:** bỏ hằng màu `C_B` và 4 hàm vẽ không còn dùng (`plot_topk_frequency`, `plot_pairwise_spearman`, `plot_cies_dataset_compare`, `plot_condition_agreement`; `dataset.py`: `plot_hour_category_heatmap`, `plot_velocity`); beeswarm và biểu đồ thứ hạng theo run cố định phần xê dịch ngẫu nhiên của chấm (ảnh giống hệt giữa các lần chạy, git không còn báo thay đổi giả); CatBoost không ghi thư mục log `catboost_info/` (`allow_writing_files=False`, chỉ là file log, không đổi model).

## Việc tiếp theo

1. Chạy nhiều seed cho vài tổ hợp để biết chênh lệch CIES giữa các kỹ thuật (vd. Borderline-SMOTE thấp nhất ở 9/10 cặp, nhưng chỉ chênh 0,0005–0,011 trên Sparkov) có vượt nhiễu hay không.
2. Đối chứng KernelSHAP (`AGENT_SPEC.md` §6.2).
3. (Tuỳ chọn) ablation `SNAP_SYNTHETIC_ONEHOT=True` trên vài tổ hợp; dò ngưỡng cho `class_weighting` (F1 thấp ở ngưỡng 0,5 — nhận xét 4).
4. Quyết định có tính thêm CIES đúng công thức gốc (nhiễu đầu vào) làm chỉ số phụ hay không.
