---
title: "Pipeline Fraud Detection CIES — báo cáo trạng thái"
---

# Pipeline Fraud Detection CIES — báo cáo trạng thái

Tài liệu phản ánh **code hiện tại** (cập nhật 2026-09-26), không phải thiết kế trong `AGENT_SPEC.md` (mục 11 của file đó liệt kê các chỗ lệch spec). Sơ đồ kiến trúc: [`system_architecture.html`](system_architecture.html).

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
| **CIES — ĐÃ IMPLEMENT** | Bootstrap resample train → re-encode từ đầu → xử lý mất cân bằng → train → SHAP trên eval set cố định, lặp N_RUNS lần | `src/explainability/cies.py::run_cies_experiment` |
| **CIES — Stability metric** | Rank-weighted distance (trọng số `1/min(rank_a, rank_b)` theo THỨ HẠNG từng feature, chuẩn hoá tổng = 1; trước đây bị gán theo vị trí cột nên phụ thuộc thứ tự cột) trung bình trên mọi cặp run, `cies_score = 1 − khoảng cách trung bình`; kèm Spearman trung bình để đối chiếu | `src/explainability/cies.py::compute_stability_metric / compute_rank_weighted_distance / shap_to_ranks` |
| **CIES — Feature-level** | Stability cấp từng feature (mean\|SHAP\|, hệ số biến thiên, rank_std) — chỉ cho Sparkov | `src/explainability/cies.py::compute_feature_level_cies` |
| **CIES — chạy cách ly** | Wrap `run_cies_experiment` trong subprocess riêng | `src/explainability/cies.py::run_cies_experiment_isolated` |
| **Trực quan hoá** | Insight dataset (giờ đêm, category×giờ, số tiền, nhịp giao dịch, tính mùa vụ, ULB top-feature), SHAP (beeswarm, so sánh model, đồng thuận thứ hạng, dependence), CIES (heatmap, ổn định thứ hạng theo feature, top-k, Spearman giữa các run, Sparkov vs ULB, trade-off PR-AUC) | `notebooks/06_visualizations.ipynb`; `src/visualization/dataset.py`, `explain.py`; ảnh ở `reports/figures/` |
| **Lưu kết quả CIES** | `results/cies_summary_results.json`, `cies_summary_results_ulb.json` — ghi từng tổ hợp qua khoá file + ghi atomic, an toàn khi nhiều tiến trình cùng ghi | `src/explainability/cies.py::merge_cies_result`; `src/utils/jsonio.py::update_json` |

## Trạng thái chạy thực nghiệm

| Hạng mục | Trạng thái |
|---|---|
| Tune lần 1 (30 trial, vùng hẹp) | LR / XGBoost / CatBoost / ANN xong (PR-AUC CV 0.319 / 0.929 / 0.926 / 0.892), RF xong 0.857 nhưng `max_depth` chạm biên 18. Nhiều tham số chạm biên (CatBoost `depth`, `iterations`; ANN `epochs`; LR `C`) |
| Tune lần 2 (100 trial, vùng đã nới, cùng ngân sách cho mọi model) | Xong: LR 0.319, XGBoost 0.933, CatBoost 0.9271, RF 0.8872 (70 xong / 30 cắt). **ANN còn lại** (chạy nhiều phiên Kaggle với checkpoint SQLite, ~30/100 trial). `best_params.json` hiện là hỗn hợp hai giao thức cho riêng ANN (30 trial cũ), chưa dùng cho benchmark/CIES của ANN |
| **Thiết kế hiện tại (2026-09-26)** | Chia theo thời gian + bỏ cột định danh khách hàng + tune validate theo thời gian (mục "Lệch so với spec" 9–10). `data/processed/` đã tạo lại theo thiết kế này, kèm 4 tập train đã resample (`data/processed/resampled/train_encoded_<kỹ thuật>.parquet`). Kết quả của thiết kế trước: bảng "thiết kế trước" bên dưới; file gốc xem bằng `git show d562d24:results/cies_summary_results.json`, `git show 0b86295:results/cies_summary_results_ulb.json`, `git show 074fc1f:results/model_benchmark_results.csv` |
| Tune (50 trial, 3 fold thời gian, vùng tìm ban đầu) | **Xong 4 model × 2 dataset** (chạy local). PR-AUC CV Sparkov: LR 0,2460 · RF 0,9122 · XGBoost 0,9214 · CatBoost 0,9242; ULB: LR 0,8085 · RF 0,7938 · XGBoost 0,8095 · CatBoost 0,8118. Kiểm tra nới biên: `design_decisions.md` mục 8 |
| Benchmark Sparkov + ULB | **Xong 20/25 mỗi dataset** (4 model × 5 kỹ thuật) — `model_benchmark_results.csv`, `model_benchmark_results_ulb.csv` |
| CIES Sparkov (subsample 100k, N_RUNS=20) + ULB (toàn bộ 227.846 dòng, N_RUNS=20) | **Xong 20/25 mỗi dataset** — `cies_summary_results.json`, `cies_summary_results_ulb.json` |
| ANN: tune + benchmark + CIES, cả 2 dataset | Chờ chạy trên Kaggle (`kaggle_pipeline.ipynb`, `MODELS_SCOPE = ["ann"]`, `RUN_ULB = True`); vùng tìm ANN mới: `design_decisions.md` mục 9 |
| Đối chứng KernelSHAP (`AGENT_SPEC.md` §6.2) | Chưa làm |

## Kết quả benchmark + CIES — thiết kế trước (chia ngẫu nhiên, còn cột định danh; sẽ thay khi chạy lại)

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

**Nhận xét (4/5 model, chưa có ANN, 1 seed):**

1. **Model quyết định CIES nhiều hơn kỹ thuật imbalance — và điều này lặp lại trên cả 2 dataset.** Sparkov: LR 0,84–0,86, 3 model cây/boosting 0,95–0,98. ULB: LR 0,87–0,89, cây/boosting 0,91–0,96. Chênh lệch CIES giữa 5 kỹ thuật trong cùng 1 model nhỏ: Sparkov CatBoost 0,0026, XGBoost 0,0092, RF 0,0141, LR 0,0167; ULB 0,0146–0,0209. Xếp hạng CIES của 20 tổ hợp giữa 2 dataset tương quan Spearman **0,959**, dù mức CIES lệch nhau (TB 0,027): model cây trên ULB thấp hơn Sparkov, LR trên ULB lại cao hơn.
2. **Borderline-SMOTE cho CIES thấp nhất ở 7/8 cặp (model, dataset)**; cặp còn lại (CatBoost, Sparkov) nó hoà với SMOTE-ENN (chênh < 0,00001). Đây là quan sát về **thứ tự**, nhất quán trên cả 2 dataset; về độ lớn thì chênh với kỹ thuật kế tiếp chỉ 0,002–0,01, chưa kiểm được có vượt nhiễu giữa các seed hay không.
3. **`class_weighting` cho PR-AUC cao nhất ở cả 3 model cây/boosting**, và **CIES cao nhất trên Sparkov** ở cả 3 — nhưng trên ULB chỉ còn đúng với CatBoost (RF: thứ 4/5, XGBoost: thứ 3/5), nên phần CIES **không khái quát sang ULB**. F1 của nó thấp nhất ở XGBoost (0,68) và thấp nhì ở CatBoost (0,75, chỉ trên SMOTE-ENN); với RF lại cao nhất. Giải thích khả dĩ: trọng số lớp đẩy xác suất dự đoán của boosting lên cao, nên ở ngưỡng cố định 0,5 sinh nhiều cảnh báo sai hơn (FP của XGBoost 1.608 so với 256–710 ở 4 kỹ thuật kia; CatBoost 1.092 so với 337–965) dù đường cong PR tổng thể vẫn tốt; RF không bị vì bỏ phiếu theo cây. Chưa kiểm chứng bằng cách dò ngưỡng.
4. **CIES và Spearman lệch nhau tuỳ model.** LR lệch nhiều nhất (CIES ~0,85 nhưng Spearman 0,55–0,61 trên Sparkov): top feature ổn định, phần đuôi xáo trộn mạnh. CatBoost lệch nhiều thứ nhì (CIES ~0,955, Spearman 0,87–0,89). RF lệch ít nhất.
5. **Với 3 model cây/boosting, SMOTE-ENN cho PR-AUC thấp nhất và F1 thấp nhất hoặc nhì**, lại tốn thời gian nhất (benchmark CatBoost × SMOTE-ENN trên 1.48M dòng mất ~80 phút, chủ yếu ở bước ENN). **Không đúng với LR**: PR-AUC/F1 của SMOTE-ENN ở LR đứng thứ 3/5, CIES đứng thứ 2/5.
6. **Đổi kỹ thuật imbalance gần như không đổi feature nào được coi là quan trọng.** Độ đồng thuận thứ hạng feature giữa 5 kỹ thuật (cùng model, rank-weighted distance như CIES, trên mean\|SHAP\| trung bình 20 run) thấp nhất 0,931 trên Sparkov và 0,915 trên ULB, trung bình 0,956–0,980 ở mọi model. Lưu ý đây là câu hỏi khác với CIES: CIES đo ổn định **giữa các lần bootstrap** của cùng 1 kỹ thuật; LR có đồng thuận giữa kỹ thuật cao (0,98 trên Sparkov) dù CIES thấp — tức độ bất ổn của LR đến từ dữ liệu, không từ kỹ thuật. Hình: `reports/figures/technique_agreement_*.png`.

**Đây là kết luận sơ bộ**: 1 seed, chưa có ANN, tham số chưa tune lại trên dữ liệu hiện tại — nên diễn giải là quan sát cần kiểm chứng thêm.

## Lệch so với spec và giới hạn

1. **Notebook 04/05 chạy đủ 5×5.** Sparkov subsample **phân tầng 100.000 dòng** (`SUBSAMPLE_N`, ≈521 fraud, giữ tỷ lệ 0,52%) vì train đầy đủ 1.48M dòng quá nặng cho 20 run/tổ hợp; mẫu 10.000 cũ chỉ có ~59 fraud (bootstrap còn ~37 fraud khác nhau) nên CIES sẽ đo nhiễu mẫu nhỏ. Kiểm tra độ nhạy (notebook 04 mục 8, `xgboost × class_weighting` và `random_forest × smote`; đo trên phiên bản dữ liệu trước — 2 file kết quả `results/cies_sensitivity_subsample*.json` không còn trong repo, xem lại bằng `git show a93e1d7^:results/cies_sensitivity_subsample.json`; chạy lại mục 8 nếu cần số trên dữ liệu hiện tại): CIES tăng dần theo cỡ mẫu ở cả hai tổ hợp, không bão hoà rõ ở 100k — XGBoost 0,951/0,957/0,956/0,965, RF+SMOTE 0,941/0,959/0,961/0,969 (10k/30k/50k/100k; 52/156/261/521 fraud); nhiễu giữa các run (std khoảng cách hạng) giảm dần theo cỡ mẫu ở cả hai (RF+SMOTE: 0,0106→0,0034). Hai tổ hợp độc lập nhất quán về xu hướng nên chọn 100k làm cỡ mẫu chuẩn; chưa kiểm tra SMOTE-ENN, CatBoost, ANN.
2. **ULB có tune, benchmark và CIES riêng** (notebook 05): tham số `best_params_ulb.json`, benchmark `model_benchmark_results_ulb.csv`. Test ULB chỉ có ~75 fraud nên PR-AUC trên ULB dao động nhiều hơn Sparkov; mỗi khối validation lúc tune chỉ có 24–36 fraud. Không cần encoding vì feature đã là số, và bỏ cột `Time`.
3. **Tune tách khỏi imbalance**: tham số tối ưu cho trường hợp không xử lý mất cân bằng, dùng chung cho cả 5 kỹ thuật.
4. **CIES của repo là biến thể của CIES gốc** (arXiv:2603.05024): gốc nhiễu hoá đầu vào lúc suy luận, so độ lớn SHAP, chuẩn hoá theo độ lớn giải thích gốc; repo nhiễu hoá dữ liệu huấn luyện và so thứ hạng. Bảng so sánh chi tiết và nguồn cho các quyết định khác: [`literature_support.md`](literature_support.md).
5. **`timeout=None`** cho tune/benchmark: không còn lưới an toàn nếu tiến trình treo thật.
6. **Sparkov là dữ liệu mô phỏng**; nhiều hình dạng "quá sạch" là dấu vết của bộ sinh dữ liệu.
7. **Thời gian resample SMOTE-ENN** tăng ~4× mỗi lần gấp đôi dữ liệu (25k→1.4s, 50k→5.1s, 100k→19.7s ⇒ ≈1.2 giờ cho 1.48M dòng, chưa tính train).
8. **Vùng tìm Optuna giữ như ban đầu** dù vài tham số sát biên (XGBoost Sparkov `n_estimators=550`/600, `learning_rate=0,0137`; RF Sparkov `max_depth=27`/30; LR ULB `C` ở 12% thang log). Đã thử nới biên: PR-AUC CV không đổi (XGBoost Sparkov 0,9214 → 0,9212 với 1850 cây, lr 0,0043; LR ULB 0,8085 → 0,8089) mà mỗi trial chậm ~4×. Chi tiết: [`design_decisions.md`](design_decisions.md) mục 8. RF chưa được kiểm tra bằng vùng nới.
9. **Không đưa cột định danh khách hàng vào model** (`config.CUSTOMER_IDENTITY_COLS`, cùng `cc_num`, `zip`). Gộp lại chúng chỉ ra đúng 1 người: 98,6% cặp (lat, long) và 90,1% giá trị `city_pop` chỉ thuộc 1 thẻ, 92,2% city chỉ có 1 thẻ, `zip` ứng đúng 1 cặp tọa độ, tọa độ cửa hàng luôn trong ~1,4° quanh nhà khách. Model dùng chúng để học thuộc "khách nào từng bị hack" (762/983 thẻ bị hack trong giai đoạn train). `age`, `gender` giữ lại vì là hồ sơ nhân khẩu học (Sparkov sinh fraud theo hồ sơ), không định danh.
10. **Chia train/test theo thời gian** (thay chia ngẫu nhiên theo dòng). Chia theo dòng khiến 100% thẻ có fraud ở test (845/845) cũng có dòng fraud ở train. Đo trên XGBoost (tham số tune cũ, `class_weighting`, PR-AUC; 1 lần chia, 1 seed):

    | Cách chia | Đủ feature | Bỏ nhóm định danh (lat/long, city, state, job) | Bỏ thêm `city_pop` + tọa độ cửa hàng (thiết kế hiện tại) | Bỏ cả `age`, `gender`, `city_pop` |
    |---|---|---|---|---|
    | Ngẫu nhiên theo dòng (trước) | 0,932 | — | — | 0,828 |
    | Theo thẻ (`cc_num`) | 0,914 | — | — | 0,820 |
    | **Theo thời gian (hiện tại)** | **0,240** | 0,881 | **0,882** (pipeline thật: 0,883) | 0,765 |

    Chia theo thẻ bị loại vì không phải cách đánh giá chuẩn trong literature và không áp dụng được cho ULB (không có mã khách hàng). Luật của Fraud Detection Handbook "bỏ thẻ đã bị lộ khỏi tập test" không dùng: trên Sparkov nó bỏ 87% tập test và đẩy tỷ lệ fraud 0,39% → 2,9%, vì bộ mô phỏng không bao giờ khoá thẻ bị hack. Tham chiếu: Wu (arXiv:2607.14686, 2026) dùng đúng tập test gốc của Sparkov, không định danh, đạt AP ≈ 0,93 với feature số tiền tự thiết kế.

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

## Việc tiếp theo

1. Upload 4 file `data/processed/` mới lên Kaggle, chạy `notebooks/kaggle_pipeline.ipynb`: tune cả 5 model (50 trial, validation theo thời gian) → benchmark → CIES Sparkov, đủ 25/25 tổ hợp.
2. Chạy lại CIES ULB (notebook 05, đủ 25/25) sau khi có tham số mới — ULB đã được chia lại theo thời gian.
3. Chạy lại notebook 06, rồi kiểm lại toàn bộ nhận xét ở mục "Kết quả" trên số mới (nhiều nhận xét, vd. LR coi `zip` là quan trọng nhất, gắn với thiết kế trước).
4. (Tuỳ chọn) ablation `SNAP_SYNTHETIC_ONEHOT=True` trên vài tổ hợp.
5. Quyết định có tính thêm CIES đúng công thức gốc (nhiễu đầu vào) làm chỉ số phụ hay không; làm đối chứng KernelSHAP.
6. Chạy nhiều seed cho vài tổ hợp để biết chênh lệch CIES 0,002–0,01 giữa các kỹ thuật (vd. Borderline-SMOTE luôn thấp nhất) có vượt nhiễu hay không.
