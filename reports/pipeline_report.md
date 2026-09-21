---
title: "Sơ đồ Pipeline Thực Tế — Fraud Detection CIES"
---

# Sơ đồ Pipeline Thực Tế — Fraud Detection CIES

> Sơ đồ này phản ánh **đúng code hiện tại** trong `notebooks/` và `src/` (sau audit), không phải bản thiết kế lý thuyết trong `AGENT_SPEC.md`. Các điểm mà code hiện tại đang chạy khác/hẹp hơn so với spec gốc được đánh dấu ⚠️ trực tiếp trên sơ đồ.

Sơ đồ kiến trúc hệ thống (3 góc nhìn: kiến trúc phân tầng, hợp đồng dữ liệu giữa các bước, một lượt CIES; kèm bảng tra cứu module và giới hạn đã biết): [`system_architecture.html`](system_architecture.html) — mở thẳng bằng trình duyệt bất kỳ, không cần công cụ ngoài.

---

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

---

## ⚠️ Ghi chú quan trọng — sai lệch giữa code hiện tại và spec đầy đủ

1. **`notebooks/04_cies_experiment.ipynb` (cell 4) hiện chỉ chạy SUBSET**, không phải full 5×5:
   - Model: chỉ `logistic_regression`, `random_forest`, `xgboost` — **CatBoost và ANN chưa nằm trong vòng lặp thực chạy**.
   - Kỹ thuật imbalance: chỉ `class_weighting`, `smote`, `smote_enn` — **ADASYN và Borderline-SMOTE chưa được chạy**.
   - Train set bị subsample xuống 10,000 dòng trước khi bootstrap (`train_raw.sample(n=min(10000, len(train_raw)))`).
   - Bản thân module `src/explainability/cies.py` hỗ trợ đầy đủ 5 model × 5 kỹ thuật — giới hạn này nằm ở notebook (có thể là chủ đích để chạy debug/local nhanh trước khi lên Kaggle/Colab chạy full scale, đúng như comment "hoặc toàn bộ 25 tổ hợp khi chạy trên GPU server/Kaggle").
2. **Dataset phụ ULB**: CIES đã chạy được (`notebooks/05_cies_experiment_ulb.ipynb`, subset `logistic_regression`/`random_forest`/`xgboost` × `class_weighting`/`smote`/`smote_enn`, `feature_level=False`) để so `cies_score` với Sparkov trên cùng tổ hợp — nhưng **chưa có bước benchmark PR-AUC/F1/... riêng cho ULB** (chỉ notebook 03 chạy Sparkov). ULB không cần bước encoding categorical (toàn bộ feature V1..V28 + Amount đã là số). Bootstrap chạy trên **TOÀN BỘ 227,845 dòng train** (không subsample như Sparkov phải làm ở dataset 1.48M dòng) — ULB đủ nhỏ để train trực tiếp trong thời gian hợp lý (đã đo: RandomForest ~27-50s/run, XGBoost <5s/run; tổng ~60-70 phút cho 9 tổ hợp × N_RUNS=20), loại bỏ hoàn toàn câu hỏi về tính công bằng của việc subsample/điều chỉnh tỷ lệ class.
3. **Hai bug được phát hiện nhờ vẽ biểu đồ (đã sửa)**: (a) `build_model` tính `scale_pos_weight` của XGBoost **ngược** (`w0/w1 ≈ 0.0017` thay vì `≈ 577`), làm class_weighting *giảm* trọng số fraud — PR-AUC XGBoost + class_weighting trên ULB tụt 0.877 → 0.704, ảnh hưởng cả benchmark lẫn CIES của tổ hợp này; (b) model hằng số cho SHAP toàn 0 → mọi thứ hạng hoà nhau → CIES giả = 1.0, nay run như vậy bị loại (`run_cies_experiment`). **Mọi kết quả benchmark/CIES của `xgboost × class_weighting` chạy trước bản sửa đều không hợp lệ và cần chạy lại.**
4. **Bug thứ ba trong chính metric CIES (đã sửa)**: `compute_rank_weighted_distance` gán trọng số `1/i` theo *chỉ số cột* `i` thay vì theo thứ hạng feature, nên cùng 1 cú đổi chỗ top-2 cho khoảng cách 0.122 hoặc 0.030 tuỳ 2 feature nằm ở cột nào. **Mọi `cies_score` tính trước bản sửa đều phụ thuộc thứ tự cột và cần chạy lại.** Công thức mới là cách hiện thực hoá ý định ghi trong docstring; spec chỉ ghi "theo công thức paper CIES gốc" nên cần đối chiếu với paper gốc.
5. **Đợt rà soát toàn bộ mã nguồn — các lỗi đã sửa** (mỗi lỗi có test hồi quy):
   - **`.gitignore` nuốt `src/data/`**: dòng `data/` khớp cả `src/data/`, nên `encoding.py`/`download.py` chưa từng vào git — clone mới không import được CIES (`No module named 'src.data'`). Nay chỉ ignore `/data/` ở gốc.
   - **ANN không scale input**: feature thô cỡ 1e9 (`unix_time`), 1e6 (`city_pop`) làm ANN gần như không học — PR-AUC 0.16 so với 0.74 khi scale (đo trên Sparkov). Nhận định trước đó "BatchNorm nên không cần scale" là sai. Nay ANN dùng `StandardScaler` (lưu trong `input_scaler`, áp cho cả predict lẫn SHAP). **Mọi số ANN tính trước bản sửa không hợp lệ.**
   - **ANN crash khi `len(train) % batch_size == 1`** (`BatchNorm1d` không nhận batch 1 mẫu) — có thể làm hỏng cả 1 lần Optuna sau ~150 lần fit; nay `drop_last`.
   - **`encode_train` sinh NaN/lệch hàng** khi `df_train` có index không mặc định (19% giá trị NaN trong thử nghiệm); các notebook hiện tại luôn `reset_index` nên kết quả cũ không bị ảnh hưởng, nhưng gọi trực tiếp thì dính. Nay gán theo vị trí.
   - **ULB đưa nhầm cột `Time` vào feature** (notebook 05 dùng mọi cột trừ `Class`, trái `config.ULB_FEATURE_COLS`); nay chỉ V1..V28 + Amount.
   - **Ghép PR-AUC với CIES sai khoá** (notebook 04 và `plot_cies_vs_prauc`): cột kỹ thuật ở benchmark tên `imbalance_technique` nên merge rơi về chỉ theo `model` → tích Descartes.
   - **Đường lỗi của CIES thiếu khoá `mean_spearman`** → `KeyError` làm sập cả vòng lặp; notebook 04 nay cũng lưu từng tổ hợp và bỏ qua tổ hợp lỗi như notebook 05.
   - **Ô `pip install` trên Kaggle** thiếu ngoặc kép quanh `optuna>=3.4.0...` nên shell coi `>` là chuyển hướng file (bỏ mất ràng buộc version, để lại file rác `=3.4.0`).

   **Các điểm còn lại sau rà soát — xử lý như sau:**
   - *Bootstrap có dòng trùng làm rò nhãn trong target encoding của CIES* — **đã sửa**: `encode_train(..., groups=)` dùng `StratifiedGroupKFold` để mọi bản sao của 1 dòng luôn cùng fold (chỉ áp dụng cho CIES; đường Optuna/benchmark vẫn dùng `StratifiedKFold` như spec).
   - *Feature SHAP = 0 hoà hạng xếp theo thứ tự cột* — **đã sửa**: `shap_to_ranks` dùng hạng trung bình cho các feature hoà. Lưu ý: feature bằng 0 ở *mọi* run vẫn giữ hạng cố định và không đóng góp khoảng cách; cách sửa chỉ loại chênh lệch hạng giả do thứ tự cột khi tập feature bằng 0 khác nhau giữa các run.
   - *SMOTE-ENN vượt timeout ở notebook 03* — **đã sửa**: đo thực tế thời gian resample tăng ~4× mỗi lần gấp đôi dữ liệu (25k→1.4s, 50k→5.1s, 100k→19.7s ⇒ ≈1.2 giờ cho 1.48M dòng); notebook 03 nay dùng `timeout=None` và lưu/resume từng tổ hợp. Các kỹ thuật khác (SMOTE, ADASYN, Borderline-SMOTE) gần như tức thì.
   - *`global_mean` trong target encoding tính trên toàn bộ train* — **đã sửa** (theo yêu cầu, có cập nhật `AGENT_SPEC.md` §3.2): prior làm mịn tính chỉ từ fold train. Đo trước khi sửa: giá trị encode lệch ≤2.5e-6 trên thang ~0.005, PR-AUC XGBoost 0.9054 → 0.9044 (1 seed, cỡ nhiễu). Sau khi sinh lại `train_encoded.parquet`, chỉ 3 cột target-encoded đổi (tối đa 5.6e-7), `test_encoded` không đổi ⇒ kết quả Optuna đã có vẫn hợp lệ.
   - *SMOTE sinh giá trị phân số cho cột one-hot* — **đã sửa** nhưng có đánh đổi lớn: `apply_imbalance(..., onehot_groups=)` ép mỗi nhóm one-hot của dòng tổng hợp về 1 category hợp lệ (argmax, tất định, áp đồng nhất cho SMOTE/SMOTE-ENN/ADASYN/Borderline-SMOTE). Công tắc: `config.SNAP_SYNTHETIC_ONEHOT` (True = hợp lệ, False = SMOTE thuần). Đo trên Sparkov 300k: SMOTE gốc có 90% dòng tổng hợp mang ≥1 cột phân số (81% "bật" >1 category); nhưng PR-AUC khi ép hợp lệ **tụt mạnh** — XGBoost: không resample 0.905, SMOTE gốc 0.887, snap argmax 0.766, lấy mẫu theo trọng số 0.736, SMOTENC 0.733; Random Forest: 0.815 / 0.719 / 0.573 / 0.547 / 0.487. Ba cách hợp lệ độc lập cho kết quả gần nhau nên đây không phải lỗi của 1 cách làm; giả thuyết: giá trị phân số vô tình cô lập dòng tổng hợp khỏi vùng dữ liệu thật. Hệ quả cho nghiên cứu: với cách hợp lệ, các kỹ thuật resample sẽ trông kém hơn rõ rệt so với trước (và so với `class_weighting`).
   - **Cần chạy lại vì các thay đổi này:** notebook 03 (benchmark), 04 và 05 (CIES) — toàn bộ kết quả cũ (nếu có) không còn hợp lệ. Optuna: chỉ còn ANN chưa tune.
