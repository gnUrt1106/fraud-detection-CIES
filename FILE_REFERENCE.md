# File Reference — Chức năng từng file trong project

Danh mục theo trạng thái code hiện tại (chỉ liệt kê file nằm trong git, trừ `results/rerun/` và `results/full_test_reference/` là kết quả chạy thực nghiệm). Sơ đồ tổng thể: `reports/system_architecture.html`.

**Hai thiết kế đo CIES cùng tồn tại:** (1) bootstrap — `cies.py` + notebook 04/05, là thiết kế đầu tiên; (2) **chạy lại theo seed** — `rerun_stability.py` + `scripts/run_rerun.py`, là thiết kế chính thức theo yêu cầu giảng viên (`AGENT_SPEC.md` §12). Hai kết quả không so sánh trực tiếp được.

## Root

| File | Chức năng |
|------|-----------|
| `README.md` | Mô tả project, cài đặt, thứ tự chạy thí nghiệm (cả bootstrap lẫn chạy lại theo seed), chạy trên Kaggle, trạng thái |
| `AGENT_SPEC.md` | Đặc tả nghiên cứu: quyết định đã chốt, ràng buộc bắt buộc, danh sách cấm; mục 11 ghi các chỗ code hiện tại lệch spec; **mục 12 là thiết kế chạy lại theo seed (cách đo kết quả chính)** |
| `AGENTS.md` | Entry point cho AI agent — trỏ tới `.agents/rules/` và tóm tắt các bẫy đã gặp (kể cả quy tắc chạy lại theo seed và chính sách mẫu SHAP) |
| `FILE_REFERENCE.md` | File này |
| `requirements.txt` | Thư viện Python (pandas, scikit-learn, imbalanced-learn, xgboost, catboost, torch, shap, optuna, pyarrow, kagglehub, ydata-profiling, pytest...) — phiên bản tối thiểu |
| `requirements-lock.txt` | Đúng phiên bản đã dùng để tạo `results/` (Python 3.12) — dùng khi cần tái lập từng chữ số; `scripts/setup_gpu_env.py` cài từ file này (trừ torch) khi dùng GPU |
| `.gitattributes` | Khai báo file bị loại khỏi **bản nộp bài** (`export-ignore`: tài liệu phát triển, `reports/`, `.agents/`…). Tạo bản nộp: `git archive --format=zip --prefix=fraud-detection-CIES/ -o fraud-detection-CIES.zip HEAD` — chỉ gồm code, notebook, test, `results/` (không kèm checkpoint), README |
| `.env.example` | Mẫu biến môi trường `KAGGLE_USERNAME`, `KAGGLE_KEY` |
| `.gitignore` | Ignore `/data/` (chỉ thư mục ở gốc — không phải `src/data/`), `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.log`, `scratch/`, `catboost_info/`... |

---

## `src/` — Mã nguồn

| File | Chức năng |
|------|-----------|
| `config.py` | **Trung tâm cấu hình**: đường dẫn, `SEED=42`, `N_RUNS=20` (bootstrap), `N_TUNE_TRIALS=50` (số trial Optuna mặc định), `SHAP_BACKGROUND_N=1000` (số dòng nền SHAP của ANN), tên dataset Kaggle, cột target (`is_fraud` / `Class`), `CUSTOMER_IDENTITY_COLS` (không đưa vào model), `ONEHOT_COLS`, `TARGET_ENCODE_COLS`, `ULB_FEATURE_COLS` (bỏ `Time`), `MODEL_NAMES`, `IMBALANCE_TECHNIQUES`, công tắc `SNAP_SYNTHETIC_ONEHOT`. **Chạy lại theo seed:** `RERUN_N_SEEDS=10`, `RESAMPLE_CACHE_RERUN_DIR` (`data/processed/resampled_rerun/`), `RERUN_RESULTS_DIR` (`results/rerun/`), **`RERUN_SHAP_TEST_FRACTION = {"sparkov": 0.20, "ulb": None}`** (chính sách mẫu SHAP: Sparkov 20% test, ULB toàn test) |
| `data/download.py` | Tải Sparkov về `data/raw/` qua `kagglehub` (bỏ qua nếu đã có). Chạy: `python -m src.data.download` |
| `data/preprocess.py` | `preprocess_sparkov`: sắp theo thời gian, thêm `hour`/`day_of_week`/`age` (theo năm của từng giao dịch), bỏ ID và `CUSTOMER_IDENTITY_COLS`. `split_ulb_by_time`: ULB sắp theo `Time`, 20% cuối làm test (Sparkov giữ chia 70/30 của hai file gốc) |
| `data/encoding.py` | One-Hot (`gender, category`) và Stratified K-fold Target Encoding out-of-fold (`merchant`). Prior làm mịn chỉ từ fold train; tham số `groups` (dùng `StratifiedGroupKFold`) giữ bản sao của bootstrap cùng fold; `fixed_categories` giữ số cột one-hot ổn định qua các run; `encode_test` dùng thống kê toàn train và đúng giá trị category gốc lưu trong map (`onehot_values`, kể cả kiểu số/bool). Chạy lại theo seed chỉ fit encoding **một lần** (`train_encoded.parquet`) |
| `imbalance/resamplers.py` | Đúng 5 kỹ thuật: SMOTE, SMOTE-ENN, ADASYN, Borderline-SMOTE (resample) và Class Weighting (trả `class_weights`, không resample). `snap_onehot_groups` / `onehot_groups_from_columns` ép dòng tổng hợp về one-hot hợp lệ khi `SNAP_SYNTHETIC_ONEHOT=True` (mặc định `False`). `apply_imbalance_cached`: benchmark lưu tập train đã resample của mỗi kỹ thuật thành `data/processed/resampled/train_encoded_<kỹ thuật>.parquet` và mọi model dùng chung; metadata chứa dấu vân tay (dữ liệu, seed, SNAP, version imblearn/sklearn), không khớp (hoặc file thiếu cột cần đọc) thì tính lại |
| `imbalance/gpu_enn.py` | **Bước ENN của SMOTE-ENN trên GPU** (PyTorch): `knn_indices` (3 láng giềng theo lô, float64 chính xác), `enn_keep_indices`, `smote_enn_gpu`, `write_resample_cache`. SMOTE vẫn là bản của imbalanced-learn; chỉ được dùng khi `scripts/gpu_smote_enn.py check` xác nhận kết quả trùng khít file CPU. Test: `tests/test_gpu_enn.py` |
| `models/train.py` | `build_model` / `train_model` / `predict_proba` cho 5 model. CatBoost dùng chung feature đã encode (không `cat_features`). Logistic Regression và ANN được bọc kèm `StandardScaler` (trong dict model). XGBoost/CatBoost/ANN tự dùng GPU nếu có (`_has_gpu` qua `nvidia-smi`, cố ý không import torch). `train_and_evaluate_combo` chạy 1 tổ hợp cho notebook 03. Import thư viện model là lazy để tránh nạp torch cùng xgboost |
| `models/tune.py` | Optuna HPO: TPE + MedianPruner, nhận tập train **chưa encode** (`train_raw`, sắp theo thời gian); validation **theo thời gian** (`time_series_folds`: cửa sổ mở rộng, 3 khối liên tiếp ở 1/3 cuối tập train), mỗi fold encode riêng chỉ bằng dòng trước khối validation (`encode_time_folds`, tính 1 lần, dùng lại mọi trial), tối ưu PR-AUC, tune trên dữ liệu **chưa** xử lý imbalance. `tune_all_models` chạy mỗi model trong subprocess, lưu/merge `results/best_params.json` khi model đủ n_trials (model lỗi/timeout không ghi đè tham số đã tune trước đó); hỗ trợ checkpoint SQLite (`checkpoint_dir`) và ngân sách thời gian mềm (`session_budget`) để chia nhiều phiên Kaggle; `load_best_params` để `build_model` tự nạp. Tham số này **đóng băng** khi chạy lại theo seed |
| `evaluation/metrics.py` | PR-AUC (metric chính), ROC-AUC, F1, F2, Precision@Recall (0.5/0.7/0.8), confusion matrix |
| `explainability/shap_utils.py` | `compute_shap` chọn explainer theo model: `LinearExplainer` (LR), `TreeExplainer` (RF, XGBoost, CatBoost), `DeepExplainer` (ANN, tự lùi về `KernelExplainer` nếu lỗi). Luôn trả `(n_mẫu, n_feature)` cho lớp fraud; LR/ANN giải thích trong không gian đã scale và bắt buộc có dữ liệu nền `X_background` (thiếu thì báo lỗi). **`compute_mean_abs_shap`** (dùng cho chạy lại theo seed): tính mean\|SHAP\| theo lô, chọn backend `shap` / `native` / `auto` — `native` = XGBoost `pred_contribs` hoặc CatBoost `ShapValues` (`tree-native`), DeepLIFT theo lô cho ANN (`deep-native`), Random Forest qua XGBoost (`tree-xgb-gpu`/`tree-xgb-cpu`); `native` bị từ chối cho Logistic Regression. Tên explainer thực sự dùng được ghi vào kết quả |
| `explainability/deeplift_native.py` | **DeepLIFT Rescale theo lô cho ANN** (`make_deeplift_fn`): cùng thuật toán `shap.DeepExplainer`, tính trên GPU/CPU bằng torch; chỉ hỗ trợ MLP gồm Linear/ReLU/BatchNorm1d/Dropout với 1 logit, kiến trúc khác báo lỗi thay vì lùi âm thầm. Khớp `shap.DeepExplainer` đến sai số float32 (`tests/test_deeplift_native.py`) |
| `explainability/rf_gpu_shap.py` | **TreeSHAP của Random Forest qua XGBoost** (`rf_to_boosters`, `RFXgbShap`, `make_rf_gpu_shap_fn`): mỗi cây sklearn chuyển thành cây XGBoost (ngưỡng đổi sang float32 liền sau để `x ≤ thr` ⇔ `x < s`; cover = `weighted_n_node_samples`; giá trị lá = P(lớp 1)/số cây), tính `pred_contribs` (GPUTreeShap nếu có GPU, CPU nếu không). Khớp `shap.TreeExplainer` với sai lệch tương đối ~1e-6 (`tests/test_rf_gpu_shap.py`) |
| `explainability/cies.py` | Thuật toán CIES: `shap_to_ranks` (hạng trung bình khi hoà), `compute_rank_weighted_distance` (trọng số theo hạng), `compute_stability_metric` (**công thức dùng chung cho cả hai thiết kế**), `compute_feature_level_cies`, `run_cies_experiment` (bootstrap có `groups` → encode lại → imbalance → train → SHAP trên eval cố định, so với dữ liệu nền cố định ở mọi run — LR: toàn bộ train, ANN: `SHAP_BACKGROUND_N` dòng; loại run có SHAP toàn 0; dưới 2 run thành công thì raise `RuntimeError` để notebook ghi bản ghi lỗi và lần sau chạy lại, không lưu điểm 0), `run_cies_experiment_isolated`, `compute_condition_agreement_matrix` (đồng thuận thứ hạng giữa các điều kiện, vd. 5 kỹ thuật), `merge_cies_result` (ghi từng tổ hợp, an toàn nhiều tiến trình — notebook 04/05 và Kaggle dùng hàm này) |
| `explainability/rerun_stability.py` | **Chạy lại theo seed (thiết kế chính thức).** `run_rerun_combo`: một (model, kỹ thuật) trọn 10 seed trong một lần chạy — đọc resample từ cache, train toàn bộ train, PR-AUC trên toàn test, mean\|SHAP\| trên mẫu SHAP cố định, CIES bằng `compute_stability_metric`; loại run SHAP toàn 0, cần ≥ 2 seed. `build_resample_cache` (cache theo kỹ thuật + seed, không phụ thuộc model, dùng chung cho mọi model). **`resolve_shap_rows` + `stratified_eval_indices`**: mẫu SHAP phân tầng cố định (Sparkov 111.144 dòng/429 fraud; ULB toàn test). `merge_rerun_results`: gộp các file kết quả, báo trùng tổ hợp / thiếu seed / lệch dấu vân tay dữ liệu / **lệch mẫu SHAP**. `pilot_timing`, `data_fingerprint`, `environment_info`, `combo_job`/`cache_job`/`pilot_job` (chạy trong subprocess cách ly). `RERUN_DESIGN="seed_rerun_full_train"` |
| `utils/isolation.py` | `run_isolated`: chạy hàm trong subprocess `spawn` (cách ly torch/xgboost — hai thư viện này không được nạp chung một tiến trình), có timeout với leo thang SIGTERM → SIGKILL; đọc queue lần cuối trước khi coi process con là crash; `daemon=False` để Random Forest song song được |
| `utils/jsonio.py` | `update_json`: đọc–sửa–ghi file JSON kết quả dưới khoá `fcntl`, ghi atomic (file tạm + `os.replace`), báo lỗi nếu JSON hỏng thay vì coi là rỗng. Dùng bởi `tune._merge_write` và `cies.merge_cies_result`. Hai hàm này ghi file theo thứ tự `MODEL_NAMES` × `IMBALANCE_TECHNIQUES` (`cies.combo_order_key`), không theo thứ tự chạy xong |
| `visualization/dataset.py` | Biểu đồ insight dataset: mất cân bằng 2 dataset, giờ, số tiền, khoảng cách (giả thuyết sai), mùa vụ, ULB top-feature và scatter |
| `visualization/explain.py` | Biểu đồ SHAP (beeswarm, so sánh model, đồng thuận thứ hạng, dependence tô màu theo feature khác), CIES (heatmap theo khoảng giá trị thật, thứ hạng kỹ thuật trong từng model, ổn định thứ hạng feature qua các run, CIES theo cỡ mẫu, CIES vs PR-AUC theo dataset, tóm tắt đồng thuận feature giữa các kỹ thuật) và benchmark (PR-AUC lệch theo kỹ thuật, số báo nhầm ở ngưỡng 0,5) |
| `__init__.py` (các thư mục) | Đánh dấu Python package |

---

## `scripts/` — Công cụ dòng lệnh

| File | Chức năng |
|------|-----------|
| `run_rerun.py` | **CLI chạy lại theo seed.** Subcommand: `pilot` (đo thời gian), `cache` (tạo dữ liệu resample theo kỹ thuật + seed), `train` (chạy các tổ hợp, ghi `results/rerun/<name>.json` sau mỗi tổ hợp; chạy lại cùng lệnh thì bỏ qua tổ hợp đã xong), `status` (đã có gì, SHAP trên bao nhiêu dòng), `merge` (gộp các file kết quả → `cies_rerun_summary*.json/.csv`). Cờ chính: `--dataset sparkov|ulb`, `--name`, `--models`, `--techniques`, `--seeds`, `--shap-backend shap\|native\|auto`, `--shap-jobs` (số tiến trình SHAP cho `shap.TreeExplainer`/`DeepExplainer`), `--shap-batch`, `--shap-rows` (mặc định theo chính sách dataset; `all` = toàn test; số nguyên = ép số dòng), `--force-cpu` |
| `gpu_smote_enn.py` | SMOTE-ENN với ENN trên GPU: `bench` (đo tốc độ), `check --seed 0` (đối chiếu file CPU seed 0, phải trùng khít), `cache --seeds ...` (từ chối chạy nếu `check` chưa đạt) |
| `setup_gpu_env.py` | Cài môi trường cho GPU NVIDIA: thư viện trong `requirements-lock.txt` (trừ torch), rồi torch bản CUDA đúng đời driver; lùi về torch CPU nếu không có bản CUDA |

---

## `notebooks/`

| File | Chức năng |
|------|-----------|
| `01_eda.ipynb` | Khám phá dữ liệu Sparkov + data profiling (`reports/profiling/`); xuất biểu đồ ra `reports/figures/` |
| `02_preprocessing.ipynb` | Chia train/test theo thời gian (Sparkov: file gốc; ULB: theo `Time`), bỏ cột định danh khách hàng (`src/data/preprocess.py`), encoding, lưu `data/processed/*.parquet` (giữ thứ tự thời gian) và `encoding_maps.joblib` |
| `kaggle_pipeline.ipynb` | Kaggle: tune (50 trial, `MODELS_SCOPE`) → benchmark → CIES bootstrap cho Sparkov, rồi cho ULB nếu `RUN_ULB` (tham số ULB riêng, cache resample `resampled_ulb/`); với model trong `MODELS_SCOPE` bỏ kết quả clone từ repo và khôi phục tiến độ phiên trước, model ngoài `MODELS_SCOPE` giữ bản trong repo (chạy được nửa local nửa Kaggle), tự resume qua nhiều phiên, ngân sách thời gian chung. Hướng dẫn upload dữ liệu: `KAGGLE_UPLOAD_README.md` |
| `03_train_models.ipynb` | Benchmark 5 model × 5 kỹ thuật → `results/model_benchmark_results.csv` (lưu/resume từng tổ hợp, `timeout=None`) |
| `04_cies_experiment.ipynb` | CIES **bootstrap** trên Sparkov (đủ 5 model × 5 kỹ thuật, train subsample phân tầng `SUBSAMPLE_N=100.000` dòng; mục 8 kiểm tra độ nhạy theo cỡ mẫu) → `results/cies_summary_results.json` |
| `05_cies_experiment_ulb.ipynb` | Toàn bộ ULB: tune (→ `results/best_params_ulb.json`), benchmark (→ `results/model_benchmark_results_ulb.csv`, cache resample `data/processed/resampled_ulb/`) và CIES **bootstrap** đủ 5×5 trên toàn bộ train, `feature_level=False` (→ `results/cies_summary_results_ulb.json`). Benchmark và CIES dùng tham số ULB, không dùng của Sparkov |
| `06_visualizations.ipynb` | Biểu đồ insight dataset, SHAP và CIES → `reports/figures/` |

---

## `results/`, `reports/`, `tests/`

| File/Thư mục | Chức năng |
|--------------|-----------|
| `results/best_params.json` | Hyperparameter đóng băng từ Optuna (Sparkov), dùng lại ở notebook 03/04/05 và chạy lại theo seed |
| `results/best_params_ulb.json` | Như trên cho ULB |
| `results/rerun/main.json` | **Kết quả chạy lại theo seed của Sparkov** (25 tổ hợp; tên file = `--name`, mặc định `main`). Mỗi tổ hợp lưu: CIES và độ đo ổn định, PR-AUC từng seed, mean\|SHAP\| từng seed, thời gian, `config` (mẫu SHAP `n_eval_shap`/`n_eval_fraud`, `data_fingerprint`, thiết bị, phiên bản thư viện, explainer) |
| `results/rerun/ulb/main.json` | Như trên cho ULB |
| `results/rerun/cies_rerun_summary*.json/.csv` | Bảng tổng hợp do `run_rerun.py merge` tạo (chạy sau khi đủ tổ hợp) |
| `results/full_test_reference/` | **Tham chiếu, không đưa vào `merge`**: kết quả SHAP trên toàn bộ test của Sparkov (XGBoost, ANN, Logistic Regression) dùng để đối chiếu với mẫu 20%. Để lẫn vào `results/rerun/` thì `merge` báo "Mẫu SHAP KHÁC nhau" |
| `reports/system_architecture.html` | Sơ đồ kiến trúc hệ thống (3 góc nhìn, bảng tra cứu module, giới hạn đã biết): luồng chạy lại theo seed, `rerun_stability` / `gpu_enn` / `deeplift_native` / `rf_gpu_shap` trong bảng module, mẫu SHAP 20% / toàn test. Ignore khỏi git (giữ local) |
| `reports/pipeline_report.md` | Bảng giải thích từng bước pipeline, trạng thái chạy, kết quả (bootstrap và chạy lại theo seed), quyết định mẫu SHAP kèm bằng chứng, lệch so với spec, nhật ký sửa lỗi |
| `reports/literature_support.md` | Nguồn cho từng quyết định thiết kế, so sánh CIES repo với bài gốc, danh sách chỗ chưa có nguồn |
| `reports/design_decisions.md` | Lý do + số đo cho các quyết định thiết kế hiện tại (chia theo thời gian, bỏ cột định danh khách hàng, `age`, tune theo thời gian, 50 trial), phương án đã loại và câu hỏi dễ gặp — dùng khi báo cáo |
| `reports/study_guide.md` | Hướng dẫn ôn tập để bảo vệ: thứ tự đọc từng file với link tới đúng dòng quan trọng, luồng dữ liệu, 18 câu hỏi phản biện kèm câu trả lời có số liệu, bảng số cần thuộc |
| `reports/kien-thuc-dao-sau-CIES.pdf` | Tài liệu kiến thức đào sâu để ôn bảo vệ (48 trang, font IBM Plex nhúng sẵn): Phần 0 — bản đồ flow; 14 mục khái niệm → cơ chế → trích code → câu hỏi phản biện; mục L–R (2026-10-22): EDA, kiểm tra "feature áp đảo", Cohen's d so với SHAP, độ nhạy cỡ mẫu, giới hạn, câu hỏi có thể gặp, vì sao tune bằng validation. **Mục S–W (2026-10-26): thiết kế chạy lại theo seed (thiết kế chính thức), mẫu SHAP 20% / toàn test và bằng chứng, SHAP native + SMOTE-ENN GPU, kết quả và điều không tái hiện từ bootstrap.** Các mục A–R và 1–14 mô tả thiết kế bootstrap, có chú thích ở các chỗ liên quan |
| `reports/kien-thuc-dao-sau-CIES.html` | Bản nguồn của PDF trên. Sửa file này rồi in lại: Brave/Chrome headless `--print-to-pdf --no-pdf-header-footer` |
| `reports/script-thuyet-trinh.html` / `.pdf` | Kịch bản thuyết trình theo slide. **Chưa cập nhật** — còn các phát biểu về kết quả bootstrap (vd. Borderline-SMOTE kém ổn định nhất) và tập eval 250 dòng; đợi số liệu chạy lại theo seed chốt rồi dựng lại |
| `reports/slide-bao-cao-CIES-24-trang.pdf` | Bộ 24 slide báo cáo. **Chưa cập nhật** theo thiết kế chạy lại theo seed (không có nguồn HTML trong repo) |
| `reports/bao-cao-tien-do.pdf` | Báo cáo tiến độ gửi giảng viên (5 trang): đề tài, dữ liệu, phương pháp, quyết định thiết kế, tiến độ và các mốc, kết quả chính, hạn chế, kế hoạch và câu hỏi xin ý kiến. **Chưa cập nhật theo thiết kế chạy lại theo seed và quyết định mẫu SHAP** |
| `reports/benchmark_literature.md` | Kết quả đã công bố trên Sparkov, **đã kiểm chứng từng nguồn** (Wu 2026: AP 0,930–0,934 cùng tập test gốc, không định danh; FDB 2022: chỉ AUC-ROC; Jemai 2024: không có bảng số), lý do loại các dòng không kiểm chứng được của bản trước, và cách nói về khoảng trống nghiên cứu cho đúng mức |
| `reports/figures/` | 30 biểu đồ PNG: 5 ảnh EDA của notebook 01 (`02_`, `05_`–`07_`, `09_`), các ảnh insight/SHAP/CIES/benchmark của notebook 06, và 2 ảnh `eda_sparkov.png`, `eda_ulb.png` dùng trong mục L, N của PDF kiến thức |
| `reports/profiling/sparkov_profile_report.html` | Báo cáo data profiling tự động |
| `tests/test_pipeline.py` | 49 test (gồm tiền xử lý Sparkov + chia theo thời gian, fold tune theo thời gian và encode riêng từng fold, checkpoint tune, merge kết quả khi nhiều tiến trình ghi song song, file kết quả hỏng không bị ghi đè, độ đồng thuận giữa kỹ thuật): encoding chống rò rỉ (fold prior, index không mặc định, nhóm bootstrap), CatBoost không `cat_features`, 5 kỹ thuật imbalance và one-hot hợp lệ sau resample, công thức CIES (hạng thay vì vị trí cột, hoà hạng), hình dạng SHAP của Tree/ANN, `scale_pos_weight`, ANN (scale, batch 1 mẫu), dữ liệu nền SHAP giống nhau ở mọi run CIES (LR: toàn bộ train; model khác: mẫu cố định; LR dùng mọi dòng nền), EDA không lấy thời gian từ `unix_time`, end-to-end CIES |
| `tests/test_rerun_stability.py` | 18 test của `rerun_stability.py`: `parse_seeds`; mẫu SHAP phân tầng cố định; dấu vân tay dữ liệu; SHAP theo lô khớp SHAP toàn bộ; backend `native` khớp `shap.TreeExplainer` và bị từ chối với model khác; tổ hợp xác định cho CIES = 1 và có cờ; Random Forest × SMOTE với cache theo seed; cần ≥ 2 seed tốt; không tạo cache cho class_weighting; `pilot` đo thời gian; `merge` (sắp xếp, cờ trùng/lệch fingerprint/thiếu seed, bỏ bản ghi lỗi); nạp tham số và đường dẫn ULB; SHAP cây song song khớp tuần tự |
| `tests/test_shap_sample_policy.py` | 12 test chính sách mẫu SHAP: Sparkov mặc định 111.144 dòng/429 fraud (dựng nhãn tổng hợp 555.719/2.145), ULB toàn test, `all`/số nguyên, mẫu tái lập được và trùng với `--shap-rows 111144`, `combo_job` áp chính sách, `merge` cảnh báo khi lệch mẫu và im lặng khi trùng/file cũ, đối số CLI. |
| `tests/test_deeplift_native.py` | 4 test: DeepLIFT theo lô khớp `shap.DeepExplainer` và có tính cộng; nhánh delta nhỏ (dòng trùng dòng nền) giống `shap`; kết quả không phụ thuộc cỡ lô; kiến trúc không hỗ trợ thì báo lỗi, không lùi âm thầm (chạy trong subprocess vì torch) |
| `tests/test_rf_gpu_shap.py` | 5 test: ngưỡng float32 tương đương `x ≤ thr`; SHAP khớp `shap.TreeExplainer` với nhiều cỡ nhóm cây/lô; chia nhóm cây không đổi kết quả; các cấu hình rừng khác; từ chối lớp không nhị phân (cần xgboost, tự bỏ qua nếu thiếu) |
| `tests/test_rf_native_backend.py` | 4 test: backend `native` của Random Forest khớp backend `shap` và bỏ qua `n_jobs`; `auto` giữ nguyên hành vi cũ cho RF; `--force-cpu` cho explainer CPU; `native` vẫn bị từ chối với model không hỗ trợ (cần xgboost) |
| `tests/test_gpu_enn.py` | 4 test: ENN trùng `imblearn.EditedNearestNeighbours` và không phụ thuộc cỡ lô; SMOTE-ENN GPU trùng `SMOTEENN`; feature rời rạc có hoà khoảng cách vẫn trùng; cache ghi ra được `apply_imbalance_cached` nhận là cache hit |
| `tests/test_deep_parallel.py` | SHAP ANN (`shap.DeepExplainer`) chạy song song nhiều tiến trình cho kết quả như đơn tiến trình |
| `tests/test_setup_gpu_env.py` | 4 test của `setup_gpu_env.py`: tách lock bỏ đúng torch và giữ chú thích; lock thật có ghim torch; kế hoạch CUDA theo đời driver; `--dry-run` in kế hoạch không cài gì (cần `requirements-lock.txt`) |

---

## `.agents/rules/` — Hướng dẫn cho agent

| File | Chức năng |
|------|-----------|
| `project-overview.md` | Tổng quan, vai trò hai dataset, cấu trúc thư mục, tiến độ |
| `coding-standards.md` | Quy chuẩn code Python, data, ML và các bẫy đã gặp |
| `workflow.md` | Quy trình: task mới, sửa bug, thêm feature |
