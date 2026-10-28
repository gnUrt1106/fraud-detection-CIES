# Fraud Detection — CIES

Đồ án nghiên cứu **độ ổn định của giải thích mô hình phát hiện gian lận** dưới các kỹ thuật xử lý mất cân bằng dữ liệu.

Chỉ số đo là **CIES (Credibility Index via Explanation Stability)**: huấn luyện lại mô hình nhiều lần, tính SHAP trên một tập đánh giá
cố định, rồi đo thứ hạng các feature dao động bao nhiêu giữa các lần chạy. Thứ hạng càng ít đổi thì giải thích càng đáng tin
(CIES = 1: giống hệt nhau ở mọi lần).

**Thiết kế chính thức (theo yêu cầu giảng viên): chạy lại 10 lần chỉ đổi seed (0–9) trên toàn bộ train, không bootstrap** — xem mục
"Chạy lại theo seed" bên dưới. Thiết kế đầu tiên (bootstrap 20 lần, notebook 04/05) vẫn còn trong repo và trong
`results/cies_summary_results*.json` như kết quả tham khảo; hai thiết kế không so sánh trực tiếp được.

- **5 mô hình**: Logistic Regression, Random Forest, XGBoost, CatBoost, ANN (PyTorch).
- **5 kỹ thuật imbalance**: SMOTE, SMOTE-ENN, ADASYN, Borderline-SMOTE, Class Weighting.
- **2 dataset** đối xử ngang nhau: Sparkov và ULB Credit Card Fraud.

## Dữ liệu

| | Sparkov | ULB |
|---|---|---|
| Nguồn | [kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) | [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Chia train/test | **Theo thời gian**, đúng 2 file gốc: `fraudTrain.csv` (01/2019 → 06/2020, 1.296.675 dòng) / `fraudTest.csv` (06/2020 → 12/2020, 555.719 dòng) | **Theo thời gian**: sắp theo `Time`, 20% cuối làm test (227.846 / 56.961 dòng) |
| Tỷ lệ fraud | 0,58% train / 0,39% test | 0,18% train / 0,13% test |
| Tập đánh giá SHAP (chạy lại theo seed) | Mẫu **cố định, phân tầng theo nhãn = 20% tập test**: 111.144 dòng, 429/2.145 fraud | **Toàn bộ test**: 56.961 dòng, 75 fraud |
| Feature | 7 feature: `amt`, `category`, `merchant`, `hour`, `day_of_week`, `age`, `gender` → 21 cột sau encoding | V1..V28 (PCA) + `Amount` (bỏ `Time`) |
| Ghi chú | Dữ liệu **mô phỏng** ([Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation)) | Dữ liệu thật, ẩn danh |

Sparkov: các cột định danh khách hàng (`lat`, `long`, `city`, `state`, `job`, `city_pop`, `merch_lat`, `merch_long`, `cc_num`, `zip`)
**không** đưa vào model — gộp lại chúng chỉ ra đúng 1 khách hàng và model dùng chúng để học thuộc "khách nào từng bị hack" (xem
`CUSTOMER_IDENTITY_COLS` trong `src/config.py`). Dữ liệu không kèm trong bản nộp — tải theo mục "Chạy thí nghiệm".

## Cấu trúc

```
├── notebooks/
│   ├── 01_eda.ipynb                 # Khám phá dữ liệu Sparkov
│   ├── 02_preprocessing.ipynb       # Chia theo thời gian, bỏ cột định danh, encoding, tải + chia ULB
│   ├── 03_train_models.ipynb        # Benchmark Sparkov: 5 model × 5 kỹ thuật
│   ├── 04_cies_experiment.ipynb     # CIES Sparkov (+ kiểm tra độ nhạy theo cỡ mẫu)
│   ├── 05_cies_experiment_ulb.ipynb # ULB: tune + benchmark + CIES
│   ├── 06_visualizations.ipynb      # Biểu đồ insight, SHAP, CIES, hiệu năng — kèm nhận xét
│   ├── kaggle_pipeline.ipynb        # Chạy trên Kaggle GPU (dùng cho ANN): tune → benchmark → CIES
│   └── KAGGLE_UPLOAD_README.md      # Hướng dẫn chạy notebook Kaggle
├── src/
│   ├── config.py             # Đường dẫn, hằng số, danh sách model/kỹ thuật, cột feature, cấu hình chạy lại theo seed
│   ├── data/                 # download.py, preprocess.py (chia theo thời gian), encoding.py
│   ├── imbalance/            # resamplers.py — 5 kỹ thuật; gpu_enn.py — bước ENN của SMOTE-ENN trên GPU (trùng khít bản CPU)
│   ├── models/               # train.py (5 model), tune.py (Optuna, validation theo thời gian)
│   ├── evaluation/           # metrics.py (PR-AUC là chỉ số chính)
│   ├── explainability/       # shap_utils.py, cies.py (thuật toán CIES), rerun_stability.py (chạy lại theo seed),
│   │                         # deeplift_native.py (DeepLIFT theo lô cho ANN), rf_gpu_shap.py (TreeSHAP của Random Forest qua XGBoost)
│   ├── visualization/        # dataset.py, explain.py
│   └── utils/                # isolation.py (mỗi tổ hợp chạy trong tiến trình riêng), jsonio.py
├── scripts/                  # run_rerun.py (CLI chạy lại theo seed), gpu_smote_enn.py (SMOTE-ENN trên GPU), setup_gpu_env.py (môi trường GPU)
├── tests/                    # test_pipeline.py; test của phần chạy lại theo seed, GPU và backend native
├── results/                  # Kết quả thí nghiệm (xem bên dưới)
└── requirements.txt  requirements-lock.txt  .env.example
```

## Kết quả (`results/`)

| File | Nội dung |
|---|---|
| `best_params.json`, `best_params_ulb.json` | Tham số tune (Optuna, 50 trial, 3 fold theo thời gian) cho từng model, riêng mỗi dataset |
| `model_benchmark_results.csv`, `model_benchmark_results_ulb.csv` | PR-AUC, ROC-AUC, F1, F2, Precision@Recall, TP/FP/TN/FN trên tập test — 25 tổ hợp mỗi dataset |
| `cies_summary_results.json`, `cies_summary_results_ulb.json` | CIES, Spearman và mean\|SHAP\| của từng lần bootstrap (20 lần) — 25 tổ hợp mỗi dataset |
| `cies_sensitivity_subsample.json` | CIES theo cỡ mẫu bootstrap (kiểm tra độ nhạy) |
| `rerun/*.json`, `rerun/ulb/*.json` | Chạy lại theo seed (`main.json`; `--name` đổi tên file); mỗi tổ hợp gồm CIES, PR-AUC theo seed, mean\|SHAP\| từng seed, thời gian, cấu hình (mẫu SHAP, explainer, dấu vân tay dữ liệu, thiết bị, phiên bản thư viện) |
| `cies_rerun_summary[_ulb].json/.csv` | Bảng tổng hợp do `scripts/run_rerun.py merge` tạo từ các file trong `rerun/` |
| `full_test_reference/` | Kết quả Sparkov chạy SHAP trên toàn bộ test (XGBoost, ANN, Logistic Regression) — chỉ để đối chiếu với mẫu 20%, **không** đưa vào `merge` |

**Kết quả chính của thiết kế bootstrap** (1 seed; chi tiết và biểu đồ trong notebook 06):
- **Borderline-SMOTE cho giải thích kém ổn định nhất** ở 9/10 cặp model × dataset.
- **Mô hình quyết định mức ổn định nhiều hơn kỹ thuật imbalance** (Sparkov: model cây CIES 0,945–0,973, LR 0,916–0,939, ANN
  0,849–0,921; model giải thích 85% phương sai CIES, kỹ thuật 7%).
- **PR-AUC cao không đảm bảo giải thích ổn định** trên Sparkov (Spearman giữa PR-AUC và CIES −0,15, không tính LR).
- Hiệu năng tốt nhất: Sparkov CatBoost × SMOTE PR-AUC **0,887**; ULB Random Forest × SMOTE **0,819**. Trong cùng 1 model, đổi kỹ thuật
  imbalance chỉ làm PR-AUC chênh ≤ 3,4 điểm %.
- CIES tăng theo cỡ mẫu bootstrap nên chỉ so CIES **trong cùng dataset**; giữa 2 dataset chỉ so thứ hạng các kỹ thuật.
- Kết luận "Borderline-SMOTE kém ổn định nhất" **không tái hiện** khi chạy lại 10 seed trên toàn bộ train (ULB: Borderline-SMOTE không thấp nhất ở cặp nào; Sparkov: thấp nhất ở LR và ANN, không ở XGBoost và CatBoost). Thứ hạng kỹ thuật phụ thuộc dataset; xem `reports/pipeline_report.md` mục "Chế độ chạy lại theo seed".

**Kết quả sơ bộ của thiết kế chạy lại theo seed** (10 seed, toàn bộ train; Sparkov 22/25 tổ hợp — còn 3 tổ hợp Random Forest —, ULB 25/25;
bảng đầy đủ ở `reports/pipeline_report.md`):
- CIES theo model, khoảng giữa 5 kỹ thuật. Sparkov: Logistic Regression 0,996–1,000 · XGBoost 0,988–0,995 · CatBoost 0,965–0,981 · ANN 0,886–0,954
  (Random Forest mới có 2/5 tổ hợp: SMOTE 0,970; Class Weighting 0,985). ULB: Logistic Regression 0,996–1,000 · Random Forest 0,971–0,988 ·
  XGBoost 0,958–0,982 · CatBoost 0,958–0,987 · ANN 0,846–0,939.
- ANN có CIES thấp nhất ở cả hai dataset; Logistic Regression × Class Weighting tất định theo seed nên CIES = 1,000 (hệ quả của thiết kế, không phải lỗi).
- Đây là mức dao động giữa các lần chạy lại, chưa phải kết luận về kỹ thuật imbalance tốt nhất: chênh giữa các kỹ thuật trong cùng model nhỏ và chưa kiểm định thống kê.

## Cài đặt

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # macOS/Linux
pip install -r requirements-lock.txt
```

`requirements-lock.txt` ghim đúng phiên bản đã dùng để tạo `results/` (Python 3.12) — cần thiết để tái lập từng chữ số, vì kết quả
resample, model và SHAP phụ thuộc phiên bản thư viện. `requirements.txt` chỉ ghi phiên bản tối thiểu để code chạy được.

Cần Kaggle credentials để tải dữ liệu qua `kagglehub`: đặt `kaggle.json` (tạo tại [Kaggle Settings](https://www.kaggle.com/settings),
mục API) vào `~/.kaggle/`, hoặc `cp .env.example .env` rồi điền `KAGGLE_USERNAME`, `KAGGLE_KEY`.

## Chạy thí nghiệm

Chạy theo thứ tự (notebook mở bằng `jupyter lab` từ thư mục gốc):

1. **Tải Sparkov**: `python -m src.data.download` → `data/raw/`.
2. **`02_preprocessing.ipynb`**: sinh `data/processed/*.parquet` (và tải + chia ULB).
3. **Tune Sparkov**: `src.models.tune.tune_all_models(train_raw)` (local; mặc định 50 trial × 3 fold thời gian như thí nghiệm) hoặc
   `kaggle_pipeline.ipynb` (Kaggle GPU, nên dùng cho ANN) → `results/best_params.json`. Kết quả tune đã kèm trong `results/`.
4. **`03_train_models.ipynb`**, **`04_cies_experiment.ipynb`**: benchmark và CIES Sparkov.
5. **`05_cies_experiment_ulb.ipynb`**: tune, benchmark và CIES cho ULB (tham số riêng của ULB).
6. **`06_visualizations.ipynb`**: biểu đồ và nhận xét.

Mỗi tổ hợp được lưu ngay khi xong, chạy lại sẽ bỏ qua phần đã có (kết quả đã kèm trong `results/`, nên notebook 03–06 chạy lại chỉ vẽ lại
biểu đồ). Chạy test: `python -m pytest tests -q` (gồm test của phần chạy lại theo seed, GPU và backend native; một số test tự bỏ qua nếu thiếu xgboost hoặc GPU).

### Chạy lại theo seed (thiết kế chính thức)

Đo độ ổn định qua các **lần chạy lại** trên toàn bộ train (dữ liệu và phân phối giữ nguyên, chỉ đổi seed 0–9; không bootstrap), PR-AUC và CIES
lấy từ cùng các lần chạy, tham số Optuna giữ nguyên (`results/best_params.json`, `best_params_ulb.json`). Mã: `src/explainability/rerun_stability.py`,
CLI: `scripts/run_rerun.py`. Sparkov và ULB được đối xử **ngang nhau** (cùng thiết kế, cùng công thức, cùng 10 seed, cùng 5 model × 5 kỹ thuật).

**Tập đánh giá SHAP** (giảng viên ban đầu yêu cầu toàn bộ test; việc dùng mẫu 20% cho Sparkov **chưa được giảng viên xác nhận**):

| | Sparkov | ULB |
|---|---|---|
| SHAP tính trên | mẫu **cố định, phân tầng theo nhãn = 20% tập test**: 111.144/555.719 dòng, 429/2.145 fraud (≈ 6% cả dataset) | **toàn bộ test**: 56.961 dòng, 75 fraud |
| PR-AUC tính trên | toàn bộ test | toàn bộ test |

Mẫu 20% là mặc định của Sparkov (`config.RERUN_SHAP_TEST_FRACTION`), cùng một tập dòng cho mọi model, kỹ thuật và seed (`stratified_eval_indices`,
cố định theo `SEED`); `--shap-rows all` ép toàn bộ test, `--shap-rows N` ép đúng N dòng. Lý do: SHAP của Random Forest/CatBoost trên 555 nghìn dòng × 50 lần
train quá tốn. Bằng chứng mẫu đủ (15 tổ hợp XGBoost, ANN, Logistic Regression có cả bản test đầy đủ lẫn bản 20%): PR-AUC giống hệt, |ΔCIES| ≤ 0,0022,
thứ hạng feature trung bình tương quan Spearman ≥ 0,9987, top-3 feature giống nhau ở cả 15; CatBoost và Random Forest chưa có đối chiếu với test đầy đủ.
Bài gốc CIES (Văduva et al., arXiv:2603.05024) tính trên 100 dòng test ngẫu nhiên — điều này ủng hộ việc lấy mẫu test, không ủng hộ riêng con số 111.144.

```bash
python scripts/run_rerun.py pilot --force-cpu                               # đo giờ trước
python scripts/run_rerun.py cache --techniques smote_enn --seeds 0-9        # resample theo (kỹ thuật, seed) → data/processed/resampled_rerun/
python scripts/run_rerun.py train --models xgboost catboost ann random_forest --shap-backend native   # Sparkov: mẫu SHAP 20% là mặc định
python scripts/run_rerun.py train --models logistic_regression --force-cpu                           # LR: không dùng --shap-backend native
python scripts/run_rerun.py status
python scripts/run_rerun.py merge                                           # → results/cies_rerun_summary.json / .csv
```

Thêm `--dataset ulb` vào mọi lệnh cho ULB (cache ở `data/processed/resampled_rerun/ulb/`, kết quả ở `results/rerun/ulb/`, tổng hợp
`results/cies_rerun_summary_ulb.json/.csv`; SHAP trên toàn bộ test).

Cách tính SHAP (`--shap-backend`; tên explainer được ghi trong từng kết quả):
- `shap` là thư viện `shap` (Linear/Tree/DeepExplainer). `native` là cùng thuật toán nhưng nhanh hơn và dùng được GPU: XGBoost `pred_contribs` / CatBoost
  `ShapValues` (`tree-native`); DeepLIFT Rescale theo lô cho ANN (`src/explainability/deeplift_native.py`, `deep-native`, khớp `shap.DeepExplainer` đến sai số
  float32); Random Forest chuyển từng cây sang XGBoost rồi dùng TreeSHAP chính xác của XGBoost (`src/explainability/rf_gpu_shap.py`, `tree-xgb-gpu` hoặc
  `tree-xgb-cpu`, sai lệch tương đối ~1e-6 so với `shap.TreeExplainer` trên Random Forest thật). `auto` (mặc định) = native cho XGBoost/CatBoost, `shap` cho phần còn lại;
  muốn Random Forest đi đường nhanh thì đặt `--shap-backend native`.
- `--shap-jobs N` chia SHAP của Random Forest (đường `shap`) cho N tiến trình — TreeSHAP của `shap` chỉ chạy đơn luồng; kết quả không đổi.

**SMOTE-ENN trên GPU.** Bước ENN (tìm 3 láng giềng gần nhất của ~2,6 triệu dòng sau SMOTE) chiếm gần hết thời gian resample: ~145 phút/seed trên CPU.
`scripts/gpu_smote_enn.py` chạy bước này trên GPU, tìm láng giềng chính xác (không xấp xỉ, float64), ~44 phút/seed trên GPU; `check --seed 0` đã xác nhận
**trùng khít** file CPU seed 0 (2.557.107 dòng, 1.288.931 fraud) và `cache` từ chối chạy nếu `check` chưa đạt.

Mỗi (model, kỹ thuật) phải chạy trọn 10 seed trong **một** lần chạy, cùng thiết bị; mọi tổ hợp dùng cùng file `*_encoded.parquet` (`merge` báo nếu dấu vân tay dữ liệu lệch) và
cùng mẫu SHAP (`merge` báo nếu cỡ mẫu SHAP lệch giữa các tổ hợp — vd. quên chuyển file chạy test đầy đủ ra khỏi `results/rerun/`). CIES ở chế độ này
**không so sánh trực tiếp** với CIES bootstrap: nguồn dao động chỉ còn tính ngẫu nhiên của thuật toán. Model tất định theo seed (LR + class_weighting)
cho CIES = 1 — đó là hệ quả của thiết kế, không phải lỗi.

**Tái lập.** `results/` được tạo trên macOS arm64 (CPU), trừ tune và benchmark của ANN chạy trên GPU Kaggle. Cùng cấu hình đó và
`requirements-lock.txt`, chạy lại cho kết quả trùng từng chữ số (đã kiểm: dữ liệu sau tiền xử lý, benchmark, CIES của LR, CatBoost, ANN).
Trên phần cứng khác, phép tính số thực có thể khác thứ tự nên số lệch ở chữ số nhỏ; riêng ANN thì SHAP từng run lệch rõ hơn nhưng CIES gần
như trùng (đã đo GPU Kaggle so với CPU với cấu hình dữ liệu nền SHAP cũ: 0,843 / 0,838 và 0,906 / 0,907; chưa đo lại với nền
hiện tại — số trong `results/` là của nền hiện tại).
Kết quả chạy lại theo seed (`results/rerun/`) tạo trên Windows; GPU và CPU khác thứ tự phép tính số thực nên chỉ tái lập đến chữ số nhỏ, và vì vậy mỗi tổ hợp chạy trọn 10 seed trong cùng một lần chạy. Thiết bị và phiên bản thư viện được ghi trong `config` của từng kết quả.

**Lưu ý kỹ thuật:** torch (ANN) và xgboost không được nạp chung 1 tiến trình (xung đột OpenMP) — mọi model/tổ hợp chạy qua
`src/utils/isolation.py::run_isolated`.

## Ghi chú phương pháp

CIES trong đồ án là **biến thể** của CIES gốc (Văduva et al., 2026, arXiv:2603.05024): bài gốc nhiễu hoá đầu vào lúc suy luận; đồ án
nhiễu hoá dữ liệu huấn luyện (bootstrap + train lại) và đo trên thứ hạng feature.

**Mẫu SHAP.** Với Sparkov, SHAP tính trên mẫu cố định 20% tập test (111.144 dòng) thay vì toàn bộ test (mục "Chạy lại theo seed"). Bài gốc CIES chỉ dùng 100 dòng test ngẫu nhiên cho mỗi tổ hợp.

## Trích dẫn dữ liệu

```
@misc{sparkov2020,
  author = {Brandon Harris}, title = {Sparkov Data Generation}, year = {2020},
  publisher = {GitHub}, url = {https://github.com/namebrandon/Sparkov_Data_Generation}
}
@misc{kartik2020fraud,
  author = {Kartik Shenoy}, title = {Fraud Detection Dataset}, year = {2020},
  publisher = {Kaggle}, url = {https://www.kaggle.com/datasets/kartik2112/fraud-detection}
}
@misc{ulb2018,
  author = {{Machine Learning Group, ULB}}, title = {Credit Card Fraud Detection}, year = {2018},
  publisher = {Kaggle}, url = {https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud}
}
```
