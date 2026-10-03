# Hướng dẫn ôn tập source code — để trả lời khi bảo vệ

Mọi số liệu dưới đây lấy từ `results/` và `reports/` (thiết kế hiện tại, cập nhật 2026-10-03). Link dạng `file#Lxx` mở đúng dòng
trên GitHub; trong IDE thì mở file rồi `Ctrl/Cmd+G` → số dòng.

**Cách ôn (3 vòng):**
1. **Vòng 1 — bức tranh (30 phút):** học thuộc mục 1 và mục 5 (bảng số). Đủ để trả lời "đề tài làm gì, kết quả là gì".
2. **Vòng 2 — luồng dữ liệu (1–2 giờ):** đọc mục 2, mở từng file theo thứ tự, chỉ đọc những dòng được chỉ.
3. **Vòng 3 — phản biện (1 giờ):** tự trả lời mục 4 mà không nhìn đáp án.

---

## 1. Bức tranh 1 phút (học thuộc)

- **Câu hỏi nghiên cứu:** kỹ thuật xử lý mất cân bằng ảnh hưởng thế nào tới **độ ổn định của giải thích SHAP** (đo bằng CIES), bên
  cạnh hiệu năng (PR-AUC)?
- **Thí nghiệm:** 5 model (LR, RF, XGBoost, CatBoost, ANN) × 5 kỹ thuật (SMOTE, SMOTE-ENN, ADASYN, Borderline-SMOTE, class
  weighting) × 2 dataset (Sparkov — chính, mô phỏng; ULB — phụ, thật) = 50 tổ hợp, mỗi tổ hợp CIES qua 20 lần bootstrap.
- **CIES** = 1 − khoảng cách trung bình giữa các **bảng xếp hạng feature** (theo mean|SHAP|) của 20 lần bootstrap, có trọng số
  1/hạng (feature quan trọng bị đảo hạng thì phạt nặng hơn). 1 = giải thích giống hệt nhau ở mọi lần.
- **Thiết kế chống ảo tưởng:** chia train/test **theo thời gian**; **bỏ cột định danh khách hàng**; tune bằng validation **theo thời
  gian**, encode riêng từng fold.
- **Kết quả chính:**
  1. **Borderline-SMOTE cho giải thích kém ổn định nhất ở 9/10 cặp model × dataset.**
  2. Model quyết định mức ổn định nhiều hơn kỹ thuật (Sparkov: cây 0,945–0,973; LR 0,916–0,939; ANN 0,849–0,921).
  3. PR-AUC cao **không** đảm bảo giải thích ổn định (Sparkov: Spearman −0,15 khi bỏ LR).
  4. Đổi kỹ thuật chỉ làm PR-AUC chênh ≤ 3,4 điểm % trong cùng model; class weighting báo nhầm nhiều nhất ở ngưỡng 0,5.

---

## 2. Thứ tự đọc từng file

Mỗi file: **vai trò → dòng cần đọc → cần nhớ → câu dễ bị hỏi**.

### 2.1 [`src/config.py`](../src/config.py) — mọi hằng số (đọc hết, 128 dòng)
- [`SEED = 42`, `N_RUNS = 20`](../src/config.py#L10) — tái lập được, 20 lần bootstrap.
- [`CUSTOMER_IDENTITY_COLS`](../src/config.py#L79) và chú thích ngay trên (dòng 74–78): vì sao bỏ lat/long, city, state, job,
  city_pop, merch_lat/long.
- [`ONEHOT_COLS`, `TARGET_ENCODE_COLS`](../src/config.py#L84): gender, category → one-hot; merchant → target encoding.
- [`ULB_FEATURE_COLS`](../src/config.py#L92): V1..V28 + Amount, **không có Time**.
- [`SNAP_SYNTHETIC_ONEHOT = False`](../src/config.py#L115) và chú thích dòng 104–114.
- [`IMBALANCE_TECHNIQUES`](../src/config.py#L118): đúng 5 kỹ thuật, không thêm bớt.
- **Hỏi:** *"Sao Sparkov chỉ còn 7 feature?"* → amt, category, merchant, hour, day_of_week, age, gender; các cột còn lại là mã/định
  danh (1 cột hoặc tổ hợp chỉ ra đúng 1 khách) → model học thuộc "khách nào từng bị hack" (mục 4, câu 2).

### 2.2 [`src/data/preprocess.py`](../src/data/preprocess.py) — làm sạch + chia theo thời gian (49 dòng)
- [Sắp theo thời gian](../src/data/preprocess.py#L28) — tune chia fold theo thứ tự dòng nên bắt buộc.
- [`age` = năm giao dịch − năm sinh](../src/data/preprocess.py#L33) — dữ liệu trải 2019–2020, không lấy năm cố định.
- [`split_ulb_by_time`](../src/data/preprocess.py#L39): ULB sắp theo `Time`, 20% cuối làm test ([dòng 45](../src/data/preprocess.py#L45)).
- Sparkov dùng sẵn `fraudTrain.csv` (01/2019 → 21/06/2020) / `fraudTest.csv` (→ 31/12/2020): train kết thúc trước test.
- **Nhớ số:** Sparkov train 1.296.675 dòng (7.506 fraud, 0,579%), test 555.719 (2.145 fraud, 0,386%); ULB train 227.846 (417 fraud),
  test 56.961 (75 fraud).

### 2.3 [`src/data/encoding.py`](../src/data/encoding.py) — encoding (240 dòng)
- [`stratified_kfold_target_encode`](../src/data/encoding.py#L30): target encoding **out-of-fold** trên train (mỗi dòng được encode
  bằng thống kê của các fold khác → nhãn của chính nó không lọt vào). Làm mịn: category hiếm bị kéo về trung bình.
  [Prior lấy từ fold train](../src/data/encoding.py#L80), không phải toàn train.
- [`full_mapping`](../src/data/encoding.py#L92): test dùng thống kê **toàn bộ train**, không bao giờ fit trên test.
- Tham số `groups` (dòng 52–55): bootstrap có dòng trùng → bản sao cùng 1 fold (StratifiedGroupKFold), tránh rò nhãn.
- [`encode_test`](../src/data/encoding.py#L182): merchant chưa thấy → trung bình toàn cục; one-hot giữ đúng cột của train.
- **Hỏi:** *"Target encoding có rò rỉ không?"* → trên train: out-of-fold; test: chỉ thống kê train; lúc tune: encode riêng từng fold
  thời gian (mục 2.6). Hạn chế đã ghi: trong train, fold xáo trộn nên 1 dòng có thể dùng nhãn của dòng train xảy ra sau — chỉ ảnh
  hưởng feature lúc train, không ảnh hưởng tập test (`pipeline_report.md`, giới hạn 14).

### 2.4 [`src/imbalance/resamplers.py`](../src/imbalance/resamplers.py) — 5 kỹ thuật (215 dòng)
- [`get_resampler`](../src/imbalance/resamplers.py#L34): SMOTE, SMOTEENN, ADASYN, BorderlineSMOTE — tham số mặc định imbalanced-learn
  (k=5; borderline-1, m=10; ENN k=3 làm sạch cả 2 lớp), cân bằng về 1:1.
- [`get_class_weights`](../src/imbalance/resamplers.py#L65): class weighting = `compute_class_weight("balanced")`
  ([dòng 78](../src/imbalance/resamplers.py#L78)) → trọng số lớp fraud ≈ 86, lớp hợp lệ ≈ 0,5 trên Sparkov. **Không** sinh dòng mới.
- [`apply_imbalance`](../src/imbalance/resamplers.py#L115): chỉ áp lên **train đã encode**, không bao giờ lên test.
- [`apply_imbalance_cached`](../src/imbalance/resamplers.py#L168) + [dấu vân tay](../src/imbalance/resamplers.py#L158): benchmark lưu
  tập đã resample (SMOTE-ENN trên toàn train mất 145 phút); dấu vân tay gồm dữ liệu, kỹ thuật, seed, version thư viện → khác là
  tính lại, không dùng nhầm.
- **Hỏi:** *"SMOTE nội suy cột one-hot ra giá trị 0,7/0,3 có sai không?"* → mục 4, câu 7.

### 2.5 [`src/models/train.py`](../src/models/train.py) — tạo/train/dự đoán model (494 dòng)
- [`build_model`](../src/models/train.py#L131): đọc từng nhánh model. Class weighting đưa vào model:
  LR/RF `class_weight` ([166](../src/models/train.py#L166), [185](../src/models/train.py#L185)); XGBoost
  [`scale_pos_weight` = w₁/w₀](../src/models/train.py#L211); CatBoost [`class_weights`](../src/models/train.py#L237); ANN
  [`pos_weight` trong loss](../src/models/train.py#L358).
- LR có [`StandardScaler`](../src/models/train.py#L175); ANN có [scaler riêng](../src/models/train.py#L351) — cây không cần scale.
- [`_train_ann`](../src/models/train.py#L329): mạng 21→128→64→32→1, BatchNorm, Dropout, Adam; cắt batch từ tensor mỗi epoch.
- [`train_and_evaluate_combo`](../src/models/train.py#L440): 1 tổ hợp benchmark = resample → build → train → predict test → metrics.
- **Hỏi:** *"Sao LR trên Sparkov kém (PR-AUC 0,12)?"* → tín hiệu `hour` có dạng "cao ban đêm" (22h–3h: 1,4–2,9% fraud, giờ khác
  ~0,1%) mà 1 hệ số tuyến tính không mô tả được; trên ULB (feature PCA gần tuyến tính) LR đạt 0,69–0,70.

### 2.6 [`src/models/tune.py`](../src/models/tune.py) — Optuna (499 dòng)
- [`sample_hyperparameters`](../src/models/tune.py#L37): vùng tìm từng model (thuộc tên tham số chính: LR `C`; RF `n_estimators`,
  `max_depth`, `min_samples_*`; XGBoost/CatBoost số cây, độ sâu, `learning_rate`; ANN `lr`, `dropout`, `epochs`, `batch_size`).
- [`time_series_folds`](../src/models/tune.py#L94): 1/3 cuối train chia 3 khối validation liên tiếp, mỗi fold train trên **mọi dòng
  trước** khối đó ([dòng 103](../src/models/tune.py#L103)).
- [`encode_time_folds`](../src/models/tune.py#L107): encode riêng từng fold chỉ bằng dữ liệu trước khối validation.
- [`tune_model`](../src/models/tune.py#L143): objective = PR-AUC trung bình 3 fold ([215](../src/models/tune.py#L215));
  [MedianPruner](../src/models/tune.py#L229) cắt trial kém sau fold đầu; [TPE seed 42](../src/models/tune.py#L243); checkpoint SQLite.
- Tune trên dữ liệu **chưa xử lý mất cân bằng**, tham số dùng chung cho 5 kỹ thuật (mục 4, câu 4).
- **Nhớ số (PR-AUC CV):** Sparkov LR 0,246 · RF 0,912 · XGBoost 0,921 · CatBoost 0,924 · ANN 0,919; ULB LR 0,809 · RF 0,794 ·
  XGBoost 0,809 · CatBoost 0,812 · ANN 0,803.

### 2.7 [`src/evaluation/metrics.py`](../src/evaluation/metrics.py) — chỉ số (78 dòng)
- [PR-AUC = `average_precision_score`](../src/evaluation/metrics.py#L50) — chỉ số chính, không phụ thuộc ngưỡng.
- [F1, F2, FP… ở ngưỡng 0,5](../src/evaluation/metrics.py#L45) — chỉ tham khảo (mục 4, câu 10).

### 2.8 [`src/explainability/shap_utils.py`](../src/explainability/shap_utils.py) — SHAP (226 dòng)
- [`EXPLAINER_MAP`](../src/explainability/shap_utils.py#L23): LR → LinearExplainer; RF/XGBoost/CatBoost → TreeExplainer (chính xác);
  ANN → DeepExplainer (xấp xỉ, tất định; Kernel chỉ dự phòng).
- [`compute_shap`](../src/explainability/shap_utils.py#L36): LR và ANN **bắt buộc** truyền dữ liệu nền (`X_background`), thiếu là báo lỗi
  ngay — không âm thầm lấy tập eval làm nền (mốc sai mà không ai biết). Model cây không cần nền.
- [`_compute_shap_linear`](../src/explainability/shap_utils.py#L82): LR giải thích trong **không gian đã scale** (cùng scaler lúc train);
  dùng **mọi dòng** nền (`max_samples=len(nền)` — mặc định của shap chỉ giữ 100 dòng, mốc sẽ đổi theo mẫu con).
- [`_compute_shap_tree`](../src/explainability/shap_utils.py#L109): chuẩn hoá output về (mẫu × feature) cho lớp fraud.
- [`_compute_shap_deep`](../src/explainability/shap_utils.py#L186): ANN; nếu lỗi thì lùi Kernel và **ghi lại** explainer đã dùng.
- **Hỏi:** *"Sao ANN dùng Deep chứ không Kernel?"* → Kernel tự có nhiễu lấy mẫu (2 lần chạy cùng model chỉ khớp Spearman ~0,85) làm
  CIES thấp giả; Deep tất định. Cả 200 run ANN đều dùng Deep.

### 2.9 [`src/explainability/cies.py`](../src/explainability/cies.py) — **trái tim của đề tài** (đọc kỹ nhất)
- [`compute_rank_weighted_distance`](../src/explainability/cies.py#L43): [trọng số `1/min(hạng_a, hạng_b)`](../src/explainability/cies.py#L69),
  chuẩn hoá tổng = 1; khoảng cách = Σ trọng số × |chênh hạng| / (số feature − 1).
- [`compute_stability_metric`](../src/explainability/cies.py#L147): so mọi cặp trong 20 run (190 cặp);
  [`cies_score = 1 − khoảng cách TB`](../src/explainability/cies.py#L199); kèm Spearman trung bình để đối chiếu.
- [`run_cies_experiment`](../src/explainability/cies.py#L279) — mỗi run (thuộc 5 bước):
  0. Trước vòng lặp: [chọn dữ liệu nền SHAP cố định](../src/explainability/cies.py#L333) từ tập train gốc (LR: toàn bộ; ANN: 1000 dòng);
  1. [Bootstrap train](../src/explainability/cies.py#L356) (lấy có hoàn lại, seed = số thứ tự run — seed này cũng đặt cho resampler và model);
  2. [Encode lại từ đầu](../src/explainability/cies.py#L361) (có `groups` chống rò nhãn);
  3. [Áp kỹ thuật imbalance](../src/explainability/cies.py#L378);
  4. Train model (tham số cố định đã tune);
  5. [SHAP trên tập đánh giá **cố định**](../src/explainability/cies.py#L395) (250 mẫu test: 50 fraud + 200 hợp lệ), so với dữ liệu nền cố định;
     [run có SHAP toàn 0 bị loại](../src/explainability/cies.py#L402) (tránh CIES = 1 giả).
- [`compute_condition_agreement_matrix`](../src/explainability/cies.py#L117): đồng thuận **giữa các kỹ thuật** (khác CIES: CIES là
  giữa các run của cùng 1 kỹ thuật).
- **Hỏi:** *"CIES của em có giống bài gốc không?"* → mục 4, câu 8.

### 2.10 [`src/utils/isolation.py`](../src/utils/isolation.py), [`jsonio.py`](../src/utils/jsonio.py) — hạ tầng (lướt)
- [`run_isolated`](../src/utils/isolation.py#L74): mỗi tổ hợp chạy trong tiến trình con riêng — torch và xgboost cùng 1 tiến trình có
  thể xung đột OpenMP (treo/segfault).
- [`update_json`](../src/utils/jsonio.py#L28): ghi file kết quả có khoá + ghi atomic — nhiều tiến trình chạy song song không mất kết quả.

### 2.11 Notebook (đọc phần chữ + kết quả, không cần thuộc code)
| Notebook | Vai trò |
|---|---|
| `01_eda` | Khám phá Sparkov: mức mất cân bằng, phân bố số tiền, tỷ lệ fraud theo thứ/category/giới tính/tuổi |
| `02_preprocessing` | Gọi preprocess + encoding, lưu `data/processed/` |
| `03_train_models` | Benchmark 25 tổ hợp Sparkov |
| `04_cies_experiment` | CIES Sparkov (mẫu phân tầng 100.000 dòng); mục 8: kiểm tra độ nhạy cỡ mẫu |
| `05_cies_experiment_ulb` | ULB: tune + benchmark + CIES (tham số riêng của ULB) |
| `06_visualizations` | Mọi biểu đồ + đoạn "Đọc kết quả" — **nguồn tốt nhất để ôn insight** |
| `kaggle_pipeline` | Chạy ANN trên GPU Kaggle (tune → benchmark → CIES, Sparkov rồi ULB) |

### 2.12 Tài liệu lập luận (đọc để trả lời "vì sao")
- `reports/design_decisions.md` — lý do + số đo cho từng quyết định (chia thời gian, bỏ định danh, tune, 50 trial, vùng tìm, SMOTE chuẩn, ANN, dữ liệu nền SHAP).
- `reports/pipeline_report.md` — mục "Kết quả" (2 bảng + 9 nhận xét) và "Lệch so với spec" (20 giới hạn).
- `reports/literature_support.md`, `reports/benchmark_literature.md` — nguồn tài liệu đã kiểm chứng; cách nói về khoảng trống nghiên cứu.

---

## 3. Luồng dữ liệu (vẽ lại được trên bảng)

```
fraudTrain.csv / fraudTest.csv ──preprocess (sắp thời gian, thêm hour/day/age, bỏ định danh)──► train_raw / test_raw (7 feature)
                                    │
     ┌──────────────────────────────┼──────────────────────────────────────────────┐
     ▼ TUNE                          ▼ BENCHMARK                                     ▼ CIES
 3 fold thời gian, encode        encode train (OOF) + test (thống kê train)     mẫu 100k → 20 lần: bootstrap → encode lại →
 riêng từng fold, KHÔNG          → resample CHỈ train → train model →          imbalance → train → SHAP trên 250 mẫu test
 imbalance → Optuna 50 trial     PR-AUC/F1 trên test gốc                         cố định → xếp hạng feature → CIES
     └──► best_params.json ──────────┴──────────────────────────────────────────────┘
```

---

## 4. Câu hỏi phản biện thường gặp — và câu trả lời

1. **"Sao chia theo thời gian mà không chia ngẫu nhiên?"** — Thực tế model dự đoán giao dịch tương lai; chia ngẫu nhiên đặt các dòng
   của cùng 1 đợt hack thẻ vào cả train lẫn test (100% thẻ có fraud ở test cũng có fraud ở train). Chuẩn của Fraud Detection
   Handbook và Dal Pozzolo et al. (2018). Số đo (XGBoost): chia ngẫu nhiên 0,932 → chia thời gian + bỏ định danh 0,882.
2. **"Sao bỏ lat/long, city, job…? Mất thông tin à?"** — Chúng định danh khách (98,6% cặp lat/long chỉ thuộc 1 thẻ). Giữ lại + chia
   thời gian cho PR-AUC chỉ **0,240** (model học thuộc khách bị hack, không khái quát); bỏ đi → 0,882. Khoảng cách khách–cửa hàng
   của fraud và hợp lệ gần như trùng (76,3 so với 76,1 km) → bỏ không mất tín hiệu thật.
3. **"Sao dùng PR-AUC?"** — Mất cân bằng nặng (0,4–0,6% fraud): ROC-AUC bị phóng đại bởi rất nhiều TN (FDB báo 0,995–0,998, không phân
   biệt được model); PR-AUC tập trung vào lớp fraud (Saito & Rehmsmeier 2015), không phụ thuộc ngưỡng.
4. **"Sao tune trên dữ liệu chưa resample rồi dùng chung cho 5 kỹ thuật?"** — Để chênh lệch CIES/PR-AUC giữa các kỹ thuật là do kỹ
   thuật, không do tham số khác nhau. Nhược điểm (nói thẳng): tham số không tối ưu riêng cho từng kỹ thuật; chưa có nguồn tài liệu.
5. **"50 trial có đủ không?"** — Cả 10 lượt tune đã cách kết quả cuối ≤ 0,005 PR-AUC từ trial 13; nửa sau chỉ thêm ≤ 0,0026
   (`design_decisions.md` mục 5). Cùng 50 trial cho mọi model = cùng ngân sách (công bằng theo Bischl et al. 2023).
6. **"Tham số chạm biên sao không nới?"** — Đã thử nới (XGBoost tới 2000 cây): PR-AUC CV 0,9214 → 0,9212, mỗi trial chậm ~4× → giữ.
   ANN chạm biên dropout/epochs có xu hướng nhưng mức tăng ≤ 0,002, dưới dao động (mục 8–9).
7. **"SMOTE sinh one-hot 0,7/0,3 — không hợp lệ?"** — Đúng, là SMOTE chuẩn như literature. Dòng tổng hợp chỉ nằm trong train; mọi đánh
   giá và giải thích tính trên giao dịch **thật**. Bản ép one-hot hợp lệ làm PR-AUC giảm mạnh (XGBoost + SMOTE 0,887 → 0,766) còn CIES gần như không đổi (0,957 → 0,959) — số đo
   trên **thiết kế trước**; nếu bị hỏi số hiện tại thì nói rõ chưa đo lại.
8. **"CIES có phải công thức bài gốc?"** — Không hoàn toàn: mượn ý tưởng khoảng cách có trọng số theo hạng của Văduva et al. (2026),
   nhưng nguồn nhiễu là **dữ liệu huấn luyện** (bootstrap) thay vì nhiễu đầu vào, vì câu hỏi là ảnh hưởng của kỹ thuật imbalance.
9. **"Sao Borderline-SMOTE kém ổn định nhất?"** — Kết quả: hạng 5 ở 9/10 cặp. Cách hiểu (chưa kiểm chứng): nó chỉ sinh mẫu quanh các
   fraud nằm ở ranh giới, mà tập ranh giới đổi theo từng mẫu bootstrap → dữ liệu tổng hợp khác nhau nhiều giữa các run. Chênh lệch nhỏ
   (0,0005–0,011 trên Sparkov). Sai số jackknife của 1 điểm CIES 0,002–0,006: ở cả 9 cặp, nó thua kỹ thuật tốt nhất hơn 2 lần sai số, nhưng
   4/9 cặp chưa tách được khỏi kỹ thuật kế tiếp; mới 1 seed.
10. **"Class weighting F1 thấp mà PR-AUC vẫn tốt?"** — Trọng số đẩy xác suất lên → ở ngưỡng 0,5 báo nhầm nhiều (XGBoost 4.606 so với
    1.110–1.915); PR-AUC không phụ thuộc ngưỡng nên ít bị ảnh hưởng. Dò ngưỡng sẽ khắc phục (chưa làm).
11. **"Sao không so CIES giữa Sparkov và ULB?"** — CIES tăng theo số fraud trong mẫu bootstrap (RF × SMOTE: 0,894 ở 58 fraud → 0,961 ở
    579); Sparkov 579 fraud, ULB 417 → so **mức** bị lẫn cỡ mẫu; chỉ so **thứ hạng** (Borderline-SMOTE vẫn cuối ở cả 2).
12. **"Sao lấy mẫu 100.000 dòng cho CIES Sparkov?"** — 20 run × 25 tổ hợp trên 1,3 triệu dòng quá nặng; 100k giữ đúng tỷ lệ fraud (579
    fraud). Đã kiểm tra độ nhạy; mốc 100k tính lại trùng từng chữ số.
13. **"PR-AUC cao thì giải thích có ổn định hơn không?"** — Sparkov: không (Spearman −0,15 bỏ LR; CatBoost × SMOTE PR-AUC cao nhất nhưng
    CIES trung bình; ANN PR-AUC ngang CatBoost nhưng CIES thấp nhất). ULB: cùng chiều (0,50) nhưng test chỉ 75 fraud.
14. **"Kết quả so với nghiên cứu khác?"** — Wu (2026): cùng tập test gốc, không định danh, XGBoost AP 0,930–0,934 (có thêm feature
    velocity theo thẻ); luận văn CatBoost 0,887 không dùng velocity. Không so với bài chia ngẫu nhiên (con số bị thổi phồng).
15. **"Hạn chế lớn nhất?"** — 1 seed; ULB ít fraud (75 ở test, 24–36 mỗi khối validation); tune tách khỏi imbalance; vài tham số sát
    biên; Sparkov là dữ liệu mô phỏng (gần như mọi khách đều bị lộ thẻ); PR-AUC (train trên toàn bộ train) và CIES (bootstrap từ mẫu
    100k) đến từ 2 cách train khác nhau; ANN chỉ tái lập được về mặt thống kê giữa các máy; `age` tính theo năm (lệch +1 tuổi ở
    44,6% dòng chưa tới sinh nhật — giới hạn 19); mốc SHAP của model cây đổi theo từng run (giới hạn 20, câu 16).
16. **"LR/ANN kém ổn định là do model hay do cách tính SHAP?"** — LinearExplainer và DeepExplainer đo SHAP so với 1 tập dữ liệu nền
    (LR: `hệ số × (x − trung bình nền)`). Nền giống nhau ở mọi run và đủ lớn: LR dùng toàn bộ tập train, ANN dùng 1000 dòng (với 100 dòng,
    chỉ đổi dòng nào làm nền đã làm CIES ANN lệch tới 0,031). Model cây thì khác: TreeExplainer lấy mốc từ dữ liệu train của chính từng
    run, nên **so mức CIES giữa cây và LR/ANN không cùng điều kiện mốc** (RF với mốc cố định: +0,029) — so giữa kỹ thuật trong cùng
    model thì không bị ảnh hưởng.
17. **"CIES đo nhiễu gì?"** — Dao động khi **train lại từ đầu**: seed của run đổi mẫu bootstrap, phần ngẫu nhiên của SMOTE… và của
    model (khởi tạo ANN, lấy mẫu của RF/XGBoost). Cố định: tập eval, tham số, kỹ thuật, dữ liệu nền SHAP. 5 kỹ thuật dùng chung 20 mẫu
    bootstrap nên so sánh giữa kỹ thuật là so theo cặp.
18. **"CIES 0,91 là cao hay thấp?"** — Thứ hạng hoàn toàn ngẫu nhiên cho CIES ≈ 0,56 (không phải 0); chỉ đổi chỗ top-1 với top-2 cho
    0,976 (21 feature). CIES 0,91 nghĩa là trung bình các run đảo hạng nhiều hơn 1 lần đổi chỗ top-2.

---

## 5. Bảng số cần thuộc

| Nội dung | Số |
|---|---|
| Sparkov train / test | 1.296.675 (0,579% fraud) / 555.719 (2.145 fraud) |
| ULB train / test | 227.846 (417 fraud) / 56.961 (75 fraud) |
| Feature sau encode | Sparkov 21, ULB 29 |
| PR-AUC test tốt nhất | Sparkov CatBoost × SMOTE **0,887**; ULB RF × SMOTE **0,819** |
| CIES Sparkov | cây 0,945–0,973 · LR 0,916–0,939 · ANN 0,849–0,921; cao nhất CatBoost × class weighting **0,973** |
| Kết quả nhất quán nhất | Borderline-SMOTE CIES thấp nhất **9/10** cặp |
| Phương sai CIES do model / kỹ thuật | Sparkov **85% / 7%**; ULB 75% / 18% |
| Mốc đọc CIES (21 feature) | ngẫu nhiên ≈ **0,56**; đổi chỗ top-1/top-2 = **0,976** |
| Tác động kỹ thuật lên PR-AUC | ≤ **3,4** điểm % trong cùng model |
| Chia ngẫu nhiên vs thời gian (XGBoost) | 0,932 vs 0,882; giữ định danh + chia thời gian: 0,240 |
| Độ nhạy cỡ mẫu (RF × SMOTE) | 0,894 (58 fraud) → 0,961 (579 fraud) |
| Tham chiếu ngoài | Wu 2026: AP 0,930–0,934 (cùng test, không định danh) |
