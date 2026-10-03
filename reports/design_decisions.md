# Lý do các quyết định thiết kế (dùng khi báo cáo / bảo vệ)

Mỗi mục: **vấn đề → bằng chứng đo được → nguồn → quyết định → phương án đã loại → câu hỏi dễ gặp.**
Mọi con số dưới đây đo trên dữ liệu thật của repo (không ước lượng), trừ khi ghi khác.

Thiết kế hiện tại (chốt 2026-09-26, `AGENT_SPEC.md` §8 và §11):

| | Trước | Hiện tại |
|---|---|---|
| Chia train/test | Gộp 2 file Sparkov, chia ngẫu nhiên 80/20 theo dòng | **Theo thời gian**: `fraudTrain.csv` (2019-01 → 2020-06-21) / `fraudTest.csv` (2020-06-21 → 2020-12-31). ULB: 20% cuối theo `Time` |
| Feature Sparkov | 7 feature bên phải + tọa độ nhà (`lat`, `long`), `city`, `state`, `job`, `city_pop`, `zip`, tọa độ cửa hàng | 7 feature: `amt`, `category`, `merchant`, `hour`, `day_of_week`, `age`, `gender` |
| `age` | `2020 − năm sinh` | Năm của từng giao dịch − năm sinh |
| Validation khi tune | 5-fold xáo trộn, trên dữ liệu đã encode sẵn | 3 fold theo thời gian (cửa sổ mở rộng), mỗi fold encode riêng |
| Số trial Optuna | 100 | 50 (cùng vùng tìm cho mọi model và 2 dataset; đã thử nới biên — mục 8) |
| Tham số ULB | Dùng tham số tune trên Sparkov | **Tune riêng trên ULB** (mục 10) |
| Vùng tìm ANN | batch 256–1024, 10–100 epoch | batch 1024–4096, 5–50 epoch (mục 9) |
| So CIES giữa 2 dataset | So trực tiếp mức CIES | **Chỉ so thứ hạng**; mức CIES phụ thuộc cỡ mẫu (mục 11) |
| Dữ liệu nền SHAP (LR, ANN) trong CIES | 100 dòng đầu của mẫu bootstrap của từng run | **Cố định ở mọi run: LR = toàn bộ train, ANN = 1000 dòng** (mục 12) |

> **Lưu ý khi trình bày kết quả:** kết quả cuối của thiết kế hiện tại nằm ở `reports/pipeline_report.md`, mục "Kết quả". Các số PR-AUC dùng để so sánh lựa chọn trong mục 1–3 của tài liệu này là của **XGBoost + `class_weighting`, tham số tune cũ, 1 lần chia, 1 seed** — dùng để so sánh các lựa chọn với nhau, không phải kết quả cuối.

---

## 1. Chia train/test theo thời gian thay vì ngẫu nhiên

**Vấn đề.** Gian lận thẻ đến **theo đợt**: 1 thẻ bị lộ sinh ra nhiều giao dịch gian lận liên tiếp. Trong Sparkov, 9.651 dòng fraud chỉ đến từ 977 đợt lộ thẻ — **89,9% dòng fraud là tiếp nối của 1 đợt đã có** (notebook 06). Chia ngẫu nhiên theo dòng thì các dòng của cùng 1 đợt rơi vào cả train lẫn test.

**Bằng chứng.**
- Với cách chia cũ, **100% thẻ có fraud ở test (845/845) cũng có dòng fraud ở train** — model được "xem trước" đúng đợt hack mà nó bị chấm điểm.
- Với cách chia theo thời gian gốc của Sparkov, chỉ **4/218** thẻ có fraud ở test từng có fraud ở train (đợt hack ngắn nên hiếm khi vắt qua mốc chia).

**Nguồn.**
- *Fraud Detection Handbook* (Le Borgne và cộng sự, nhóm ULB/Worldline — nhóm tạo ra dataset ULB) — **đã đọc 2 trang liên quan**: dữ liệu được chia **theo khối thời gian**, train là quá khứ, test là tương lai, giữa hai phần có **khoảng trễ** (delay period) vì nhãn fraud chỉ biết sau khi khách khiếu nại/điều tra; và **bỏ khỏi tập test các thẻ đã biết bị lộ**.
- Dal Pozzolo và cộng sự, *Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy*, IEEE TNNLS 2018 — **chỉ đọc tóm tắt**: mô hình hoá bài toán theo điều kiện vận hành thật (luồng giao dịch theo thời gian, concept drift, độ trễ xác minh), dùng cửa sổ trượt theo thời gian.
- Hayat & Magnier, *Data Leakage and Deceptive Performance: A Critical Examination of Credit Card Fraud Detection Methodologies*, Mathematics 13(16):2563, 2025 — **chỉ đọc tóm tắt**: nêu "validation theo thời gian không đầy đủ" là 1 trong 4 lỗi phương pháp phổ biến làm kết quả đẹp giả.
- Bản thân Sparkov được phát hành với cách chia theo thời gian (`fraudTrain.csv` → `fraudTest.csv`). Wu (arXiv:2607.14686, 2026, **preprint**) — **đã đọc bản HTML**: dùng đúng tập test gốc này (555.719 dòng, 2.145 fraud), không đưa định danh vào model, đạt AP ≈ 0,93 với feature số tiền tự thiết kế.

**Quyết định.** Chia theo thời gian, dùng đúng 2 file gốc. ULB: sắp theo `Time`, 20% cuối làm test.

**Phương án đã loại.**
- *Chia theo thẻ* (khách ở test chưa từng xuất hiện ở train): cũng hết rò rỉ (PR-AUC 0,914), nhưng không phải cách đánh giá chuẩn trong literature (hệ thống thật chấm cả khách cũ) và **không áp dụng được cho ULB** (không có mã khách hàng).
- *Luật "bỏ thẻ đã bị lộ khỏi test" của Handbook*: trên Sparkov nó bỏ **87%** tập test (556 nghìn → 74 nghìn dòng) và đẩy tỷ lệ fraud 0,39% → 2,9%, vì bộ mô phỏng **không bao giờ khoá** thẻ bị hack (976/999 thẻ bị hack lúc nào đó mà vẫn giao dịch bình thường). Luật này dành cho dữ liệu thật, nơi thẻ bị lộ bị khoá.
- *Khoảng trễ 1 tuần giữa train và test*: chưa áp dụng, để giữ nguyên tập test gốc của Sparkov và so sánh được với các bài dùng nó.

**Câu hỏi dễ gặp.**
- *"Spec ban đầu chốt chia ngẫu nhiên để tránh concept drift làm nhiễu CIES, sao đổi?"* — Drift là điều kiện thật của bài toán; chia ngẫu nhiên tránh được drift nhưng đổi lại rò rỉ theo đợt hack (100% thẻ fraud ở test đã có ở train), tức đo trên một bài toán dễ hơn thực tế. Chia theo thời gian cũng là chuẩn của literature và giúp so sánh được với các bài dùng cùng tập test Sparkov.
- *"Sao tỷ lệ fraud train 0,58% mà test 0,39%?"* — Đó là phân phối thật của 2 giai đoạn; không phân tầng lại để giữ đúng điều kiện "dự đoán tương lai".

---

## 2. Bỏ các cột định danh khách hàng

**Vấn đề.** Nhiều cột thuộc tính khách hàng, gộp lại, chỉ ra **đúng 1 người**. Model dùng chúng để học thuộc "khách nào từng bị hack" thay vì học dấu hiệu gian lận.

**Bằng chứng — cấu trúc dữ liệu:**

| Cột | Mức định danh |
|---|---|
| `lat`, `long` (tọa độ nhà) | 98,6% cặp tọa độ chỉ thuộc 1 thẻ |
| `city` | 92,2% city chỉ có 1 thẻ |
| `city_pop` | 90,1% giá trị chỉ thuộc 1 thẻ |
| `zip` | mỗi mã ứng đúng 1 cặp tọa độ và 1 city; 98,6% mã chỉ có 1 thẻ |
| `merch_lat`, `merch_long` (tọa độ cửa hàng) | luôn nằm trong ~1,4° quanh nhà khách (trung vị 0,8°) — tọa độ nhà có nhiễu |
| `state`, `job` | gộp với các cột trên để định danh |

**Bằng chứng — thí nghiệm** (XGBoost + `class_weighting`, PR-AUC):

| Bộ feature | Chia theo dòng | Chia theo thẻ | **Chia theo thời gian** |
|---|---|---|---|
| Giữ đủ | 0,932 | 0,914 | **0,240** |
| Bỏ cả 8 thuộc tính khách hàng (gồm `age`, `gender`, `city_pop`) | 0,828 | 0,820 | 0,765 |
| Bỏ nhóm định danh (`lat`/`long`, `city`, `state`, `job`) | — | — | 0,881 |
| + bỏ `city_pop` | — | — | 0,874 |
| + bỏ tọa độ cửa hàng | — | — | 0,889 |
| **+ bỏ cả hai (thiết kế hiện tại)** | — | — | **0,882** (pipeline thật: **0,883**) |

**Đọc bảng.**
- Chia theo thời gian mà giữ đủ feature, model **sụp xuống 0,24**. Cơ chế: 762/983 thẻ bị hack trong giai đoạn train; model học "khách X từng bị hack → nghi ngờ", nhưng giai đoạn sau người bị hack lại là người khác, còn giao dịch sạch của khách cũ thì bị gắn cờ → precision sụp.
- Bỏ nhóm định danh: 0,24 → 0,88. Nhóm định danh là thủ phạm chính.
- Bỏ thêm `city_pop`, tọa độ cửa hàng: 0,874–0,889, trong mức nhiễu 1 lần chạy — không mất điểm, đổi lại khẳng định được **không feature nào định danh khách hàng lọt vào model** (cách làm của Wu 2026).
- Giữ `age`, `gender`: bỏ chúng làm 0,881 → 0,765. Đây là **hồ sơ nhân khẩu học**, không định danh (nhiều người cùng tuổi/giới), và bộ mô phỏng Sparkov sinh fraud theo hồ sơ khách hàng.

**Riêng `zip`.** Trước khi bỏ, Logistic Regression coi `zip` là feature quan trọng nhất (~17% SHAP, trên cả `amt` ~12%) — vô nghĩa, vì hệ số tuyến tính trên một mã bưu chính không có ý nghĩa.

**Câu hỏi dễ gặp.**
- *"Bỏ thông tin địa lý thì mất tín hiệu à?"* — Đo được: không. Chia theo thời gian, bỏ chúng làm PR-AUC **tăng** từ 0,24 lên 0,88. Thông tin "ở đâu" trong Sparkov thực chất là "là ai".
- *"Sao chia ngẫu nhiên thì các cột này lại giúp (0,932 so với 0,828)?"* — Vì chia ngẫu nhiên có rò rỉ theo đợt hack: nhận ra khách đang bị hack là đủ đoán đúng các dòng test cùng đợt. Chính con số đó là bằng chứng rò rỉ.

---

## 3. Tính `age` theo năm của từng giao dịch

**Vấn đề.** `age` từng tính `2020 − năm sinh` (hardcode), trong khi dữ liệu trải 2019-01 → 2020-12: **924.850 dòng (~50%) thuộc năm 2019** bị tính thừa 1 tuổi.

**Bằng chứng không bỏ qua được.** `age` xếp hạng 4–20/79 theo SHAP ở các model cây (CatBoost + `class_weighting` hạng 4, XGBoost hạng 6, RF hạng 8).

**Ảnh hưởng đo được** (thiết kế trước, tham số cũ, trước/sau khi sửa): PR-AUC lệch tối đa 0,0038 (TB 0,0011), CIES lệch tối đa 0,0051 (TB 0,0011). Nhỏ, nhưng sai về logic nên vẫn sửa.

---

## 4. Tune: validation theo thời gian, encode riêng từng fold

**Vấn đề 1 — xáo trộn khi tune.** 5-fold xáo trộn đặt các dòng của cùng 1 đợt hack vào cả phần train lẫn phần chấm điểm → tham số nào **học thuộc** giỏi thì thắng.

*Dấu hiệu (giả thuyết, chưa chứng minh):* tham số tune được đều bị đẩy về phía phức tạp nhất — RF `max_depth=30`, `min_samples_split=2`, `min_samples_leaf=1` (cả 3 chạm biên), CatBoost `depth=11` (biên 12), XGBoost `n_estimators=550` (biên 600). Model càng phức tạp càng nhớ được từng đợt hack.

**Quyết định.** Cửa sổ mở rộng 3 fold, 1/3 cuối tập train chia làm 3 khối validation liên tiếp:

| Fold | Train | Validation | Fraud trong validation |
|---|---|---|---|
| 1 | 2019-01-01 → 2019-12-18 (864.450 dòng) | 2019-12-18 → 2020-02-18 | 825 |
| 2 | → 2020-02-18 (1.008.525) | 2020-02-18 → 2020-04-26 | 848 |
| 3 | → 2020-04-26 (1.152.600) | 2020-04-26 → 2020-06-21 | 915 |

Cùng logic với cách chia train/test ("train quá khứ, chấm tương lai"), và nhanh hơn 5-fold (3 lần train/trial, các fold đầu ít dữ liệu hơn).

**Vấn đề 2 — target encoding thấy nhãn tương lai.** `merchant` được thay bằng tỷ lệ fraud của cửa hàng. Nếu encode 1 lần trên cả tập train rồi mới cắt fold, thì mã hoá của dòng 2019 đã dùng cả các vụ fraud của giai đoạn validation 2020 → model "nhìn trước" giai đoạn nó bị chấm.

*Quyết định:* mỗi fold encode lại **chỉ bằng các dòng trước khối validation** (`tune.py::encode_time_folds`). Có test chứng minh: cửa hàng chỉ bị lợi dụng ở giai đoạn sau thì được mã hoá thấp ở fold train (cách cũ mã hoá cao). Không tốn thêm thời gian: encode cả 3 fold trên toàn bộ dữ liệu mất 1,7 giây, dùng lại cho mọi trial.

*Phạm vi:* chỉ ảnh hưởng điểm nội bộ lúc tune. Tập test cuối cùng và CIES vốn đã sạch (test nằm hoàn toàn sau train; mỗi lần bootstrap CIES tự encode lại trên train).

---

## 5. 50 trial thay vì 100

**Lý do.**
- Quota Kaggle 30 giờ GPU/tuần phải đủ cho cả tune 5 model, benchmark và CIES. ANN trước đây chạy nhiều phiên 7,5 giờ mới tới ~30/100 trial.
- Bằng chứng hội tụ (từ lần tune 100 trial trước, ghi trong `pipeline_report.md`): RF đạt PR-AUC CV 0,8865 ở trial 37, chỉ lên 0,8872 ở trial 89 và 98. MedianPruner cắt 26–68/100 trial tuỳ model — phần lớn thông tin nằm ở nửa đầu.

**Bằng chứng trên thiết kế hiện tại** (đo từ checkpoint SQLite của cả 10 lượt tune, `results/tuning_checkpoints*/`; "PR-AUC CV tốt nhất tính tới trial n"):

| Dataset | Model | Best (trial 50) | Đạt best ở trial | Cách best ≤ 0,005 từ trial | Cải thiện từ trial 25 → 50 |
|---|---|---|---|---|---|
| Sparkov | LR | 0,2460 | 32 | 1 | 0,0000 |
| Sparkov | RF | 0,9122 | 47 | 1 | 0,0004 |
| Sparkov | XGBoost | 0,9214 | 22 | 4 | 0,0000 |
| Sparkov | CatBoost | 0,9242 | 25 | 4 | 0,0000 |
| Sparkov | ANN | 0,9190 | 34 | 9 | 0,0026 |
| ULB | LR | 0,8085 | 35 | 7 | 0,0013 |
| ULB | RF | 0,7938 | 6 | 5 | 0,0000 |
| ULB | XGBoost | 0,8095 | 45 | 13 | 0,0020 |
| ULB | CatBoost | 0,8118 | 12 | 6 | 0,0000 |
| ULB | ANN | 0,8028 | 38 | 5 | 0,0019 |

Mọi lượt đã cách kết quả cuối ≤ 0,005 từ trial 13; nửa sau (trial 25→50) chỉ thêm tối đa 0,0026 (6/10 lượt không thêm gì). Nói chính xác: **chưa hội tụ tuyệt đối** (vài lượt còn nhích ở trial 34–47), nhưng phần cải thiện còn lại nhỏ hơn mức dao động giữa các trial của ULB và nhỏ hơn chênh lệch giữa các model/kỹ thuật được so sánh.

---

## 6. Giữ SMOTE chuẩn — không ép dòng tổng hợp về one-hot hợp lệ

**Vấn đề.** SMOTE tạo dòng mới = A + λ·(B − A) giữa 2 fraud thật, áp cho **mọi cột**, kể cả one-hot. Ví dụ thật trong `train_encoded_smote.parquet`: A là khách nữ mua `kids_pets`, B là khách nam mua `personal_care`, λ = 0,283 → `gender_F = 0,717, gender_M = 0,283`, `category_kids_pets = 0,717, category_personal_care = 0,283` — "72% nữ", không tồn tại ngoài đời. Trên dữ liệu hiện tại: 33,8% dòng tổng hợp có `gender` lưng chừng, 29,0% "bật" >1 `category`.

**Quyết định (chốt 2026-09-26).** Không ép (`SNAP_SYNTHETIC_ONEHOT = False`, mặc định):
- Đúng SMOTE chuẩn của spec và literature → so sánh được với các bài khác.
- Mọi **đánh giá** (PR-AUC trên test) và **giải thích** (SHAP/CIES trên eval) tính trên giao dịch **thật**; dòng tổng hợp chỉ có trong dữ liệu train.
- Lần đo trước (thiết kế cũ): ép one-hot làm PR-AUC giảm mạnh (XGBoost + SMOTE 0,887 → 0,766; SMOTENC 0,733) còn CIES gần như không đổi (0,957 → 0,959).

**Nói chính xác khi bị hỏi.** Dòng tổng hợp không lọt vào phần chấm điểm, nhưng **có** ảnh hưởng gián tiếp qua việc train (chính vì vậy ép hay không làm PR-AUC khác nhau). Câu trả lời an toàn: *"Dòng tổng hợp có one-hot không hợp lệ chỉ nằm trong dữ liệu train; mọi đánh giá và giải thích tính trên giao dịch thật. Chúng tôi giữ SMOTE chuẩn như literature; bản ép one-hot hợp lệ làm PR-AUC giảm mạnh còn CIES gần như không đổi."* Số 0,887 → 0,766 đo trên thiết kế trước — nếu cần số trên thiết kế hiện tại, chạy lại so sánh với `SNAP_SYNTHETIC_ONEHOT=True`.

---

## 7. Sửa lỗi kỹ thuật (không ảnh hưởng kết quả đã có, nhưng nên biết)

- **Ghi file kết quả dùng chung không khoá** (`best_params.json`, `cies_summary_results*.json`): stress test 8 tiến trình ghi cùng lúc — cách cũ mất 176/200 kết quả, cách mới (khoá + ghi atomic, `src/utils/jsonio.py`) giữ đủ 200/200. Các lần chạy song song trước không mất gì vì mỗi tổ hợp ghi cách nhau hàng chục phút.
- **Notebook Kaggle bỏ qua gần hết việc** (phát hiện trước khi chạy): `git clone` mang theo kết quả cũ trong repo khiến notebook tưởng mọi tổ hợp đã xong.
- **`encode_test` ghép lệch dòng khi bảng test có index khác 0..n−1** (phát hiện khi rà soát 2026-09-27): số dòng nhân đôi, nửa là NaN. Mọi nơi gọi đều truyền index mặc định — file test đã encode tái tạo khớp từng giá trị — nên không kết quả nào bị ảnh hưởng.
- **CIES không ghi explainer thực sự đã dùng**: ANN tự lùi DeepExplainer → KernelExplainer nếu Deep lỗi, mà không để lại dấu vết. Nay mỗi run ghi `explainer`; cả 200 run ANN đều là DeepExplainer.

---

## 8. Giữ vùng tìm Optuna ban đầu — đã thử nới biên, không cải thiện (2026-09-26)

**Vấn đề.** Nếu giá trị tốt nhất nằm sát biên vùng tìm, có thể giá trị tốt hơn nằm ngoài vùng. Lần tune đầu của thiết kế hiện tại có vài tham số sát biên (vị trí trong khoảng, thang log với tham số log):

| Dataset | Model | Tham số | Giá trị tốt nhất | Vùng tìm | Vị trí |
|---|---|---|---|---|---|
| Sparkov | XGBoost | `n_estimators` | 550 | 100–600 | 90% |
| Sparkov | XGBoost | `learning_rate` | 0,0137 | 0,01–0,3 (log) | 9% |
| Sparkov | RF | `max_depth` | 27 | 6–30 | 88% |
| ULB | LR | `C` | 0,00085 | 1e-4–1e4 (log) | 12% |

**Kiểm tra độ nhạy.** Tune lại với vùng nới rộng (`n_estimators`/`iterations` 100–2000, `learning_rate` 0,001–0,3, RF `max_depth` 6–50, LR `C` 1e-6–1e4), cùng 50 trial, 3 fold thời gian:

| Dataset | Model | PR-AUC CV vùng ban đầu | PR-AUC CV vùng nới | Tham số vùng nới chọn |
|---|---|---|---|---|
| Sparkov | XGBoost | 0,9214 (550 cây, lr 0,0137) | 0,9212 (1850 cây, lr 0,0043; 45 trial) | nhiều cây hơn, lr nhỏ hơn |
| ULB | LR | 0,8085 (`C`=0,00085) | 0,8089 (`C`=0,00084) | gần như cùng giá trị |
| ULB | XGBoost | 0,8095 (550 cây, `max_depth`=3) | 0,8133 (800 cây, `max_depth`=7) | nhiều cây hơn, sâu hơn |
| ULB | CatBoost | 0,8118 (400 iterations, `depth`=4) | 0,8111 (750 iterations, `depth`=4) | nhiều iterations hơn |

- Nới biên, Optuna chuyển sang **nhiều cây + learning rate nhỏ hơn** — hai tham số bù cho nhau, PR-AUC **không đổi**. 15 trial tốt nhất của vùng nới chỉ chênh 0,0023 PR-AUC, từ 450 tới 2000 cây: đây là một dải phẳng, không phải tối ưu bị chặn ở biên.
- ULB: XGBoost chênh +0,0038, CatBoost −0,0007 — cả hai nằm trong mức dao động giữa các trial của ULB (mỗi khối validation chỉ có 24–36 fraud; các trial hoàn tất của XGBoost dao động 0,79–0,81).
- LR ULB: vùng nới xuống tới 1e-6 nhưng vẫn chọn `C` ≈ 0,00084 (vị trí 29%) — tối ưu thật nằm trong vùng ban đầu.
- **Chi phí:** trial XGBoost 1200–2000 cây mất trung bình 1,6 phút, so với 0,4 phút ở vùng ban đầu (~4×); benchmark và CIES (20 bootstrap × 5 kỹ thuật) chậm theo cùng tỉ lệ.

**Quyết định.** Giữ vùng tìm ban đầu (`tune.py::sample_hyperparameters`), **chung cho cả Sparkov và ULB**. Nới biên không làm PR-AUC tốt hơn mà làm mọi bước sau chậm ~4×.

**Trả lời khi bị hỏi.** *"Chúng tôi đã thử nới biên ở các tham số sát biên: model chuyển sang nhiều cây hơn với learning rate nhỏ hơn nhưng PR-AUC CV không đổi trên Sparkov (0,9214 → 0,9212); trên ULB chênh không quá 0,004, nằm trong mức dao động do mỗi khối validation chỉ có 24–36 fraud. Thời gian train tăng ~4×, nên giữ vùng tìm ban đầu."*

**Hạn chế còn lại.** RF Sparkov `max_depth=27` (88%) chưa được kiểm tra bằng vùng nới (lượt tune nới dừng trước RF). Một số tham số chạm biên **dưới** mà cả hai vùng đều không nới: ULB XGBoost `max_depth=3`, `colsample_bytree=0,607`; ULB CatBoost `depth=4` (vùng nới cũng chọn 4). Kết quả vùng nới lưu tại `results/tuning_sensitivity_wide_space/`.

---

## 9. ANN: giữ cấu trúc mạng, dùng batch lớn và ít epoch hơn (2026-09-26)

**Vấn đề.** Tune ANN trên CPU không xong nổi 1 trial trong 20 phút; trên Kaggle (GPU) lần tune trước mất nhiều phiên 7,5 giờ mới được ~30 trial. Đo 1 epoch trên 200k dòng ngẫu nhiên × 21 feature (CPU, torch 2.14, 6 luồng):

| Batch | Bước/epoch | DataLoader (cũ) | Cắt thẳng từ tensor (mới) |
|---|---|---|---|
| 256 | 782 | 2,32 s | 2,41 s |
| 1024 | 196 | 2,45 s | 1,10 s |
| 2048 | 98 | 2,23 s | 0,90 s |
| 4096 | 49 | 1,94 s | 0,77 s |

- Mạng nhỏ (21 → 128 → 64 → 32 → 1, ~13,6k tham số): mỗi bước train gần như không có phép tính, thời gian là chi phí cố định mỗi bước (Python, autograd, Adam). Batch nhỏ = nhiều bước = chậm; GPU cũng không giúp vì phép tính quá nhỏ.
- `DataLoader(TensorDataset)` lấy từng dòng rồi ghép: 47–53% thời gian epoch ở batch 1024–4096.
- Vùng tìm cũ (batch 256–1024, 10–100 epoch) × 3 fold (~3 triệu dòng/epoch): ~30 phút/trial trung bình, ~60 phút nếu 100 epoch.

**Quyết định.**
- **Cấu trúc mạng giữ nguyên, không tune.** ~13,6k tham số so với ~1 triệu dòng: đủ học tương tác giữa feature, khó học thuộc. Không đưa số lớp/số nơ-ron vào 50 trial để vùng tìm không bị loãng.
- **`batch_size` ∈ {1024, 2048, 4096}** (cũ: 256, 512, 1024). Lý do chính là mất cân bằng: Sparkov ~0,58% fraud → batch 256 chỉ có ~1,5 fraud (nhiều batch không có vụ nào), batch 2048 có ~12. BatchNorm cũng ổn định hơn với batch lớn.
- **`epochs` 5–50** (cũ: 10–100). Batch 2048 trên ~1 triệu dòng ≈ 560 bước/epoch; 20 epoch ≈ 11k bước, đủ cho mạng nhỏ này.
- `lr` 1e-4–1e-2 (log) và `dropout` 0,1–0,5 giữ nguyên.
- **Cắt batch thẳng từ tensor** thay `DataLoader` (`train.py::_train_ann`): cùng model, loss, dữ liệu; mỗi epoch vẫn đi qua mọi dòng đúng 1 lần theo 1 hoán vị ngẫu nhiên (có test).
- Vùng tìm chung cho Sparkov và ULB, như các model khác.

**Vì sao không dùng early stopping.** CIES đo độ ổn định SHAP qua 20 lần bootstrap. Early stopping làm mỗi lần dừng ở một epoch khác → SHAP dao động vì số epoch chứ không vì kỹ thuật imbalance. Số epoch cố định (được tune) giữ phép đo công bằng.

**Đã đo (Kaggle, GPU T4, 2026-09-27):** tune ANN Sparkov 50 trial ~1,1 giờ (trial hoàn tất ~2–2,5 phút; trước khi đổi: không xong nổi 1 trial trong 20 phút trên CPU). ULB ~0,1 phút/trial.

**Kiểm tra biên sau khi tune** (50 trial, trial hoàn tất gom theo nhóm; PR-AUC CV cao nhất / trung vị):

| Dataset | Tham số tốt nhất | Chạm biên | PR-AUC theo nhóm |
|---|---|---|---|
| Sparkov | lr 0,00181 · dropout 0,124 · epochs 40 · batch 2048 (PR-AUC CV 0,9190) | `dropout` (6% của 0,1–0,5) | dropout 0,1–0,15: 0,9190 / 0,9167 · 0,15–0,2: 0,9182 / 0,9157 · 0,2–0,3: 0,9135 / 0,9128 · 0,3–0,5: 0,9117 / 0,8893 |
| ULB | lr 0,00208 · dropout 0,409 · epochs 5 · batch 2048 (PR-AUC CV 0,8028) | `epochs` (= biên dưới 5) | 5 epoch: 0,8028 / 0,7980 · 10: 0,8009 / 0,7953 · 15: 0,7939 · 20–40: 0,767–0,789 |

Khác XGBoost (mục 8, dải phẳng), ở đây có **xu hướng thật** về phía biên: dropout thấp hơn (Sparkov) và ít epoch hơn (ULB) cho PR-AUC cao hơn. Nhưng mức tăng ở sát biên đã rất nhỏ — 2 nhóm dropout thấp nhất chỉ chênh 0,0008; 5 và 10 epoch trên ULB chênh 0,002, dưới mức dao động giữa các trial của ULB (24–36 fraud mỗi khối validation). ULB chọn ANN "nhẹ" giống các model cây trên ULB (XGBoost `max_depth=3`, CatBoost `depth=4`): chỉ 417 fraud trong train nên model đơn giản ít học thuộc hơn.

**Quyết định:** giữ vùng tìm, ghi vào hạn chế — nới biên (dropout 0–0,5, epochs 1–50) tốn thêm 1 phiên Kaggle (~6–7 giờ GPU: tune + benchmark + CIES ANN cả 2 dataset) cho mức tăng PR-AUC kỳ vọng ~0,001–0,002, cùng tiêu chí đã áp cho XGBoost (chỉ nới khi có lợi đáng kể).

---

## 10. ULB được tune, benchmark và giải thích bằng tham số riêng

**Vấn đề.** Ban đầu notebook 05 gọi CIES ULB mà không truyền tham số → `build_model` tự nạp `best_params.json` — tức **tham số tune trên
Sparkov** — cho dữ liệu ULB (khác hẳn: 29 feature PCA, 417 fraud). Tham số tối ưu khác xa giữa 2 dataset, ví dụ LR: `C` = 0,214
(Sparkov) so với 0,00085 (ULB); XGBoost ULB chọn `max_depth` = 3, Sparkov chọn 11.

**Quyết định.** ULB tune riêng với **cùng giao thức** (cùng vùng tìm, 50 trial, 3 fold theo thời gian trên `ulb_train`), lưu
`best_params_ulb.json`; benchmark và CIES ULB truyền tham số này tường minh và **báo lỗi** nếu thiếu — không bao giờ lặng lẽ dùng tham số
Sparkov (có test hồi quy). ULB cũng có benchmark riêng (`model_benchmark_results_ulb.csv`) để biết CIES cao có đi kèm model bắt được
fraud không.

**Nói chính xác khi bị hỏi.** Trên tập test ULB, tham số Sparkov có lúc cho PR-AUC nhỉnh hơn (XGBoost × SMOTE: 0,813 so với 0,791) —
nhưng chọn tham số theo kết quả test là rò rỉ; tham số phải được chọn trên dữ liệu train của chính dataset đó. Với 75 fraud ở test,
chênh lệch cỡ 0,02 nằm trong mức dao động.

---

## 11. Chỉ so CIES trong cùng dataset — mức CIES phụ thuộc cỡ mẫu

**Vấn đề.** CIES Sparkov bootstrap từ mẫu phân tầng 100.000 dòng (579 fraud), ULB từ toàn bộ train (417 fraud). Nếu CIES phụ thuộc số
fraud trong mẫu thì so mức CIES giữa 2 dataset là không công bằng.

**Bằng chứng** (kiểm tra độ nhạy, thiết kế hiện tại, `results/cies_sensitivity_subsample.json`):

| Số fraud trong mẫu | 58 | 174 | 289 | 579 |
|---|---|---|---|---|
| XGBoost × class weighting | 0,943 | 0,951 | 0,957 | 0,970 |
| RF × SMOTE | 0,894 | 0,936 | 0,945 | 0,961 |

CIES **tăng theo cỡ mẫu và chưa bão hoà**; mốc 579 fraud tính lại độc lập trùng từng chữ số với CIES đã lưu.

**Quyết định.** So CIES **trong cùng dataset** (giữa kỹ thuật, giữa model — cùng cỡ mẫu). Giữa 2 dataset chỉ so **thứ hạng** của các kỹ
thuật (hình `cies_technique_ranks.png`: Borderline-SMOTE hạng cuối ở 9/10 cặp). Câu "ULB kém ổn định hơn Sparkov ở 21/25 tổ hợp" **không**
được dùng làm kết luận — nó lẫn giữa đặc tính dữ liệu và cỡ mẫu.

**Phương án đã loại.** Lấy mẫu Sparkov bằng đúng 417 fraud để 2 dataset ngang nhau: làm CIES Sparkov nhiễu hơn (ít fraud hơn) và phải chạy
lại toàn bộ 25 tổ hợp; không cần thiết vì câu hỏi nghiên cứu là so **kỹ thuật** trong cùng dataset.

---

## 12. Dữ liệu nền SHAP của LR và ANN: cố định giữa các run, đủ lớn để không phụ thuộc việc chọn dòng

**Vấn đề 1 — nền đổi theo run.** LinearExplainer (LR) và DeepExplainer (ANN) tính SHAP **so với một tập dữ liệu nền**: với LR, SHAP
của feature j = `hệ số_j × (x_j − trung bình nền_j)`. Nếu nền lấy từ mẫu bootstrap của từng run, mốc so sánh dịch theo run: thứ hạng
feature dao động thêm vì mốc chứ không vì model.

| Tổ hợp (Sparkov, giữ nguyên model đã train, chỉ đổi nền) | Nền đổi theo run | Nền cố định (100 dòng) |
|---|---|---|
| LR × SMOTE | 0,912 | 0,924 |
| LR × Borderline-SMOTE | 0,906 | 0,913 |
| LR × class weighting | 0,924 | 0,938 |
| ANN × SMOTE | 0,838 | 0,867 |
| ANN × class weighting | 0,907 | 0,921 |

Feature hưởng lợi nhiều nhất là các cột one-hot `category_*` và `amt` (đuôi dài). Trên ULB (feature PCA liên tục) chênh lệch nhỏ.

**Vấn đề 2 — nền 100 dòng quá nhỏ.** Cố định 100 dòng thì hết dao động giữa các run, nhưng kết quả lại phụ thuộc việc chọn 100 dòng
nào (100 dòng chọn bằng seed 42 không có dòng fraud nào). Đo bằng cách giữ nguyên 20 model đã train, chỉ đổi seed chọn nền (42/0/1/2/3):

| Tổ hợp (Sparkov) | Khoảng CIES theo seed — nền 100 dòng | Nền 1000 dòng | Nền = toàn bộ train |
|---|---|---|---|
| LR × SMOTE | 0,0072 | 0,0062 | (1 giá trị: 0,9242) |
| LR × class weighting | 0,0099 | 0,0025 | (0,9375) |
| LR × ADASYN | 0,0075 | 0,0068 | (0,9392) |
| ANN × SMOTE | **0,0305** | 0,0066 | — |

Với LR, ngay cả 1000 dòng vẫn dao động ~0,006: mean\|SHAP\| của một cột one-hot hiếm phụ thuộc trực tiếp vào tỉ lệ của nó trong nền,
và tỉ lệ ~3% trên 1000 dòng vẫn lệch ±18%. Khoảng dao động này ngang chênh lệch giữa vài kỹ thuật của LR.

**Quyết định.**
- **LR:** nền = **toàn bộ tập train gốc** (encode theo mapping của từng run). LinearExplainer chỉ dùng trung bình nên không tốn thêm,
  và kết quả không còn phụ thuộc việc chọn dòng. `shap.maskers.Independent` phải đặt `max_samples=len(nền)` — mặc định nó chỉ giữ
  tối đa 100 dòng (tự lấy mẫu con).
- **ANN:** nền = **1000 dòng** chọn 1 lần từ tập train gốc (seed 42, `config.SHAP_BACKGROUND_N`), cùng các dòng ở mọi run. DeepExplainer
  tốn công theo số dòng nền (0,4 → 3,1 giây/lần), nên không dùng toàn bộ train; 1000 dòng giảm dao động theo seed từ 0,031 xuống 0,007.
- Notebook 06 (SHAP so sánh model) dùng cùng nguyên tắc cho LR.
- Test hồi quy: `test_cies_shap_background_is_the_same_rows_in_every_run`, `test_cies_shap_background_is_encoded_with_each_runs_mapping`,
  `test_cies_shap_background_for_other_models_is_a_fixed_sample`, `test_linear_shap_uses_every_background_row`.

**Model cây (giữ nguyên, ghi giới hạn).** TreeExplainer mặc định (`tree_path_dependent`) không nhận nền ngoài: mốc của nó là số mẫu đi qua
từng nhánh lúc train, tức dữ liệu train của chính run đó — cũng đổi theo run. Đo trên RF × class weighting (Sparkov, 10 run): CIES
0,949 với cách mặc định, 0,978 khi dùng mốc cố định 100 dòng (`feature_perturbation="interventional"`) — chênh +0,029. XGBoost không chạy
được chế độ này với phiên bản `shap` hiện tại. Vì vậy **so sánh mức CIES giữa họ cây và LR/ANN không cùng điều kiện mốc**; câu hỏi chính
(so 5 kỹ thuật trong cùng 1 model) không bị ảnh hưởng vì mọi kỹ thuật của 1 model dùng cùng một loại mốc.

**Hệ quả.** Mọi yếu tố nằm ngoài model đều cố định giữa 20 run (tập eval, tham số, kỹ thuật, dữ liệu nền với LR/ANN). Số CIES hiện tại:
`reports/pipeline_report.md`, mục Kết quả.

---

## Nguồn

- Le Borgne, Siblini, Lebichot, Bontempi — *Reproducible Machine Learning for Credit Card Fraud Detection – Practical Handbook*. Baseline (chia theo thời gian, delay period, bỏ thẻ đã bị lộ): <https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/BaselineModeling.html>; chiến lược validation: <https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_5_ModelValidationAndSelection/ValidationStrategies.html>
- Dal Pozzolo, Boracchi, Caelen, Alippi, Bontempi — IEEE TNNLS, 2018: <https://www.researchgate.net/publication/319867396_Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy>
- Hayat, Magnier — Mathematics 13(16):2563, 2025: <https://arxiv.org/abs/2506.02703>
- Wu — arXiv:2607.14686, 2026 (preprint): <https://arxiv.org/abs/2607.14686>
- Chi tiết số đo và nhật ký: `reports/pipeline_report.md` (mục "Lệch so với spec" 9–10); nguồn cho các quyết định khác: `reports/literature_support.md`.
