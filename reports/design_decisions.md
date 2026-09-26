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
| Số trial Optuna | 100 | 50 |

> **Lưu ý khi trình bày kết quả:** các file trong `results/` hiện vẫn là của thiết kế trước, cho tới khi chạy lại (xem `pipeline_report.md`). Mọi số PR-AUC trong tài liệu này là của **XGBoost + `class_weighting`, tham số tune cũ, 1 lần chia, 1 seed** — dùng để so sánh các lựa chọn với nhau, không phải kết quả cuối.

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

**Hạn chế nói thẳng.** Bằng chứng hội tụ chỉ có cho RF; dữ liệu từng trial của lần tune cũ không còn lưu nên không vẽ lại được đường hội tụ cho các model khác. Muốn chắc hơn: chạy lại tune có lưu SQLite (`storage_path`) và vẽ "PR-AUC tốt nhất tới trial n" cho từng model.

---

## 6. Sửa lỗi kỹ thuật (không ảnh hưởng kết quả đã có, nhưng nên biết)

- **Ghi file kết quả dùng chung không khoá** (`best_params.json`, `cies_summary_results*.json`): stress test 8 tiến trình ghi cùng lúc — cách cũ mất 176/200 kết quả, cách mới (khoá + ghi atomic, `src/utils/jsonio.py`) giữ đủ 200/200. Các lần chạy song song trước không mất gì vì mỗi tổ hợp ghi cách nhau hàng chục phút.
- **Notebook Kaggle bỏ qua gần hết việc** (phát hiện trước khi chạy): `git clone` mang theo kết quả cũ trong repo khiến notebook tưởng mọi tổ hợp đã xong.

---

## Nguồn

- Le Borgne, Siblini, Lebichot, Bontempi — *Reproducible Machine Learning for Credit Card Fraud Detection – Practical Handbook*. Baseline (chia theo thời gian, delay period, bỏ thẻ đã bị lộ): <https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/BaselineModeling.html>; chiến lược validation: <https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_5_ModelValidationAndSelection/ValidationStrategies.html>
- Dal Pozzolo, Boracchi, Caelen, Alippi, Bontempi — IEEE TNNLS, 2018: <https://www.researchgate.net/publication/319867396_Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy>
- Hayat, Magnier — Mathematics 13(16):2563, 2025: <https://arxiv.org/abs/2506.02703>
- Wu — arXiv:2607.14686, 2026 (preprint): <https://arxiv.org/abs/2607.14686>
- Chi tiết số đo và nhật ký: `reports/pipeline_report.md` (mục "Lệch so với spec" 9–10); nguồn cho các quyết định khác: `reports/literature_support.md`.
