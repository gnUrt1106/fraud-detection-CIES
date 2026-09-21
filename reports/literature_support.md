# Cơ sở tài liệu cho thiết kế pipeline (cập nhật 2026-09-21)

Mục đích: khi bị hỏi "vì sao thiết kế như vậy", có nguồn để trả lời — và biết chỗ nào **chưa có** nguồn.

**Cách kiểm chứng.** Bài CIES gốc (mục 1) đã được tải về và đọc phần phương pháp. Các nguồn còn lại chỉ được xác nhận tồn tại và đúng tác giả/năm/nơi đăng qua trang kết quả tìm kiếm (tiêu đề + tóm tắt); **chưa đọc toàn văn**. Cột "hỗ trợ điều gì" chỉ ghi phần đã thấy trong tóm tắt. Không nguồn nào ở đây được trích từ trí nhớ.

---

## 1. Bài CIES gốc — và CIES trong repo KHÁC nó

**Nguồn:** Văduva, Oprea, Bâra, *Measuring the Fragility of Trust: Devising Credibility Index via Explanation Stability (CIES) for Business Decision Support Systems*, arXiv:2603.05024, 05/03/2026 (bản preprint; chưa kiểm tra đã qua phản biện hay chưa). <https://arxiv.org/abs/2603.05024>

| | Bài gốc (mục 3.2–3.4) | CIES trong repo |
|---|---|---|
| Nhiễu được tạo ở đâu | Trên **đầu vào** `x` lúc suy luận: `x'ⱼ = xⱼ + N(0, (ε·\|xⱼ\|)²)`, K = 20 láng giềng | Trên **dữ liệu huấn luyện**: bootstrap → encode lại → imbalance → **train lại**, 20 run |
| Mô hình | Train một lần | Train lại ở mỗi run |
| Đối tượng so sánh | Giải thích SHAP của `x` và của `x'` (từng mẫu) | Vector thứ hạng mean\|SHAP\| toàn cục giữa các run |
| Khoảng cách | `D_w = Σ wⱼ·\|φⱼ(x) − φⱼ(x')\|` — trên **độ lớn SHAP** | Trên **thứ hạng** feature |
| Trọng số | `wⱼ = (1/rⱼ) / Σ(1/r)`, `rⱼ` = hạng trong giải thích **gốc** | `1/min(hạng_a, hạng_b)`, chuẩn hoá tổng = 1, trên mọi cặp run |
| Chuẩn hoá | Chia cho `‖φ(x)‖_w = Σ wⱼ\|φⱼ(x)\|`, cắt `max(0, ·)` | Không chia; `cies = 1 − khoảng cách TB` |
| Imbalance | Chỉ là điều kiện bật/tắt SMOTE | 5 kỹ thuật, là biến nghiên cứu chính |
| Dữ liệu | Churn, credit risk, HR attrition; 4 mô hình cây | Sparkov + ULB (fraud); 5 mô hình gồm LR và ANN |

**Giống nhau:** ý tưởng phạt nặng hơn khi feature quan trọng nhất bị xáo trộn (trọng số theo nghịch đảo hạng), điểm trong [0,1] với 1 = ổn định tuyệt đối, K/N_RUNS = 20.

**Hệ quả:**
- CIES của repo là **biến thể theo hướng nhiễu huấn luyện** (đo độ ổn định của giải thích khi dữ liệu huấn luyện đổi), không phải cùng một công thức. `AGENT_SPEC.md` ghi "theo công thức paper CIES gốc" là chưa chính xác.
- Cách nói an toàn với giảng viên: *"Em mượn ý tưởng rank-weighted distance của CIES (Văduva et al., 2026) nhưng đổi nguồn nhiễu từ đầu vào sang dữ liệu huấn luyện vì câu hỏi nghiên cứu là ảnh hưởng của kỹ thuật imbalance lên độ ổn định của giải thích."*
- Có thể tính thêm CIES đúng công thức gốc (nhiễu đầu vào, trên eval set cố định) như chỉ số phụ để đối chiếu. Đây là quyết định của bạn, tôi chưa đổi code.

---

## 2. Từng quyết định thiết kế và nguồn

| Quyết định | Nguồn | Hỗ trợ điều gì (theo tóm tắt đã thấy) | Mức hỗ trợ |
|---|---|---|---|
| Giải thích dễ mất ổn định, cần đo | Ghorbani, Abid, Zou (AAAI 2019) [S2]; Krishna et al. (2022) [S3] | [S2]: hai đầu vào gần như giống nhau, cùng nhãn dự đoán, có thể cho giải thích rất khác nhau. [S3]: các phương pháp giải thích hay bất đồng với nhau, người làm ML chưa có cách xử lý có nguyên tắc | Tốt cho **động cơ**; [S2] nói về ảnh, không phải tabular/SHAP |
| Nhiều mô hình tốt ngang nhau nhưng cấu trúc khác nhau (lý do train lại nhiều lần) | Breiman (2001), "Rashomon effect" [S4] | Hiệu ứng Rashomon: nhiều mô hình cùng độ chính xác nhưng cấu trúc khác | Tốt cho **lý do bootstrap + train lại** |
| Xếp hạng SHAP có thể đảo khi dữ liệu/mô hình đổi | Caraker et al. (arXiv 2026) [S5] | Khảo sát 77 dataset công khai, 68% có bất ổn định thuộc tính (attribution instability) | Chỉ mang tính tham khảo: preprint mới, tôi chỉ xác nhận được câu trên; **chưa** xác nhận phát biểu "đổi seed thì đảo hạng" |
| SHAP | Lundberg & Lee (NeurIPS 2017) [S6]; TreeSHAP: Lundberg et al. (Nat. Mach. Intell. 2020) [S7] | [S6]: khung SHAP thống nhất; [S7]: thuật toán thời gian đa thức cho mô hình cây | Tốt |
| PR-AUC là metric chính | Saito & Rehmsmeier (PLOS ONE 2015) [S8] | Đồ thị ROC dễ gây hiểu lầm trên dữ liệu mất cân bằng; PR phản ánh đúng phần dương đoán đúng | Tốt |
| SMOTE | Chawla et al. (JAIR 2002) [S9] | Bài gốc SMOTE; kèm biến thể SMOTE-NC cho dữ liệu có thuộc tính danh nghĩa | Tốt |
| SMOTE-ENN | Batista, Prati, Monard (SIGKDD Explorations 2004) [S10] | SMOTE+ENN cho kết quả rất tốt khi có ít mẫu dương | Tốt |
| ADASYN | He et al. (IJCNN 2008) [S11] | Sinh nhiều mẫu tổng hợp hơn cho mẫu khó học | Tốt |
| Borderline-SMOTE | Han, Wang, Mao (ICIC 2005) [S12] | Chỉ oversample các mẫu thiểu số gần ranh giới | Tốt |
| Ép one-hot hợp lệ sau SMOTE (`SNAP_SYNTHETIC_ONEHOT`) | SMOTE-NC trong [S9] (thuộc tính danh nghĩa lấy theo mode của láng giềng, không nội suy) | Xác nhận rằng nội suy trực tiếp thuộc tính danh nghĩa là vấn đề đã được biết | **Một phần**: cách "argmax của nhóm one-hot" của repo không phải SMOTE-NC; đây là lựa chọn riêng, đã đo được đánh đổi (xem `pipeline_report.md`) |
| Target encoding | Micci-Barreca (SIGKDD Explorations 2001) [S13] | Tiền xử lý thuộc tính danh nghĩa nhiều giá trị theo empirical Bayes | Tốt cho **cách mã hoá và làm mịn**; phần out-of-fold để tránh rò rỉ là quyết định riêng, chưa có nguồn |
| Optuna (TPE + pruner) | Akiba et al. (KDD 2019) [S14] | Khung tối ưu siêu tham số define-by-run kèm searching/pruning | Tốt |
| CIES, rank-weighted distance | Văduva et al. (2026) [S1] | Xem mục 1 | Ý tưởng có nguồn; công thức của repo là biến thể |

## 3. Chỗ CHƯA có nguồn (nên chuẩn bị câu trả lời riêng)

1. **Tune tách khỏi imbalance** (tune trên dữ liệu chưa resample rồi dùng chung cho 5 kỹ thuật): chưa tìm nguồn. Lập luận riêng: giữ tham số cố định để chênh lệch CIES là do kỹ thuật imbalance, không do tham số. Nhược điểm: tham số không tối ưu riêng cho từng kỹ thuật.
2. **Prior của target encoding lấy từ fold train**, `StratifiedGroupKFold` cho bản sao bootstrap: quyết định chống rò rỉ của repo, chưa có nguồn; đã kiểm bằng test hồi quy.
3. **Cỡ mẫu subsample Sparkov (100.000 dòng)**: chưa có nguồn; cần kết quả kiểm tra độ nhạy (notebook 04 mục 8).
4. **DeepExplainer cho ANN so với KernelSHAP**: tôi chưa tìm/đọc nguồn; căn cứ hiện có là số đo riêng trong `pipeline_report.md`.
5. **Tác động của kỹ thuật imbalance lên độ ổn định của SHAP trong bài toán gian lận**: tìm kiếm không ra nghiên cứu trực tiếp — nhất quán với nhận định "khoảng trống nghiên cứu" trong `benchmark_literature.md`, nhưng đây là kết quả của một vài truy vấn, **không phải khảo sát tài liệu có hệ thống**; đừng khẳng định "chưa ai làm" với giảng viên. Bài gần nhất tôi thấy là chính [S1] (SMOTE ảnh hưởng độ ổn định giải thích, kết quả phụ thuộc dataset).
6. Bảng trong `benchmark_literature.md` (Jemai et al. 2024, "Fraud Dataset Benchmark", v.v.) **chưa được kiểm chứng ở đợt này**; nhiều nguồn ghi mơ hồ, không có link. Không nên trích chúng trong báo cáo cho đến khi từng nguồn được mở và đối chiếu.

Có một bài liên quan tới gian lận + độ tin cậy của SHAP dưới dịch chuyển phân phối (MDPI FinTech 5(3):77, "The Explainability–Reliability Gap in Fraud Detection") xuất hiện trong kết quả tìm kiếm nhưng trang bị chặn (HTTP 403), nên **chưa đọc và chưa dùng làm nguồn**.

---

## 4. Danh mục nguồn

- [S1] Văduva, A.-G., Oprea, S.-V., Bâra, A. (2026). *Measuring the Fragility of Trust: Devising Credibility Index via Explanation Stability (CIES) for Business Decision Support Systems.* arXiv:2603.05024. <https://arxiv.org/abs/2603.05024>
- [S2] Ghorbani, A., Abid, A., Zou, J. (2019). *Interpretation of Neural Networks Is Fragile.* AAAI, 33, 3681–3688. <https://ojs.aaai.org/index.php/AAAI/article/view/4252>
- [S3] Krishna, S., Han, T., Gu, A., Wu, S., Jabbari, S., Lakkaraju, H. (2022; TMLR). *The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective.* <https://arxiv.org/abs/2202.01602>
- [S4] Breiman, L. (2001). *Statistical Modeling: The Two Cultures.* Statistical Science, 16(3), 199–231. <https://projecteuclid.org/journals/statistical-science/volume-16/issue-3/Statistical-Modeling--The-Two-Cultures-with-comments-and-a/10.1214/ss/1009213726.full>
- [S5] Caraker, D., Arnold, B., Rhoads, D. (2026). *The Attribution Impossibility: No Feature Ranking Is Faithful, Stable, and Complete Under Collinearity.* arXiv:2605.21492. <https://arxiv.org/abs/2605.21492>
- [S6] Lundberg, S. M., Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS. <https://arxiv.org/abs/1705.07874>
- [S7] Lundberg, S. M., et al. (2020). *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence, 2, 56–67. <https://doi.org/10.1038/s42256-019-0138-9>
- [S8] Saito, T., Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLOS ONE. <https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0118432>
- [S9] Chawla, N. V., Bowyer, K. W., Hall, L. O., Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 16, 321–357. <https://www.jair.org/index.php/jair/article/view/10302>
- [S10] Batista, G. E. A. P. A., Prati, R. C., Monard, M. C. (2004). *A study of the behavior of several methods for balancing machine learning training data.* ACM SIGKDD Explorations, 6(1), 20–29. <https://dl.acm.org/doi/10.1145/1007730.1007735>
- [S11] He, H., Bai, Y., Garcia, E. A., Li, S. (2008). *ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning.* IJCNN, 1322–1328. <https://dblp.org/rec/conf/ijcnn/HeBGL08.html>
- [S12] Han, H., Wang, W.-Y., Mao, B.-H. (2005). *Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning.* ICIC, 878–887. <https://doi.org/10.1007/11538059_91>
- [S13] Micci-Barreca, D. (2001). *A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems.* ACM SIGKDD Explorations, 3(1), 27–32. <https://dl.acm.org/doi/10.1145/507533.507538>
- [S14] Akiba, T., et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD. <https://arxiv.org/abs/1907.10902>
