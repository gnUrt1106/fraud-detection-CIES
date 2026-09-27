# Kết quả đã công bố trên Sparkov — đã kiểm chứng (cập nhật 2026-09-27)

`AGENT_SPEC.md` yêu cầu file này: liệt kê kết quả của các nghiên cứu khác trên Sparkov (`kartik2112/fraud-detection`), cột CIES
để "N/A". Bản trước (tạo trước 2026-09-21) **không kiểm chứng được** — xem mục 2. Bản này chỉ giữ những gì đã mở và đọc nguồn gốc.

**Cách kiểm chứng (2026-09-27):** mở từng bài, đọc phần phương pháp + kết quả; số liệu dưới đây chép đúng như trong bài. "Đã đọc
toàn văn" = đọc cả phần kết quả, không chỉ tóm tắt.

---

## 1. Nguồn đã kiểm chứng

| Bài | Chia train/test | Feature | Model | Imbalance | Kết quả trên Sparkov | CIES | So được với luận văn? |
|---|---|---|---|---|---|---|---|
| **Wu (2026)**, *Class Weighting versus Amount Conditioning in Credit-Card Fraud Detection: A Dollar-Metric Study with a Temporal Explanation Audit*, arXiv:2607.14686 — đã đọc toàn văn | **Theo thời gian**, test = `fraudTest` gốc (555.719 dòng, 2.145 fraud); train chia tiếp theo thời gian thành train/validation | **Không dùng cột định danh** (bài có kiểm tra lúc chạy: không cột định danh/thời gian thô nào vào model); có thêm feature tỷ lệ số tiền và velocity theo từng thẻ | XGBoost | Không trọng số, **class weighting**, trọng số theo log số tiền | **AP (PR-AUC) 0,930–0,934** (Bảng I: không trọng số 0,930; class weighting 0,931; theo log số tiền 0,932–0,934) | N/A — nhưng có phân tích SHAP **theo thời gian** (xem mục 3) | **Gần nhất**: cùng tập test, cùng không dùng định danh. Chênh với CatBoost của luận văn (PR-AUC 0,887) phần lớn do Wu có feature velocity/tỷ lệ số tiền theo thẻ, luận văn không có (đã bỏ `cc_num`) |
| **Grover et al.**, *FDB: Fraud Dataset Benchmark*, Amazon, **arXiv:2208.14417 (2022)** — đã đọc phần kết quả | Dùng train/test gốc của Kaggle, **lấy mẫu ngẫu nhiên giảm tập test** | Theo loader của FDB | 5 framework AutoML: AFD OFI, AFD TFI, AutoGluon, H2O, Auto-sklearn | Không | **Chỉ báo cáo AUC-ROC**: AFD OFI 0,998; AutoGluon 0,997; H2O 0,997; Auto-sklearn 0,995 (AFD TFI không chạy) | N/A | **Không** so PR-AUC được (bài không có PR-AUC/F1); tập test đã bị lấy mẫu giảm |
| **Jemai, Zarrad, Daud (2024)**, *Identifying Fraudulent Credit Card Transactions Using Ensemble Learning*, IEEE Access 12:54893–54900, DOI 10.1109/ACCESS.2024.3380823 — đã đọc toàn văn | **Ngẫu nhiên** 70/15/15 + KFold 5 | 7 feature sau khi bỏ nhiều cột (có label-encode `category`, `job`) | Gaussian Naïve Bayes; RF (**5 cây, sâu 3**); XGBoost (**5 cây, sâu 3**) | Oversampling, undersampling, SMOTE | **Không có bảng số** — kết quả chỉ ở dạng hình. Kết luận của bài: model "perform ... significantly poorly on the simulated dataset" (Sparkov), SMOTE tốt hơn 2 cách kia, XGBoost tốt nhất | N/A | **Không**: chia ngẫu nhiên, model rất nhỏ, không có số để so |

**Đọc bảng:** con số so sánh hợp lệ duy nhất là của Wu (cùng tập test gốc, cùng không định danh). Kết quả của Wu cũng khớp với 2
quan sát của luận văn: (1) class weighting gần như không đổi PR-AUC so với không trọng số (0,931 so với 0,930; luận văn: kỹ thuật
imbalance làm PR-AUC chênh ≤ 3,4 điểm % trong cùng model); (2) mô hình cây/boosting cho AP ~0,9 trên Sparkov khi không dùng định danh.

---

## 2. Các dòng của bản trước — bị loại và lý do

| Dòng trong bản trước | Kiểm chứng | Lý do loại |
|---|---|---|
| Jemai et al. 2024 — RF + SMOTE (F1 0,854), XGBoost + SMOTE (F1 0,861), **Logistic Regression** (F1 0,612) | Bài có thật; số **không có** trong bài | Bài không dùng Logistic Regression; không có bảng số nào (chỉ hình); bài kết luận model làm kém trên Sparkov, trái với F1 ~0,86 |
| "Fraud Dataset Benchmark (FDB)" 2024 — LightGBM, CatBoost, XGBoost + SMOTE với PR-AUC 0,818–0,831, F1, Recall | Bài có thật (**2022**, không phải 2024); số **không có** trong bài | FDB không chạy LightGBM/CatBoost/XGBoost + SMOTE, không báo cáo PR-AUC, F1 hay Recall — chỉ AUC-ROC của 5 framework AutoML |
| "Systematic Review on Imbalance Fraud (MDPI)" 2025; "Deep Learning for Financial Fraud" 2024; "Graph & Ensemble Approaches" 2025 | **Không kiểm chứng được** | Không có tác giả, tên bài đầy đủ hay link — không xác định được bài nào để mở |

Không dùng bất kỳ con số nào của bản trước trong luận văn.

---

## 3. Khoảng trống nghiên cứu — nói đúng mức

Bản trước viết các công trình "hoàn toàn chưa từng đánh giá độ ổn định giải thích". **Câu đó sai:**
- **Văduva et al. (2026)** — bài CIES gốc (`literature_support.md` mục 1) — đã xét SMOTE bật/tắt ảnh hưởng tới độ ổn định giải thích,
  nhưng trên dữ liệu churn, credit risk, HR attrition, không phải gian lận thẻ.
- **Wu (2026)** có phân tích SHAP **theo thời gian trên Sparkov**: giải thích của nhóm fraud dịch chuyển nhiều hơn toàn bộ giao dịch
  (khoảng cách Jensen-Shannon trung bình 0,057 so với 0,012). Đây là độ ổn định theo **thời gian**, không phải theo kỹ thuật imbalance
  hay qua bootstrap.

**Cách nói an toàn:** *"Các nghiên cứu chúng tôi tìm được trên Sparkov chủ yếu báo cáo hiệu năng; một nghiên cứu gần đây (Wu, 2026)
xét SHAP dịch chuyển theo thời gian, và bài CIES gốc (Văduva et al., 2026) xét ảnh hưởng của SMOTE trên dữ liệu ngoài gian lận. Chúng
tôi chưa tìm thấy nghiên cứu so sánh độ ổn định SHAP qua bootstrap giữa 5 kỹ thuật imbalance trên dữ liệu gian lận."* Đây là kết quả
của vài lượt tìm kiếm, **không phải khảo sát tài liệu có hệ thống** — không nói "chưa ai làm".

---

## 4. Nhận định từ 3 nguồn đã kiểm

1. **Chỉ số:** chỉ Wu dùng AP/PR-AUC; FDB chỉ dùng AUC-ROC (0,995–0,998 — gần như bão hoà, không phân biệt được model trên dữ liệu
   mất cân bằng); Jemai dùng accuracy/precision/recall/F1. Ủng hộ việc luận văn lấy PR-AUC làm chỉ số chính.
2. **Cách chia dữ liệu quyết định con số:** Jemai chia ngẫu nhiên; FDB lấy mẫu giảm tập test; chỉ Wu chia theo thời gian trên tập test
   gốc. Con số giữa các bài không so trực tiếp được nếu khác cách chia (xem `pipeline_report.md`, mục "Lệch so với spec" 10: cùng
   XGBoost, chia ngẫu nhiên cho PR-AUC 0,932 còn chia theo thời gian không định danh 0,882).
3. **Mô hình cây/boosting dẫn đầu:** XGBoost tốt nhất ở Jemai; Wu dùng XGBoost đạt AP ~0,93 — khớp với luận văn (CatBoost/XGBoost/RF
   PR-AUC 0,84–0,89, LR 0,10–0,12 trên Sparkov).
