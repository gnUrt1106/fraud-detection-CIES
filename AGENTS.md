# Fraud Detection - CIES: Agent Instructions

## Project Overview
- Nghiên cứu độ ổn định của giải thích SHAP (chỉ số CIES) dưới 5 kỹ thuật xử lý mất cân bằng, trên 5 model và 2 dataset (Sparkov, ULB).
- Ngôn ngữ chính: Python. Dùng virtual environment `.venv`.
- Đọc `AGENT_SPEC.md` trước khi đổi bất cứ thứ gì liên quan tới thiết kế thí nghiệm (encoding, kỹ thuật imbalance, SHAP, CIES): đó là các quyết định đã chốt. Mục 11 của file đó ghi các chỗ code hiện tại lệch spec.

## Hướng dẫn chi tiết
Xem các file trong `.agents/rules/`:
- `project-overview.md` — Tổng quan project và cấu trúc thư mục
- `coding-standards.md` — Quy chuẩn viết code Python, data, ML và các bẫy đã gặp
- `workflow.md` — Quy trình làm việc (task mới, sửa bug, thêm feature)

Tra cứu nhanh: `FILE_REFERENCE.md` (chức năng từng file), `reports/system_architecture.html` (sơ đồ kiến trúc), `reports/pipeline_report.md` (trạng thái, lệch spec, nhật ký sửa lỗi).

## Những điều PHẢI nhớ (đã từng gây lỗi thật)
- Chạy `python -m pytest tests -q` trước khi commit; mỗi bug sửa phải kèm test hồi quy.
- **Không nạp torch và xgboost trong cùng một process.** Mọi model/tổ hợp chạy qua `src/utils/isolation.py::run_isolated`. Test cần torch thì chạy trong subprocess riêng.
- `.gitignore` chỉ được ignore `/data/` ở gốc. Viết `data/` trần sẽ nuốt luôn `src/data/`.
- Feature của Logistic Regression và ANN phải qua `StandardScaler` (đã nằm trong dict model); SHAP của hai model này tính trong không gian đã scale.
- Gán `Series`/cột vào DataFrame khác index phải theo vị trí (`.to_numpy()`), không theo nhãn index.
- Trong notebook Kaggle, `!pip install` phải đặt spec trong ngoặc kép (`"optuna>=3.4.0"`), nếu không `>` thành chuyển hướng file.
- Không đổi công thức CIES, danh sách 5 kỹ thuật, hay cách encoding mà không hỏi người dùng — chúng ảnh hưởng tính hợp lệ của thí nghiệm.
- **Chia train/test theo thời gian, không đưa cột định danh khách hàng vào model** (`AGENT_SPEC.md` §8): chia ngẫu nhiên theo dòng + feature định danh từng cho PR-AUC ảo. Tune cũng validate theo thời gian, encode riêng từng fold.
- Cập nhật tài liệu (`README.md`, `FILE_REFERENCE.md`, `reports/pipeline_report.md`) khi thay đổi hành vi hoặc cấu trúc.
