# Workflow Guidelines

## Khi nhận task mới
1. Đọc hiểu yêu cầu trước khi code
2. Kiểm tra code hiện tại liên quan; nếu chạm vào thiết kế thí nghiệm thì đọc `AGENT_SPEC.md`
3. Viết plan nếu task phức tạp
4. Implement từng bước, test sau mỗi bước

## Khi sửa bug
1. Reproduce bug trước (chạy thật, đo số, không chỉ đọc code)
2. Tìm root cause
3. Fix và viết test hồi quy; kiểm tra test **thất bại trên code cũ** rồi mới tin nó
4. Nếu bug làm sai kết quả đã tính, ghi rõ kết quả nào không còn hợp lệ trong `reports/pipeline_report.md`

## Khi thêm feature
1. Kiểm tra không break code hiện tại
2. Thêm unit test cho feature mới
3. Cập nhật documentation nếu cần

## Trước khi commit
1. `python -m pytest tests -q` phải xanh
2. Cập nhật `README.md` / `FILE_REFERENCE.md` / `reports/pipeline_report.md` nếu đổi hành vi hoặc cấu trúc
3. Không commit dữ liệu (`data/`), file tạm, `__pycache__`, `catboost_info/`
4. Đổi công thức CIES, danh sách kỹ thuật/model, hoặc cách encoding: dừng lại và hỏi người dùng
