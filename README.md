# 🔍 Fraud Detection Thesis — Phát hiện gian lận giao dịch tài chính

## Mô tả

Đồ án nghiên cứu về **phát hiện gian lận giao dịch tài chính** (Financial Fraud Detection) sử dụng các phương pháp Machine Learning. Project sử dụng bộ dữ liệu **Sparkov** được tạo bằng [Sparkov Data Generation Tool](https://github.com/namebrandon/Sparkov_Data_Generation) — một công cụ mô phỏng giao dịch thẻ tín dụng hợp lệ và gian lận dựa trên hồ sơ khách hàng thực tế.

### Nguồn dữ liệu

- **Kaggle Dataset**: [kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- **Công cụ sinh dữ liệu**: [Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation) by Brandon Harris
- **Thời gian bao phủ**: Giao dịch từ 01/01/2019 đến 31/12/2020
- **Quy mô**: ~1.3 triệu giao dịch train + ~550K giao dịch test

### Cấu trúc Project

```
fraud-detection-thesis/
├── data/
│   ├── raw/                  # Dữ liệu gốc tải từ Kaggle
│   └── processed/            # Dữ liệu đã tiền xử lý (bước sau)
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory Data Analysis
├── src/
│   ├── config.py             # Đường dẫn và hằng số dùng chung
│   └── data/
│       └── download.py       # Script tải dataset từ Kaggle
├── reports/
│   ├── figures/              # Biểu đồ xuất ra từ EDA
│   └── profiling/            # Báo cáo data profiling (HTML)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Hướng dẫn cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd fraud-detection-thesis
```

### 2. Tạo môi trường ảo và cài dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Cấu hình Kaggle API Credentials

Bạn cần Kaggle API token để tải dữ liệu. Chọn **một trong hai cách** sau:

#### Cách 1 — File `kaggle.json` (khuyến nghị)

1. Truy cập [Kaggle Settings](https://www.kaggle.com/settings)
2. Mục **API** → nhấn **"Create New Token"**
3. File `kaggle.json` sẽ tự tải về, chứa `username` và `key`
4. Di chuyển file vào thư mục Kaggle:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### Cách 2 — Biến môi trường

```bash
cp .env.example .env
# Mở file .env và điền KAGGLE_USERNAME, KAGGLE_KEY
```

Hoặc export trực tiếp:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### 4. Tải dữ liệu

```bash
python -m src.data.download
```

Script sẽ tải dataset Sparkov từ Kaggle về `data/raw/`. Nếu dữ liệu đã tồn tại, script sẽ bỏ qua.

### 5. Chạy notebook EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Hoặc dùng JupyterLab:

```bash
jupyter lab
```

---

## Tiến độ

- [x] Khởi tạo cấu trúc project
- [x] Script tải dữ liệu
- [x] Exploratory Data Analysis (EDA)
- [x] Data Profiling tự động
- [ ] Tiền xử lý dữ liệu (encoding, xử lý mất cân bằng, train/test split)
- [ ] Huấn luyện mô hình (LR, RF, XGBoost, CatBoost, ANN)
- [ ] Đánh giá và so sánh mô hình
- [ ] Giải thích mô hình (SHAP, CIES)

---

## License

Project phục vụ mục đích nghiên cứu học thuật.

## Trích dẫn

```
@misc{sparkov2020,
  author = {Brandon Harris},
  title = {Sparkov Data Generation},
  year = {2020},
  publisher = {GitHub},
  url = {https://github.com/namebrandon/Sparkov_Data_Generation}
}

@misc{kartik2020fraud,
  author = {Kartik Shenoy},
  title = {Fraud Detection Dataset},
  year = {2020},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/kartik2112/fraud-detection}
}
```
