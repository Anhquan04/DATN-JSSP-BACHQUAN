# ĐATN: Áp dụng học tăng cường (Actor-Critic) để tối ưu lập lịch sản xuất trong mô hình Job-Shop

**Sinh viên**: Bạch Công Quân &nbsp;|&nbsp; **MSSV**: 1671020261  
**GVHD**: ThS. Tạ Chí Hiếu &nbsp;|&nbsp; **Lớp**: CNTT-1602 &nbsp;|&nbsp; **Năm**: 2026

---

## 📋 Giới thiệu

Đề tài áp dụng thuật toán **Actor-Critic (A2C)** — một phương pháp học tăng cường (Reinforcement Learning) — để giải bài toán **Job-Shop Scheduling Problem (JSSP)**, sau đó so sánh hiệu quả với các phương pháp heuristic truyền thống (FIFO, SPT, LPT, EDD).

**Kết quả nổi bật trên instance FT06 (6 Jobs × 6 Machines):**

| Algorithm | Mean Makespan | Best Makespan | vs FIFO |
|-----------|--------------|---------------|---------|
| **A2C**   | ~97          | **63**        | −36%    |
| EDD       | 101          | 101           | −34%    |
| SPT       | 109          | 109           | −28%    |
| LPT       | 129          | 129           | −15%    |
| FIFO      | 152          | 152           | baseline|

---

## 🗂️ Cấu trúc Project

```
jssp_a2c/
│
├── agent/
│   ├── a2c_numpy.py          ← A2C thuần NumPy (không cần PyTorch) ✅ Dùng chính
│   └── a2c_agent.py          ← A2C bằng PyTorch (cần cài torch)
│
├── baselines/
│   └── dispatching_rules.py  ← FIFO, SPT, LPT, EDD, Random
│
├── data/
│   └── instances.py          ← Benchmark instances (3x3, 4x4, 5x5, FT06, FT10)
│
├── env/
│   └── job_shop_env.py       ← Môi trường JSSP (MDP — Gym-compatible)
│
├── evaluation/
│   └── evaluate.py           ← So sánh A2C vs Baselines, vẽ charts, Wilcoxon test
│
├── notebooks/
│   └── DATN_Analysis.ipynb   ← Phân tích dữ liệu từ CSV (Jupyter Notebook)
│
├── results/                  ← Model (.npz), CSV data, biểu đồ
│   ├── model_ft06.npz
│   ├── training_log_ft06.csv
│   ├── evaluation_ft06.csv
│   └── schedule_ft06_*.csv
│
├── static/
│   ├── css/style.css         ← Giao diện dark theme
│   └── js/app.js             ← Logic frontend (Canvas animation, API calls)
│
├── templates/
│   └── index.html            ← Giao diện web (Jinja2)
│
├── train/
│   ├── train_numpy.py        ← Script training chính ✅ Dùng cái này
│   └── train.py              ← Training bằng PyTorch (cần cài torch)
│
├── utils/
│   └── visualization.py      ← Vẽ Gantt chart, learning curves, box plot, bar chart
│
├── main.py                   ← Flask server — Entry point chính
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt

### Yêu cầu
- Python 3.9+
- Các thư viện trong `requirements.txt`

### Bước 1: Clone / tải project về

```bash
cd jssp_a2c
```

### Bước 2: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Hoặc cài thủ công:

```bash
pip install flask numpy matplotlib scipy seaborn
# Optional (nếu muốn dùng PyTorch):
pip install torch
```

---

## 🚀 Cách chạy

### 1. Khởi động Web App (Flask)

```bash
python main.py
```

Mở trình duyệt: **http://localhost:5000**

Giao diện cho phép:
- Chọn instance (3×3, 4×4, 5×5, FT06, FT10)
- Chọn algorithm (A2C, FIFO, SPT, LPT, EDD)
- Xem Gantt chart animation real-time
- So sánh tất cả algorithms song song
- Train model mới trực tiếp từ trình duyệt

---

### 2. Train Model A2C

```bash
python train/train_numpy.py --instance ft06 --episodes 2000
```

**Các tham số:**

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--instance` | `ft06` | Instance: `3x3`, `4x4`, `5x5`, `ft06`, `ft10` |
| `--episodes` | `1000` | Số episode huấn luyện |
| `--hidden` | `128` | Số neuron mỗi lớp ẩn |
| `--lr_actor` | `3e-4` | Learning rate Actor |
| `--lr_critic` | `5e-4` | Learning rate Critic |
| `--gamma` | `0.99` | Discount factor |
| `--eval_runs` | `30` | Số lần chạy để đánh giá thống kê |
| `--save_dir` | `results` | Thư mục lưu kết quả |

**Output sau khi train:**
```
results/
├── model_ft06.npz                 ← Model đã train
├── training_log_ft06.csv          ← Log từng episode (append — giữ data cũ)
├── evaluation_ft06.csv            ← Kết quả so sánh (append — giữ data cũ)
└── schedule_ft06_{algo}.csv       ← Lịch sản xuất tốt nhất
```

> **Lưu ý:** Mỗi lần train tạo ra một `run_id` mới và **ghi thêm vào CSV** (không xóa data cũ). Notebook tự động đọc tất cả các lần train.

---

### 3. Đánh giá Model

```bash
python evaluation/evaluate.py \
    --instance ft06 \
    --model_path results/model_ft06.npz \
    --n_runs 30
```

**Output:**
- Bảng so sánh Makespan (Mean, Std, Min, Max)
- Wilcoxon Signed-Rank Test (kiểm định thống kê)
- Charts: Gantt, Box plot, Bar chart (lưu vào `results/`)

---

### 4. Phân tích dữ liệu (Jupyter Notebook)

```bash
jupyter notebook notebooks/DATN_Analysis.ipynb
```

Mở notebook, chỉnh **Cell 1 (CONFIG)**:

```python
RESULTS_DIR = "../results"   # Đường dẫn folder chứa CSV
INSTANCE    = "ft06"         # Instance muốn phân tích
RUN_IDS     = None           # None = tất cả | [1,2] = chỉ run 1 và 2
```

Sau đó **Kernel → Restart & Run All** để tạo lại toàn bộ biểu đồ.

---

## 🧠 Kiến trúc kỹ thuật

### MDP — Mô hình hóa JSSP

| Thành phần | Mô tả |
|---|---|
| **State** | Vector 24 chiều (FT06): 3 features/job × 6 jobs + 1 feature/machine × 6 machines |
| **Action** | Chọn job_id để schedule operation tiếp theo |
| **Reward** | `-1` mỗi bước + `-0.5×idle` + `+10` khi xong job + `+50×efficiency` khi xong tất cả |

### Actor-Critic Networks

```
Actor  (MLP): State(24) → Linear(128) → ReLU → Linear(128) → ReLU → Softmax(6)
Critic (MLP): State(24) → Linear(128) → ReLU → Linear(128) → ReLU → Scalar V(s)
```

### Baselines so sánh

| Algorithm | Nguyên lý |
|---|---|
| **FIFO** | Chọn job có index nhỏ nhất (đến trước phục vụ trước) |
| **SPT** | Chọn job có operation ngắn nhất hiện tại |
| **LPT** | Chọn job có operation dài nhất hiện tại |
| **EDD** | Chọn job có thời gian hoàn thành dự kiến sớm nhất |

---

## 📊 Benchmark Instances

| Instance | Jobs | Machines | Optimal LB | Nguồn |
|----------|------|----------|------------|-------|
| Simple 3×3 | 3 | 3 | — | Tự tạo |
| Demo 4×4 | 4 | 4 | — | Tự tạo |
| Demo 5×5 | 5 | 5 | — | Tự tạo |
| **FT06** | 6 | 6 | **55** | Fisher & Thompson (1963) |
| FT10 | 10 | 10 | 930 | Fisher & Thompson (1963) |

---

## 📁 File chạy được trực tiếp

```bash
python main.py                          # Flask web app
python train/train_numpy.py             # Training
python evaluation/evaluate.py           # Đánh giá
jupyter notebook notebooks/DATN_Analysis.ipynb  # Phân tích
```

> Các file còn lại (`env/`, `agent/`, `baselines/`, `utils/`, `data/`) là **module thư viện** — chỉ dùng để import, không chạy trực tiếp.

---

## 🔧 Hướng phát triển

- [ ] Nâng cấp từ A2C lên **PPO** (Proximal Policy Optimization) để training ổn định hơn
- [ ] Dùng **Graph Neural Network** để biểu diễn state thay vector phẳng
- [ ] Mở rộng sang **Flexible Job-Shop** (mỗi operation có nhiều lựa chọn máy)
- [ ] Tích hợp vào phần mềm ERP/MES thực tế
