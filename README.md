# ĐATN: Áp dụng học tăng cường (Actor-Critic) để tối ưu lập lịch sản xuất trong mô hình Job-Shop

**Sinh viên**: Bạch Công Quân &nbsp;|&nbsp; **MSSV**: 1671020261  
**GVHD**: ThS. Tạ Chí Hiếu &nbsp;|&nbsp; **Lớp**: CNTT-1602 &nbsp;|&nbsp; **Năm**: 2026

---

## 📋 Giới thiệu

Đề tài áp dụng thuật toán **Actor-Critic (A2C)** — một phương pháp học tăng cường (Reinforcement Learning) — để giải bài toán **Job-Shop Scheduling Problem (JSSP)**, sau đó so sánh hiệu quả với các phương pháp heuristic truyền thống (FIFO, SPT, LPT, EDD).

**Kết quả nổi bật trên instance FT06 (6 Jobs × 6 Machines):**

| Algorithm | Best Makespan | vs FIFO | Ghi chú |
|-----------|---------------|---------|---------|
| **A2C**   | **59**        | **−61%** | Stochastic — 30 runs, GAE + Entropy Decay |
| EDD       | 101           | −34%    | Deterministic — 1 run |
| SPT       | 109           | −28%    | Deterministic — 1 run |
| LPT       | 129           | −15%    | Deterministic — 1 run |
| FIFO      | 152           | baseline| Deterministic — 1 run |
| Optimal   | 55            | —       | Fisher & Thompson (1963) |

> **A2C đạt makespan = 59**, chỉ cách optimal (55) khoảng 7.3% — vượt trội so với tất cả heuristic baselines.

---

## 🗂️ Cấu trúc Project

```
jssp_a2c/
│
├── agent/
│   ├── a2c_numpy.py          ← A2C thuần NumPy (CHÍNH) — GAE, entropy decay
│   └── a2c_agent.py          ← A2C bằng PyTorch (phụ — cần cài torch)
│
├── baselines/
│   └── dispatching_rules.py  ← FIFO, SPT, LPT, EDD, Random
│
├── data/
│   └── instances.py          ← Benchmark instances (3×3, 4×4, 5×5, FT06, FT10)
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
│   ├── train_numpy.py        ← Training chính ✅ (NumPy agent)
│   └── train.py              ← Training phụ (PyTorch agent — cần torch)
│
├── utils/
│   └── visualization.py      ← Vẽ Gantt chart, learning curves, box plot, bar chart
│
├── diagnostic.py             ← Vẽ learning curves từ CSV (debug training)
├── inspect_model.py          ← Xem thống kê model .npz (weights, shape, stats)
├── view_npz.py               ← Xem nội dung chi tiết bên trong file .npz
├── main.py                   ← Flask server — Entry point chính
├── requirements.txt
└── README.md
```

---

## 🔀 `train_numpy.py` vs `train.py` — NumPy vs PyTorch

Project có **2 phiên bản agent + training**, phục vụ mục đích khác nhau:

### ✅ `train_numpy.py` + `a2c_numpy.py` — PHIÊN BẢN CHÍNH

| Thuộc tính | Chi tiết |
|---|---|
| **Thư viện** | Chỉ cần `numpy` — không cần cài PyTorch |
| **Model format** | `.npz` (NumPy compressed archive) |
| **Agent** | `agent/a2c_numpy.py` — forward/backward pass viết tay bằng NumPy |
| **Features** | GAE (λ=0.9), Entropy Decay, Advantage Normalization, Gradient Clipping |
| **Khi nào dùng** | **Mặc định** — chạy trên mọi máy, không cần GPU |
| **Ưu điểm** | Nhẹ, nhanh, không phụ thuộc PyTorch, hiểu rõ backward pass |
| **Nhược điểm** | Không hỗ trợ GPU acceleration |

```bash
# Training với NumPy agent (KHUYẾN NGHỊ)
python train/train_numpy.py --instance ft06 --episodes 2500

# Output: results/model_ft06.npz
```

**Xem model đã train:**

```bash
python check_model/diagnostic.py results/training_log_ft06.csv   
python check_model/view_npz.py results/model_ft06.npz 
```

### 📦 `train.py` + `a2c_agent.py` — PHIÊN BẢN PHỤ (PyTorch)

| Thuộc tính | Chi tiết |
|---|---|
| **Thư viện** | Cần cài `torch` (~2GB download) |
| **Model format** | `.pt` (PyTorch state_dict) |
| **Agent** | `agent/a2c_agent.py` — dùng `torch.nn.Module`, autograd tự tính gradient |
| **Features** | A2C cơ bản (vanilla), không có GAE |
| **Khi nào dùng** | Khi muốn dùng GPU hoặc mở rộng sang PPO/SAC |
| **Ưu điểm** | Tận dụng GPU, dễ mở rộng, autograd tự động |
| **Nhược điểm** | Cần cài PyTorch (~2GB), nặng hơn |

```bash
# Cài PyTorch trước
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Training với PyTorch agent
python train/train.py --instance ft06 --episodes 2000

# Output: results/model_ft06.pt
```

### So sánh trực tiếp

| Tiêu chí | `train_numpy.py` (NumPy) | `train.py` (PyTorch) |
|---|---|---|
| **Agent file** | `agent/a2c_numpy.py` | `agent/a2c_agent.py` |
| **Model file** | `.npz` | `.pt` |
| **Dependency** | `numpy` only | `torch` (~2GB) |
| **GPU support** | ❌ CPU only | ✅ CUDA GPU |
| **GAE** | ✅ λ=0.9 | ❌ Không |
| **Entropy Decay** | ✅ 0.05 → 0.01 | ❌ Cố định |
| **Gradient Clipping** | ✅ norm=0.5 | ❌ Không |
| **Backward pass** | Viết tay (NumPy) | Tự động (autograd) |
| **Tốc độ (CPU)** | Nhanh hơn ~30% | Chậm hơn (overhead torch) |
| **Tốc độ (GPU)** | N/A | Nhanh hơn nhiều |
| **Kết quả FT06** | **Best = 59** | Best ≈ 63 |
| **Dùng cho ĐATN** | ✅ **CHÍNH** | Tham khảo |

> **Khuyến nghị:** Dùng `train_numpy.py` cho tất cả thí nghiệm. Chỉ dùng `train.py` nếu muốn mở rộng sang PPO hoặc có GPU mạnh.

---

## ⚙️ Cài đặt

### Yêu cầu

- Python 3.9+
- Các thư viện trong `requirements.txt`

### Cài đặt

```bash
cd jssp_a2c
pip install -r requirements.txt
```

Hoặc cài thủ công:

```bash
# Bắt buộc
pip install flask numpy matplotlib scipy seaborn pandas

# Tùy chọn (phiên bản PyTorch)
pip install torch
```

---

## 🚀 Cách chạy

### 1. Web App (Flask)

```bash
python main.py
```

Mở trình duyệt: **http://localhost:5000**

Giao diện gồm 3 tab:

- **Tab 1 — Visualization**: Chọn instance + algorithm → xem Gantt chart animation
- **Tab 2 — Evaluation**: So sánh tất cả algorithms, xếp hạng theo tier
- **Tab 3 — Real-world**: Demo xếp lịch dạy cho giáo viên (ứng dụng thực tế)

### 2. Train Model A2C

```bash
# NumPy (khuyến nghị) — auto-recipe theo instance
python train/train_numpy.py --instance ft06

# PyTorch (tùy chọn)
python train/train.py --instance ft06 --episodes 2000
```

**Auto-recipe** (nếu không truyền tham số):

| Instance | Episodes | Hidden | lr_actor | Eval Every |
|----------|----------|--------|----------|------------|
| 3×3 | 1000 | 128 | 3e-4 | 100 |
| 4×4 | 1500 | 128 | 3e-4 | 100 |
| 5×5 | 2000 | 256 | 3e-4 | 200 |
| **FT06** | **2500** | **256** | **3e-4** | **200** |
| FT10 | 5000 | 256 | 3e-4 | 500 |

**Tham số tùy chỉnh** (chỉ `train_numpy.py`):

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--instance` | `ft06` | Instance: `3x3`, `4x4`, `5x5`, `ft06`, `ft10` |
| `--episodes` | Auto | Số episode huấn luyện |
| `--hidden` | Auto | Số neuron mỗi lớp ẩn |
| `--lr_actor` | Auto | Learning rate Actor |
| `--lr_critic` | `5e-4` | Learning rate Critic |
| `--gamma` | `0.99` | Discount factor |
| `--gae` | `0.9` | GAE lambda |
| `--entropy_min` | `0.01` | Entropy floor |
| `--eval_every` | Auto | Eval model mỗi N episodes |
| `--n_eval_runs` | `30` | Số lần eval cuối cùng |
| `--seed` | `42` | Random seed |

**Output:**

```
results/
├── model_ft06.npz              ← Model weights
├── training_log_ft06.csv       ← Log từng episode (append)
├── evaluation_ft06.csv         ← So sánh algorithms (append)
└── schedule_ft06_{algo}.csv    ← Lịch sản xuất tốt nhất
```

> **CSV append mode:** Mỗi lần train tạo `run_id` mới, ghi thêm vào CSV (không xóa data cũ). Deterministic baselines ghi 1 row; stochastic ghi `n_eval_runs` rows.

### 3. Đánh giá

```bash
python evaluation/evaluate.py \
    --instance ft06 \
    --model_path results/model_ft06.npz \
    --n_runs 30
```

### 4. Diagnostic

```bash
python diagnostic.py results/training_log_ft06.csv   # Learning curves
python view_npz.py results/model_ft06.npz --limit 10  # Xem weights
```

### 5. Jupyter Notebook

```bash
jupyter notebook notebooks/DATN_Analysis.ipynb
```

---

## 🧠 Kiến trúc kỹ thuật

### MDP — Mô hình hóa JSSP

| Thành phần | Mô tả |
|---|---|
| **State** | Vector `3N + M` chiều: 3 features/job + 1 feature/machine, normalize [0,1] |
| **Action** | `Discrete(N)` — chọn job_id, action masking cho invalid actions |
| **Reward** | Hybrid: `-1/step` + `-0.5×idle_ratio` + `+10/job_done` + `+50×efficiency` |

### Actor-Critic Network (NumPy agent)

```
Actor:  State(3N+M) → Linear(256) → ReLU → Linear(256) → ReLU → Softmax(N)
Critic: State(3N+M) → Linear(256) → ReLU → Linear(256) → ReLU → Scalar V(s)
```

**Cải tiến so với vanilla A2C:**

| Feature | Mô tả | Nguồn |
|---|---|---|
| **GAE (λ=0.9)** | Generalized Advantage Estimation — giảm variance gradient | Schulman et al. (2016) |
| **Entropy Decay** | 0.05 → 0.01 — explore → exploit | Mnih et al. (2016), A3C |
| **Gradient Clipping** | L2 norm = 0.5 — chống exploding gradient | Standard RL practice |
| **Action Masking** | Invalid → logits = −∞ → prob ≈ 0 | Huang & Ontañón (2020) |
| **Orthogonal Init** | Weights init chuẩn cho RL | PPO paper |
| **Rolling Eval** | Save model dựa trên eval mean, không 1 episode lucky | Practical engineering |

### Baselines

| Algorithm | Nguyên lý | Loại |
|---|---|---|
| **FIFO** | Job index nhỏ nhất (đến trước phục vụ trước) | Deterministic |
| **SPT** | Operation ngắn nhất hiện tại | Deterministic |
| **LPT** | Operation dài nhất hiện tại | Deterministic |
| **EDD** | Thời gian hoàn thành dự kiến sớm nhất | Deterministic |
| **Random** | Chọn ngẫu nhiên trong valid actions | Stochastic |

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

## 🔧 Hướng phát triển

- [ ] Nâng cấp từ A2C lên **PPO** (Proximal Policy Optimization)
- [ ] Dùng **Graph Neural Network** để biểu diễn state
- [ ] Mở rộng sang **Flexible Job-Shop** (nhiều lựa chọn máy)
- [ ] Tích hợp vào phần mềm ERP/MES thực tế
- [ ] Curriculum Learning: train từ instance nhỏ → lớn
