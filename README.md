# Project 1 - Search và Nature-Inspired Optimization

Dự án triển khai các thuật toán tối ưu lấy cảm hứng tự nhiên, thuật toán tìm kiếm cổ điển, và một giao diện UI (Tkinter) để chạy toàn bộ pipeline theo yêu cầu đề bài.

## 1. Yêu cầu hệ thống

- Python 3.10+ (khuyến nghị 3.10 hoặc 3.11 để tránh lỗi cài `matplotlib` từ source)
- Hệ điều hành: Windows/Linux/macOS
- Có `pip`

## 2. Cài đặt môi trường

### Cách nhanh (khuyến nghị)

Trong thư mục gốc dự án:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo:

- PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

- CMD:

```cmd
.venv\Scripts\activate.bat
```

Cài dependencies:

```bash
pip install -r requirements.txt
```

Nếu thiếu thư viện vẽ, cài thêm thủ công:

```bash
pip install numpy matplotlib
```

## 3. Chạy UI

Chạy lệnh sau ở thư mục gốc dự án:

```bash
python main.py
```

UI sẽ mở với 3 tab chính:

- `Core Workflow`
- `Requirement Tools`
- `One Click`

## 4. Hướng dẫn sử dụng UI

### Tab `Core Workflow`

- `Step 1: Required Algorithms + Baseline Compare`
: Chạy so sánh thuật toán chính + baseline BFS/DFS/UCS/Greedy/A*/HC/SA.

- `Step 2: Required Visualizations`
: Hiển thị trực quan cụ thể (convergence, diversity, 3D landscape, SA/HC) và tạo hình local search.

- `Step 3: Required Metrics`
: Chạy thống kê, benchmark requirement (quality, time, memory, robustness, scalability, exploration/exploitation).

- `Step 4: Sensitivity + Hypothesis + Discrete`
: Chạy phân tích độ nhạy tham số, kiểm định giả thuyết, và bài toán TSP rời rạc.

- `Run Core Step 1 -> 4`
: Chạy toàn bộ 4 bước theo đúng thứ tự.

### Tab `Requirement Tools`

- `Generate SA/HC Convergence Figures`
- `Requirement Benchmark (Quick)`
- `Requirement Benchmark (Full)`
- `Parameter Sensitivity`
- `Hypothesis Test (HC vs SA)`

Tab này hữu ích khi bạn muốn chạy từng phần riêng lẻ để lấy số liệu/hình riêng.

### Tab `One Click`

- `Run All For Report`
: Chạy pipeline tổng hợp để lấy dữ liệu phục vụ viết báo cáo.

## 5. Output được sinh ở đâu

Hình ảnh được lưu trong:

- `visualizes/`

Ví dụ:

- `evolution_convergence.png`
- `swarm_convergence.png`
- `tlbo_convergence.png`
- `local_search_sa_convergence.png`
- `local_search_hc_convergence.png`
- `requirements_convergence.png`
- `requirements_robustness.png`
- `sensitivity_hc_step_size.png`
- `sensitivity_sa_cooling_rate.png`

Bảng số liệu được lưu trong:

- `experiments/results/`

Ví dụ:

- `continuous_metrics.csv`
- `scalability_metrics.csv`
- `exploration_metrics.csv`
- `discrete_shortest_path_metrics.csv`
- `sensitivity_hc_step_size.csv`
- `sensitivity_sa_cooling_rate.csv`

## 6. Chạy script trực tiếp (không qua UI)

- Benchmark requirement:

```bash
python -m experiments.benchmark_requirements
```

- Sensitivity:

```bash
python -m experiments.parameter_sensitivity
```

- Hypothesis test:

```bash
python -m experiments.hypothesis_test
```

## 7. Lỗi thường gặp

- Lỗi `No module named matplotlib`
: Cài lại bằng `pip install matplotlib` trong đúng môi trường đang chạy.

- Lỗi import local package (`No module named problems`, `algorithms`, ...)
: Đảm bảo bạn đang chạy lệnh tại thư mục gốc dự án (`Project1`).

- UI không mở
: Kiểm tra Python có hỗ trợ `tkinter`.

## 8. Cấu trúc chính của dự án

- `main.py`: UI điều khiển trung tâm.
- `algorithms/`: toàn bộ thuật toán.
- `problems/`: định nghĩa bài toán.
- `experiments/`: các script benchmark/phân tích.
- `visualization/`: hàm vẽ biểu đồ.
- `report/`: nội dung báo cáo LaTeX.
