# FPGA-Based Hardware Accelerator for Modified ResNet-18
## Phân loại Khuyết tật Tấm Wafer Bán dẫn (Semiconductor Wafer Map Defect Classification) trên FPGA Xilinx Zynq-7020 (PYNQ-Z2)

---

## 1. Giới thiệu Đề tài & Mục tiêu Nghiên cứu

Trong quy trình sản xuất mạch tích hợp (Semiconductor IC Manufacturing), việc phát hiện và phân loại tự động các dạng khuyết tật bề mặt tấm wafer (**Wafer Map Defect Pattern Recognition - WM-DPR**) đóng vai trò quyết định trong việc tối ưu hóa tỷ lệ thành phẩm (Yield Optimization) và truy vết nguyên nhân lỗi thiết bị trong dây chuyền.

Đề tài tập trung nghiên cứu, thiết kế và hiện thực hóa một **Bộ tăng tốc phần cứng chuyên dụng (Dedicated Hardware Accelerator)** trên nền tảng **FPGA Xilinx Zynq-7020 (kit PYNQ-Z2)**, áp dụng phương pháp **Đồng thiết kế Phần cứng - Phần mềm (Hardware-Software Co-Design)** cho mạng nơ-ron tích chập **Modified ResNet-18**:
- **Tập dữ liệu chuẩn:** **WM-811K** gồm 811,457 tấm wafer thu thập từ 46,393 lô sản xuất thực tế.
- **9 lớp lỗi phân loại:** `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Random`, `Scratch`, `Near-full`, và `none` (bình thường).
- **Mục tiêu phần cứng:** Tối ưu hóa suy luận thời gian thực (Real-time Inference), triệt tiêu độ trễ tính toán, tiết kiệm tối đa tài nguyên on-chip (BRAM, DSP48E1) và tối ưu hiệu suất năng lượng (Performance/Watt) so với các giải pháp GPU/CPU thông thường.

---

## 2. Kiến trúc Kỹ thuật Cải tiến (Hardware-Friendly Deep Learning)

Mô hình ResNet-18 chuẩn được tinh chỉnh đặc thù nhằm đáp ứng các giới hạn tài nguyên khắt khe trên chip FPGA Zynq-7020:

```text
Wafer Input (64x64x1 INT8)
   │
   ▼
[ Stem: Conv2d (3x3, s=1, p=1, 64-ch) ]  <-- Xóa MaxPool, bảo toàn độ phân giải 64x64
   │
   ├──► [ Stage 1: 2x BasicBlock (64-ch,  64x64) ]  <-- Giữ nguyên vết xước mảnh (Scratch)
   ├──► [ Stage 2: 2x BasicBlock (128-ch, 32x32, s=2) ]
   ├──► [ Stage 3: 2x BasicBlock (256-ch, 16x16, s=2) ]
   ├──► [ Stage 4: 2x BasicBlock (512-ch,  8x8,  s=2) ]
   │
   ▼
[ Global Average Pooling (8x8) ]  <-- Cộng 64 phần tử và dịch phải 6-bit (>> 6)
   │
   ▼
[ Fully-Connected Layer (512 -> 9) ]
   │
   ▼
Dự đoán 9 Lớp Khuyết tật Wafer
```

### Các giải pháp tối ưu hóa phần cứng trọng tâm:

| Giải pháp Kỹ thuật | Cơ chế Thực hiện | Lợi ích Phần cứng (Hardware Impact) |
| :--- | :--- | :--- |
| **Bảo toàn vật lý đầu vào ($64 \times 64 \times 1$)** | Nội suy Nearest-Neighbor (`cv2.INTER_NEAREST`), ánh xạ $\{0, 1, 2\} \rightarrow \{0, 64, 127\}$. | Loại bỏ giá trị thập phân giả mạo; tiết kiệm **66.7% dung lượng BRAM đầu vào** so với ảnh RGB 3 kênh. |
| **Tinh chỉnh tầng Stem (Conv1)** | Chuyển sang `Conv2d(1, 64, 3x3, s=1, p=1)` và xóa bỏ hoàn toàn tầng `MaxPool2d` ($3 \times 3, s=2$). | Giữ nguyên độ phân giải $64 \times 64$ suốt Stage 1, ngăn chặn việc làm mờ/mất các đường khuyết tật siêu mảnh (`Scratch`). |
| **BatchNorm Folding** | Hợp nhất tham số $(\gamma, \beta, \mu, \sigma^2)$ của 20 tầng BatchNorm vào trọng số và bias của tầng Conv đứng trước: <br> $W_{\text{fold}} = W \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}$, $b_{\text{fold}} = (b - \mu) \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} + \beta$ | Triệt tiêu hoàn toàn 20 tầng BatchNorm trong quá trình suy luận; loại bỏ phép căn bậc hai và phép chia dấu phẩy động trên FPGA. |
| **Lượng tử hóa Đối xứng INT8/INT32** | Trọng số lượng tử hóa INT8 $[-127, 127]$, Bias tích lũy INT32 (`acc`). Tái lượng tử hóa bằng phép nhân $M_0$ và dịch phải: <br> $q_y = \text{clip}\left(\left(\text{acc} \cdot M_0 + 2^{n-1}\right) \gg n, \ 0, \ 127\right)$ | Ánh xạ 100% vào các khối DSP48E1; không sử dụng bộ chia; đồng bộ hóa nhánh cộng tắt (Residual Shortcut Alignment) hoàn toàn bằng số nguyên. |
| **Global Average Pooling không dùng chia** | Feature map cuối có kích thước $8 \times 8 = 64$ phần tử. | Phép chia cho 64 được thay thế bằng bộ cộng tích lũy 64 giá trị và phép dịch phải 6 bit (`>> 6`). |

---

## 3. Kết quả Kiểm định Thực nghiệm Phase 1 (Verification Benchmark)

Độ chính xác của mô hình được đối chiếu thực nghiệm độc lập trên **10,800 mẫu Wafer Test**:

| Mô hình Đánh giá | Định dạng Dữ liệu | Phép toán Thực thi | Test Accuracy (%) | Macro F1-Score (%) | Độ lệch Suy giảm |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **1. ResNet-18 FP32 (BatchNorm Folded)** | FP32 / FP32 | IEEE 754 Floating-Point | **97.55%** | 97.53% | Chuẩn đối chiếu (0.00%) |
| **2. PyTorch Symmetric Quantized** | INT8 / INT8 | Dequantized Float Sim | **97.55%** | 97.53% | +0.00% |
| **3. Bit-Accurate Hardware Emulator** | INT8 / INT8 (Bias INT32) | Pure Integer (`int32` + `>>`) | **97.55%** | **97.53%** | **+0.00%** |

### Kết quả Phân loại Chi tiết theo 9 Lớp Lỗi (Bit-Accurate Integer Emulator):

| Lớp Khuyết tật (Class) | Precision | Recall | F1-Score | Số lượng Mẫu Test (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Center** | 0.9713 | 0.9867 | 0.9789 | 1,200 |
| **Donut** | 0.9917 | 1.0000 | 0.9959 | 1,200 |
| **Edge-Loc** | 0.9550 | 0.9367 | 0.9457 | 1,200 |
| **Edge-Ring** | 0.9826 | 0.9908 | 0.9867 | 1,200 |
| **Loc** | 0.9476 | 0.9350 | 0.9413 | 1,200 |
| **Random** | 0.9893 | 1.0000 | 0.9946 | 1,200 |
| **Scratch** | 0.9810 | 0.9892 | 0.9851 | 1,200 |
| **Near-full** | 1.0000 | 1.0000 | 1.0000 | 1,200 |
| **none** | 0.9592 | 0.9408 | 0.9499 | 1,200 |
| **Độ chính xác Trung bình (Macro Avg)** | **0.9753** | **0.9755** | **0.9753** | **10,800** |

> **Nghiệm thu Phase 1 (RTL Readiness Sign-off):** Bộ mô phỏng phần cứng số nguyên thuần túy đạt độ khớp 100% về độ chính xác so với mô hình FP32, hoàn tất xuất toàn bộ 21 Feature Map chuẩn (`golden_fmaps/`) và bộ thông số phần cứng phục vụ kiểm thử RTL.

---

## 4. Lộ trình Triển khai Dự án theo Giai đoạn (Project Roadmap)

### 4.1. Bảng Tổng quan Tiến độ 4 Giai đoạn

| Giai đoạn (Phase) | Tên Giai đoạn | Trọng tâm Kỹ thuật | Trạng thái | Tiêu chí Hoàn thành (Deliverables) |
| :---: | :--- | :--- | :---: | :--- |
| **Phase 1** | Software Golden Model, Quantization & Verification | Tiền xử lý, huấn luyện, gộp BN, lượng tử hóa INT8/INT32, xuất Golden Hex | **Hoàn thành** | Đạt Accuracy 97.55%; xuất 21 Golden Feature Maps, file cấu hình `quant_config.vh`, và nhị phân DDR `ddr_weights_pack.bin`. |
| **Phase 2** | Hardware Architecture & RTL Design | Thiết kế PE Array, Re-quantizer, Line Buffer, Controller FSM | **Đang thực hiện** | Hoàn thành mã nguồn Verilog cho toàn bộ các khối xử lý và đệm dữ liệu trong `hardware/RTL/`. |
| **Phase 3** | Hardware Verification & Simulation | Viết Testbench kiểm định từng module và từng tầng so khớp Golden Hex | **Kế tiếp** | 100% chu kỳ và dữ liệu đầu ra của RTL khớp chính xác từng bit so với Golden Feature Maps. |
| **Phase 4** | SoC Integration & On-Board Deployment | Tích hợp AXI DMA, AXI-Lite trên Vivado, nạp bitstream lên kit PYNQ-Z2 | **Kế tiếp** | Chạy phân loại wafer trực tiếp từ Python PYNQ API; đo đạc thông lượng (FPS), độ trễ (latency), và công suất tiêu thụ. |

---

### 4.2. Bảng Chi tiết Task theo Từng Phase

#### Phase 1: Software Golden Model, Quantization & Verification (Hoàn thành)

| Mã Task | Tên Nhiệm vụ | File / Module Thực thi | Mô tả Kỹ thuật Chi tiết | Trạng thái |
| :---: | :--- | :--- | :--- | :---: |
| **T1.1** | Tiền xử lý Dữ liệu Wafer | `software/preprocessing/01_wm811k_preprocess.py` | Lọc 9 lớp từ `LSWMD.pkl`, resize Nearest-Neighbor $64 \times 64$, ánh xạ INT8 $\{0, 64, 127\}$, cân bằng 8,000 mẫu/lớp, xuất `wm811k_64x64_int8.npz` và 9 file mẫu `input_wafer_*.hex`. | Đã hoàn thành |
| **T1.2** | Huấn luyện & BatchNorm Folding | `software/Training_Model/02_resnet18_Modified_train.py` | Huấn luyện Modified ResNet-18 (Conv1 $3 \times 3$, bỏ MaxPool); gộp toàn bộ tham số 20 tầng BatchNorm vào Conv weights và bias. | Đã hoàn thành |
| **T1.3** | Lượng tử hóa & Xuất Vector Phần cứng | `software/Low_Precision_Model/03_quantizer_export.py` | Lượng tử hóa đối xứng trọng số INT8, bias INT32; tính toán hệ số scale/shift $(M_0, n)$; tiling $(T_m=16, T_n=16)$; xuất `weights_hex/`, `bias_hex/`, `quant_config.vh`, `ddr_weights_pack.bin`. | Đã hoàn thành |
| **T1.4** | Mô phỏng Phần cứng Bit-Accurate | `software/Explicit_Model/04_bit_accurate_emulator.py` | Giả lập số nguyên thuần túy; đánh giá trên 10,800 mẫu test đạt 97.55% accuracy; xuất 21 Golden Feature Maps chuẩn NHWC và lập báo cáo `Phase1_Verification_Report.md`. | Đã hoàn thành |

#### Phase 2: Hardware Architecture & RTL Design (Đang thực hiện)

| Mã Task | Tên Nhiệm vụ | Module RTL Mục tiêu | Mô tả Kỹ thuật Chi tiết | Trạng thái |
| :---: | :--- | :--- | :--- | :---: |
| **T2.1** | Processing Element & PE Array | `hardware/RTL/pe_unit.v`<br>`hardware/RTL/pe_array.v` | Thiết kế PE cơ sở (nhân số nguyên INT8 có dấu, tích lũy INT32, ánh xạ vào DSP48E1). Thiết kế mảng tính toán $T_m \times T_n = 16 \times 16$ (256 phép tính song song). | Đang triển khai |
| **T2.2** | Khối Re-quantization & Residual Alignment | `hardware/RTL/requant_unit.v` | Hiện thực hóa bộ nhân hệ số cố định $M_0$, làm tròn cộng $2^{n-1}$, dịch phải bit (`>> n`), khối cộng nhánh tắt (Shortcut Adder) và hàm kích hoạt ReLU/clipping $[0, 127]$. | Đang triển khai |
| **T2.3** | Khối GAP & Fully-Connected | `hardware/RTL/gap_fc_unit.v` | Bộ tích lũy không gian $8 \times 8$ (64 chu kỳ) kết hợp bộ dịch phải 6 bit (`>> 6`). Khối tính toán tầng FC tính điểm cho 9 lớp khuyết tật đầu ra. | Kế tiếp |
| **T2.4** | Hệ thống Bộ đệm On-chip | `hardware/RTL/line_buffer.v`<br>`hardware/RTL/ping_pong_buffer.v` | Line Buffer trượt 3 dòng cấp đồng thời cửa sổ $3 \times 3$ cho PE Array. Hệ thống đệm đôi (Ping-Pong BRAM) che giấu độ trễ đọc dữ liệu từ bộ nhớ DDR. | Kế tiếp |
| **T2.5** | Bộ điều khiển Gia tốc FSM | `hardware/RTL/accel_controller.v` | Máy trạng thái hữu hạn (FSM) quản lý tuần tự quá trình nạp tham số, nạp feature map, kích hoạt tính toán và lưu trữ kết quả qua từng tầng mạng. | Kế tiếp |

#### Phase 3: Hardware Verification & Simulation (Kế tiếp)

| Mã Task | Tên Nhiệm vụ | Kịch bản Kiểm thử | Tiêu chuẩn Đánh giá & Nghiệm thu | Trạng thái |
| :---: | :--- | :--- | :--- | :---: |
| **T3.1** | Unit Testbench | `hardware/TB/tb_pe_array.v`<br>`hardware/TB/tb_requant.v`<br>`hardware/TB/tb_line_buffer.v` | Kiểm tra tính đúng đắn về chức năng độc lập của từng module thành phần dưới các trường hợp biên (overflow, underflow, timing). | Kế tiếp |
| **T3.2** | Layer-Level Golden Verification | `hardware/TB/tb_layer_verification.v` | Đọc dữ liệu từ `input_wafer_*.hex`, `weights_hex/`, `bias_hex/`; tự động so sánh bit-by-bit kết quả đầu ra RTL với `golden_fmaps/layer*.hex`. Khớp chính xác 100%. | Kế tiếp |
| **T3.3** | Top-Level Pipeline Simulation | `hardware/Simulation/run_sim.tcl` | Mô phỏng toàn bộ chu trình suy luận end-to-end từ ảnh đầu vào đến kết quả 9 lớp lỗi trên Vivado Simulator (xsim) hoặc QuestaSim. | Kế tiếp |

#### Phase 4: SoC Integration & On-Board Deployment (Hoàn thiện)

| Mã Task | Tên Nhiệm vụ | Giao tiếp / Môi trường | Mô tả Kỹ thuật Chi tiết | Trạng thái |
| :---: | :--- | :--- | :--- | :---: |
| **T4.1** | Đóng gói IP Giao tiếp AXI | `hardware/SoC/axi_wrapper.v` | Tích hợp AXI4-Stream kết hợp AXI DMA cho luồng truyền nhận dữ liệu lớn (ảnh wafer, trọng số); tích hợp AXI4-Lite cho các thanh ghi cấu hình và điều khiển FSM. | Kế tiếp |
| **T4.2** | Block Design & Bitstream | Xilinx Vivado (Zynq-7020) | Kết nối IP Accelerator với ARM Cortex-A9 Processing System (PS); tổng hợp (Synthesis), thực thi (Place & Route), tối ưu Timing Closure và tạo bitstream `.bit`. | Kế tiếp |
| **T4.3** | Triển khai trên kit PYNQ-Z2 | Python PYNQ API / Jupyter Notebook | Nạp bitstream và file `ddr_weights_pack.bin` vào bộ nhớ; chạy suy luận trên các tấm wafer thực tế; đo đạc độ trễ (latency), FPS, và công suất tiêu thụ thực tế. | Kế tiếp |

---

## 5. Cấu trúc Thư mục Dự án

```text
RESNET-18-FPGA/
├── docs/                                 # Tài liệu kỹ thuật & Báo cáo thực nghiệm
│   ├── Phase1_Verification_Report.md     # Báo cáo kiểm định Golden Model Phase 1
│   └── Phase1_Result.txt                 # Nhật ký thực thi thực nghiệm chi tiết
├── hardware/                             # Thiết kế phần cứng & Kiểm định RTL
│   ├── RTL/                              # Mã nguồn Verilog/SystemVerilog của Accelerator
│   ├── TB/                               # Testbench kiểm tra chức năng từng khối và từng tầng
│   ├── Simulation/                       # Kịch bản mô phỏng và lưu trữ waveform
│   └── SoC/                              # Thiết kế tích hợp hệ thống AXI trên Zynq-7020
├── software/                             # Pipeline Phần mềm, Huấn luyện & Lượng tử hóa
│   ├── preprocessing/                    # Tiền xử lý dữ liệu WM-811K (Nearest-Neighbor INT8)
│   │   ├── 01_wm811k_preprocess.py
│   │   └── preprocess_data.ipynb
│   ├── Training_Model/                   # Huấn luyện FP32 & BatchNorm Folding
│   │   ├── 02_resnet18_Modified_train.py
│   │   └── checkpoints/
│   ├── Low_Precision_Model/              # Lượng tử hóa INT8/INT32 & Đóng gói tham số
│   │   ├── 03_quantizer_export.py
│   │   ├── checkpoints/
│   │   └── hw_vectors/                   # weights_hex/, bias_hex/, quant_config.vh, ddr_pack.bin
│   └── Explicit_Model/                   # Bộ mô phỏng Bit-Accurate thuần số nguyên
│       ├── 04_bit_accurate_emulator.py
│       └── hw_vectors/golden_fmaps/      # 21 file Golden Feature Maps (.hex) làm chuẩn RTL
├── requirement.txt                       # File alias cấu hình thư viện
├── requirements.txt                      # Danh mục các thư viện Python phụ thuộc
└── README.md                             # Tài liệu kỹ thuật tổng quan dự án
```
