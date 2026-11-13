# Phase 1 Verification Report: ResNet-18 (64x64x1) Hardware Golden Model

## 1. Tổng quan Kiểm định & Kiến trúc Phần cứng (Executive Summary)

Tài liệu này mô tả chuẩn đối chiếu **Phase 1 Golden Model** của mạng **Modified ResNet-18** cho bài toán phân loại 9 lớp khuyết tật tấm Wafer **WM-811K** hướng tới triển khai phần cứng RTL trên FPGA **Xilinx Zynq-7020 (PYNQ-Z2)**.

### 1.1. Bảng So sánh Độ chính xác giữa các Mô hình (Accuracy Benchmark)

*(Bảng dưới đây được tự động cập nhật số liệu thực nghiệm khi chạy `04_bit_accurate_emulator.py`)*

| Mô hình Đánh giá | Định dạng Trọng số / Activation | Phép toán Thực thi | Test Accuracy (%) | Macro F1-Score (%) | Độ lệch so với FP32 |
|:---|:---:|:---|:---:|:---:|:---:|
| **1. ResNet-18 FP32 (BatchNorm Folded)** | FP32 / FP32 | IEEE 754 Floating-Point | *Chạy `02_...` để cập nhật* | *-* | Baseline (0.00%) |
| **2. PyTorch Symmetric Quantized** | INT8 / INT8 | Dequantized Float Sim | *Chạy `03_...` để cập nhật* | *-* | *< 0.30%* |
| **3. Bit-Accurate Hardware Emulator** | INT8 / INT8 (Bias INT32) | Pure Integer (`int32`/`int64` + `>>`) | *Chạy `04_...` để cập nhật* | *-* | *< 0.50%* |

---

## 2. Đặc tả Kỹ thuật Đã Áp dụng trong Phase 1

### 2.1. Tiền xử lý Bảo toàn Vật lý (`01_wm811k_preprocess.py`)
- **Nội suy Nearest-Neighbor (`cv2.INTER_NEAREST`) về $64 \times 64$:** Loại bỏ hoàn toàn hiện tượng sinh giá trị thập phân giả giữa die lành và die lỗi.
- **Định dạng đầu vào 1 kênh INT8 ($64 \times 64 \times 1$):** Tiết kiệm **66.7% dung lượng BRAM** ở tầng đầu vào so với ảnh RGB 3 kênh.
- **Ánh xạ dải động dương INT8:**
  - `0` (Background / Ngoài wafer) $\rightarrow$ `0` (`0x00`)
  - `1` (Normal / Passing Die) $\rightarrow$ `64` (`0x40`)
  - `2` (Defective / Failing Die) $\rightarrow$ `127` (`0x7F`)

### 2.2. Kiến trúc Modified ResNet-18 & BatchNorm Folding (`02_resnet18_Modified_train.py`)
- **Thay thế `Conv1`:** Chuyển từ `Conv2d(3, 64, 7x7, s=2)` sang `Conv2d(1, 64, 3x3, s=1, p=1, bias=False)` và **xóa bỏ `MaxPool2d`** ngay sau `Conv1` để giữ nguyên độ phân giải $64 \times 64$ ở Stage 1, bảo toàn các đường xước mảnh (`Scratch`).
- **Luồng kích thước Feature Map:**
  - `Stem (Conv1)` + `Stage 1` (2 BasicBlocks, 64 kênh): $64 \times 64$
  - `Stage 2` (2 BasicBlocks, 128 kênh, $s=2$): $32 \times 32$
  - `Stage 3` (2 BasicBlocks, 256 kênh, $s=2$): $16 \times 16$
  - `Stage 4` (2 BasicBlocks, 512 kênh, $s=2$): $8 \times 8$
  - `Global Average Pooling` ($8 \times 8$): Cộng tổng 64 phần tử và dịch phải 6 bit (`>> 6`).
  - `FC Layer`: $512 \rightarrow 9$ lớp khuyết tật.
- **BatchNorm Folding:** Gộp toàn bộ tham số $(\gamma, \beta, \mu, \sigma^2)$ của 20 tầng `BatchNorm2d` vào thẳng $W_{\text{fold}}$ và $b_{\text{fold}}$ của 20 tầng `Conv2d` đứng trước:
  $$W_{\text{fold}} = W \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}, \quad b_{\text{fold}} = (b - \mu) \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

### 2.3. Lượng tử hóa Đối xứng & Căn chỉnh Nhánh Tắt (`03_quantizer_export.py`)
- **Trọng số (INT8) & Hệ số chệch (INT32):**
  $$S_w = \frac{\max(|W_{\text{fold}}|)}{127}, \quad q_w = \text{clip}\left(\text{round}\left(\frac{W_{\text{fold}}}{S_w}\right), -127, 127\right)$$
  $$S_b = S_x \cdot S_w, \quad q_b = \text{round}\left(\frac{b_{\text{fold}}}{S_x \cdot S_w}\right)$$
- **Tái lượng tử hóa Nhân + Dịch phải (Multiplier + Bit-Shift):**
  $$M_{\text{real}} = \frac{S_x \cdot S_w}{S_y} \approx \frac{M_0}{2^n}, \quad q_y = \text{clip}\left((\text{acc} \cdot M_0 + 2^{n-1}) \gg n, -128, 127\right)$$
- **Căn chỉnh Nhánh Cộng Tắt (Residual Shortcut Alignment):**
  $$q_{\text{block\_out}} = \text{ReLU}\left( \text{clip}\left( \left[(\text{acc}_{\text{conv2}} \cdot M_{\text{conv2}} + 2^{n_1-1}) \gg n_1\right] + \left[(q_{\text{shortcut}} \cdot M_{\text{sc}} + 2^{n_2-1}) \gg n_2\right], \ 0, \ 127 \right) \right)$$

---

## 3. Giao thức Sắp xếp Bộ nhớ Phần cứng (Memory Layout Protocol)

| Loại Vector Phần cứng | Đường dẫn File | Thứ tự Làm phẳng (Flatten Order) | Định dạng Dòng `.hex` |
|:---|:---|:---|:---|
| **Input Wafer Samples** | `hw_vectors/test_samples/input_wafer_*.hex` | `NHWC`: `[64, 64, 1]` | 2 ký tự Hex / dòng (INT8 bù 2) |
| **Golden Feature Maps** | `hw_vectors/golden_fmaps/layer*_out.hex` | `NHWC`: `[H, W, C]` | 2 ký tự Hex / dòng (INT8 bù 2) |
| **Quantized Weights** | `hw_vectors/weights_hex/layer*_w.hex` | Tiling: `[Out_G, In_G, Kh, Kw, Tm, Tn]` | 2 ký tự Hex / dòng (INT8 bù 2) |
| **Quantized Biases** | `hw_vectors/bias_hex/layer*_b.hex` | Tuyến tính: `[C_out]` | 8 ký tự Hex / dòng (INT32 bù 2) |
| **DDR3 Packed Binary** | `hw_vectors/ddr_weights_pack.bin` | Nối tiếp 21 tầng theo Tiling `(Tm, Tn)` | Raw binary stream (`int8`) |

---

