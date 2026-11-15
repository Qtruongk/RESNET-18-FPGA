## 1. Cấu hình Mạng Modified ResNet-18 (Dataset WM-811K: 64x64x1)

| Tầng / Khối | Kích thước Feature Map | Cấu hình Kernel | Số kênh | Ghi chú điều chỉnh phần cứng |
| :--- | :---: | :---: | :---: | :--- |
| **Input** | $64 \times 64$ | — | **1 kênh** | INT8 `{0, 64, 127}`, bảo toàn 3 trạng thái die vật lý. |
| **Stem Conv1** | $64 \times 64$ | $3 \times 3$, stride=1, pad=1 | 64 | Thay cho $7 \times 7$, s=2 để giữ nguyên độ phân giải $64 \times 64$. |
| **MaxPool** | — | **ĐÃ XÓA BỎ** | — | Không dùng MaxPool để tránh làm mờ/biến mất các vết xước mảnh (`Scratch`). |
| **Stage 1** | $64 \times 64$ | 2 BasicBlocks ($3 \times 3$) | 64 | Stride=1, duy trì trọn vẹn độ phân giải $64 \times 64$. |
| **Stage 2** | $32 \times 32$ | 2 BasicBlocks ($3 \times 3$) | 128 | Stride=2 ở block đầu (có shortcut $1 \times 1$). |
| **Stage 3** | $16 \times 16$ | 2 BasicBlocks ($3 \times 3$) | 256 | Stride=2 ở block đầu (có shortcut $1 \times 1$). |
| **Stage 4** | $8 \times 8$ | 2 BasicBlocks ($3 \times 3$) | 512 | Stride=2 ở block đầu (có shortcut $1 \times 1$). |
| **Global Avg Pool** | $1 \times 1$ | Cửa sổ $8 \times 8$ (64 điểm) | 512 | Cộng 64 phần tử & dịch phải 6 bit (`>> 6`), không dùng bộ chia. |
| **FC Layer** | $1 \times 1$ | Tuyến tính ($512 \rightarrow 9$) | **9 lớp** | 9 lớp khuyết tật: *Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, none*. |

---

## 2. Bảng Chi tiết 9 Module RTL Chốt lại:

| STT | Tên Module RTL | Nhóm chức năng | Nhiệm vụ Kỹ thuật Cụ thể |
| :---: | :--- | :---: | :--- |
| **1** | `cnn_controller.v` | Điều khiển | **FSM mức cao nhất (Top-level):** Quản lý thứ tự chạy 4 Stage của ResNet-18, nạp các tham số $(M_0, n)$ từ `quant_config.vh`, cấp phát địa chỉ bộ nhớ và báo cờ `Done`. |
| **2** | `conv_engine.v` | Điều khiển | **Bộ điều khiển tích chập:** Quét các vòng lặp Tiling ($T_m=16, T_n=16$), duyệt cửa sổ $3 \times 3$ ($K_h, K_w$), sinh tín hiệu `en` và `clr_acc` cho mảng MAC. |
| **3** | `line_buffer.v` | Bộ nhớ | Đệm trượt 3 dòng ảnh Feature Map, cấp đồng thời cửa sổ dữ liệu $3 \times 3$ cho mảng MAC mà không cần đọc lặp lại nhiều lần từ RAM. |
| **4** | `weight_buffer.v` | Bộ nhớ | Bộ đệm On-chip BRAM chứa các Tile trọng số INT8 ($16 \times 16$) nạp từ DDR3. |
| **5** | `feature_map_buffer.v` | Bộ nhớ | Hệ thống đệm đôi (**Ping-Pong BRAM**) lưu giữ Feature Map đầu vào và đầu ra giữa các tầng mạng. |
| **6** | `mac_array.v` | Tính toán | **Mảng tính toán song song $16 \times 16$:** Chứa 256 khối `pe_unit.v`, thực thi 256 phép tính MAC INT8 $\times$ INT8 mỗi chu kỳ (dùng chung cho cả tầng Conv và tầng FC). |
| **7** | `requant_unit.v` | Hậu xử lý | **Tái lượng tử hóa & Kích hoạt:** Nhân hệ số $M_0$, làm tròn cộng $2^{n-1}$, dịch phải bit (`>> n`), cộng nhánh tắt Residual Shortcut và kẹp ngưỡng hàm kích hoạt ReLU $[0, 127]$ (chuyển đổi INT32 $\rightarrow$ INT8). |
| **8** | `global_avg_pool.v` | Hậu xử lý | Bộ cộng tích lũy 64 phần tử không gian $8 \times 8$ và dịch phải 6 bit (`>> 6`) để chuyển Feature Map từ $8 \times 8 \times 512 \rightarrow 1 \times 1 \times 512$. |
| **9** | `argmax.v` | Đầu ra | So sánh 9 điểm số đầu ra của tầng FC để tìm ra nhãn lớp khuyết tật có điểm số cao nhất (từ lớp 0 đến lớp 8). |

*(Lưu ý: Khối hạt nhân tính toán `pe_unit.v` là module con nằm bên trong `mac_array.v`)*

---

## 3. Thứ tự Thực hiện Thiết kế (Phương pháp Bottom-Up):
1. **Chặng 1 (Lõi tính toán):** `pe_unit.v` &rarr; `mac_array.v`
2. **Chặng 2 (Hậu xử lý INT32 &rarr; INT8):** `requant_unit.v` &rarr; `global_avg_pool.v` &rarr; `argmax.v`
3. **Chặng 3 (Hệ thống bộ đệm dữ liệu):** `line_buffer.v` &rarr; `weight_buffer.v` & `feature_map_buffer.v`
4. **Chặng 4 (Bộ điều khiển FSM toàn mạng):** `conv_engine.v` &rarr; `cnn_controller.v`

---

## 4. Lưu ý Tối ưu Phần cứng:
1. **BatchNorm Folding:** Toàn bộ 20 tầng Batch Normalization được gộp trực tiếp vào weights và bias của lớp convolution sau quá trình training. Do đó loại bỏ hoàn toàn module `batch_norm.v`, tiết kiệm tài nguyên FPGA.
2. **Loại bỏ MaxPool:** Mạng không dùng MaxPool sau Conv1 để bảo toàn đặc trưng vết xước mảnh, loại bỏ module `maxpool.v`.
3. **GAP không dùng bộ chia:** Tầng Global Average Pooling nhận kích thước $8 \times 8 = 64$ điểm ảnh, do đó phép tính trung bình được thay thế bằng phép dịch bit (`>> 6`), triệt tiêu việc sử dụng bộ chia số học.
