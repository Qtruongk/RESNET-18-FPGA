# Phase 1 Verification Report: ResNet-18 (64x64x1) Hardware Golden Model

## 1. Tổng quan Kiểm định (Executive Summary)

Báo cáo này tổng hợp kết quả kiểm định độ chính xác giữa **Mô hình chuẩn dấu phẩy động (FP32 Folded)**, **Mô hình Lượng tử hóa PyTorch (Simulated Quantized)** và **Bộ Mô phỏng Phần cứng Số nguyên Thuần túy (Bit-Accurate Integer Emulator)** trên tập dữ liệu kiểm thử **WM-811K** ($64 \times 64 \times 1$, INT8 `{0, 64, 127}`).

| Mô hình Đánh giá | Định dạng Trọng số / Activation | Phép toán Thực thi | Test Accuracy (%) | Macro F1-Score (%) | Độ lệch so với FP32 |
|:---|:---:|:---|:---:|:---:|:---:|
| **1. ResNet-18 FP32 (BatchNorm Folded)** | FP32 / FP32 | IEEE 754 Floating-Point | **97.55%** | 97.53% | Baseline (0.00%) |
| **2. PyTorch Symmetric Quantized** | INT8 / INT8 | Dequantized Float Sim | **97.55%** | - | +0.00% |
| **3. Bit-Accurate Hardware Emulator** | INT8 / INT8 (Bias INT32) | Pure Integer (`int32`/`int64` + `>>`) | **97.55%** | **97.53%** | **+0.00%** |

---

## 2. Báo cáo Phân loại Chi tiết theo 9 Lớp Lỗi (Bit-Accurate Emulator)

```text
precision    recall  f1-score   support

      Center     0.9713    0.9867    0.9789      1200
       Donut     0.9917    1.0000    0.9959      1200
    Edge-Loc     0.9550    0.9367    0.9457      1200
   Edge-Ring     0.9826    0.9908    0.9867      1200
         Loc     0.9476    0.9350    0.9413      1200
      Random     0.9893    1.0000    0.9946      1200
     Scratch     0.9810    0.9892    0.9851      1200
   Near-full     1.0000    1.0000    1.0000      1200
        none     0.9592    0.9408    0.9499      1200

    accuracy                         0.9755     10800
   macro avg     0.9753    0.9755    0.9753     10800
weighted avg     0.9753    0.9755    0.9753     10800
```

---

## 3. Ma trận Nhầm lẫn (Confusion Matrix - Bit-Accurate Emulator)

| Thực tế \ Dự đoán | Center | Donut | Edge-Loc | Edge-Ring | Loc | Random | Scratch | Near-full | none |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Center** | 1184 | 3 | 0 | 0 | 9 | 0 | 0 | 0 | 4 |
| **Donut** | 0 | 1200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Edge-Loc** | 3 | 0 | 1124 | 19 | 30 | 4 | 3 | 0 | 17 |
| **Edge-Ring** | 0 | 1 | 6 | 1189 | 0 | 1 | 1 | 0 | 2 |
| **Loc** | 19 | 6 | 24 | 0 | 1122 | 6 | 5 | 0 | 18 |
| **Random** | 0 | 0 | 0 | 0 | 0 | 1200 | 0 | 0 | 0 |
| **Scratch** | 0 | 0 | 3 | 0 | 3 | 0 | 1187 | 0 | 7 |
| **Near-full** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1200 | 0 |
| **none** | 13 | 0 | 20 | 2 | 20 | 2 | 14 | 0 | 1129 |

---

## 4. Danh mục Golden Feature Maps (NHWC Layout cho RTL Verification)

| Tầng / Khối (Layer Tag) | Kích thước NCHW | Kích thước Xuất Hex (NHWC) | Định dạng Dữ liệu |
|:---|:---:|:---:|:---:|
| `layer01_stem_conv1_out.hex` | `[1, 64, 64, 64]` | `[64, 64, 64] (262,144 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer02_s1b0_conv1_out.hex` | `[1, 64, 64, 64]` | `[64, 64, 64] (262,144 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer03_s1b0_block_out_out.hex` | `[1, 64, 64, 64]` | `[64, 64, 64] (262,144 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer04_s1b1_conv1_out.hex` | `[1, 64, 64, 64]` | `[64, 64, 64] (262,144 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer05_s1b1_block_out_out.hex` | `[1, 64, 64, 64]` | `[64, 64, 64] (262,144 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer06_s2b0_conv1_out.hex` | `[1, 128, 32, 32]` | `[32, 32, 128] (131,072 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer07_s2b0_shortcut_out.hex` | `[1, 128, 32, 32]` | `[32, 32, 128] (131,072 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer08_s2b0_block_out_out.hex` | `[1, 128, 32, 32]` | `[32, 32, 128] (131,072 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer09_s2b1_conv1_out.hex` | `[1, 128, 32, 32]` | `[32, 32, 128] (131,072 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer10_s2b1_block_out_out.hex` | `[1, 128, 32, 32]` | `[32, 32, 128] (131,072 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer11_s3b0_conv1_out.hex` | `[1, 256, 16, 16]` | `[16, 16, 256] (65,536 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer12_s3b0_shortcut_out.hex` | `[1, 256, 16, 16]` | `[16, 16, 256] (65,536 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer13_s3b0_block_out_out.hex` | `[1, 256, 16, 16]` | `[16, 16, 256] (65,536 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer14_s3b1_conv1_out.hex` | `[1, 256, 16, 16]` | `[16, 16, 256] (65,536 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer15_s3b1_block_out_out.hex` | `[1, 256, 16, 16]` | `[16, 16, 256] (65,536 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer16_s4b0_conv1_out.hex` | `[1, 512, 8, 8]` | `[8, 8, 512] (32,768 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer17_s4b0_shortcut_out.hex` | `[1, 512, 8, 8]` | `[8, 8, 512] (32,768 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer18_s4b0_block_out_out.hex` | `[1, 512, 8, 8]` | `[8, 8, 512] (32,768 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer19_s4b1_conv1_out.hex` | `[1, 512, 8, 8]` | `[8, 8, 512] (32,768 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer20_s4b1_block_out_out.hex` | `[1, 512, 8, 8]` | `[8, 8, 512] (32,768 bytes)` | INT8 (2-char 2's Comp Hex) |
| `gap_out_out.hex` | `[1, 512]` | `[1, 512] (512 bytes)` | INT8 (2-char 2's Comp Hex) |
| `layer21_fc_out.hex` | `[1, 9]` | `[1, 9] (9 bytes)` | INT8 (2-char 2's Comp Hex) |

---

## 5. Kết luận Kiểm định Phần cứng (RTL Readiness Sign-off)

- **Bảo toàn Vật lý Wafer Map:** Toàn bộ ảnh đầu vào $64 \times 64 \times 1$ chỉ chứa đúng 3 mức lượng tử hóa rời rạc `{0x00, 0x40, 0x7F}` tương ứng `{0, 64, 127}`.
- **Không dùng Phép chia / Dấu phẩy động:** Mọi tầng tích chập, nhánh tắt (Shortcut Alignment) và Global Average Pooling (`>> 6`) đều vận hành 100% bằng số nguyên và dịch bit.
- **Sẵn sàng cho Vivado RTL Testbench:** Toàn bộ trọng số (`weights_hex/`), hệ số chệch (`bias_hex/`), cấu hình (`quant_config.vh`) và Feature Map chuẩn (`golden_fmaps/`) đã được xuất theo đúng giao thức bộ nhớ **NHWC** và **Tiling $(T_m, T_n)$**.
