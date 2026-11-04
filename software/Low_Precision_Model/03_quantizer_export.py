"""
03_quantizer_export.py
======================
Bước 3: Lượng tử hóa Đối xứng Số nguyên Thuần túy (INT8/INT32), Xử lý Nhánh Cộng Tắt
        (Residual Shortcut Alignment) và Xuất toàn bộ Vector Phần cứng (.hex, .vh, .bin).

Đặc tả kỹ thuật:
1. Lượng tử hóa đối xứng (Symmetric Quantization) cho toàn bộ 20 lớp Conv và 1 lớp FC:
   - Trọng số INT8:
       S_w = max(|W_fold|) / 127
       q_w = clip(round(W_fold / S_w), -127, 127)  in INT8
   - Hệ số chệch (Bias) INT32:
       S_b = S_x * S_w
       q_b = round(b_fold / (S_x * S_w))          in INT32

2. Tái lượng tử hóa bằng Nhân số nguyên + Dịch bit (Multiplier + Bit-Shift Re-quantization):
   - Biểu diễn M_real = (S_x * S_w) / S_y  xấp xỉ  M0 / (2^n), với M0 = round(M_real * 2^n)
   - Công thức phần cứng (kèm làm tròn Half-up + 2^(n-1)):
       q_y = clip( ((acc_int32 * M0 + 2^(n-1)) >> n), -128, 127 )

3. Căn chỉnh Nhánh Cộng Tắt (Residual Shortcut Alignment) trong BasicBlock:
   - Quy cả nhánh chính (conv2) và nhánh tắt (Identity hoặc 1x1 Conv) về cùng hệ số tỷ lệ
     đầu ra của Block là S_out trước khi cộng và qua ReLU:
       q_block_out = clip(
           ((acc_conv2 * M_conv2 + 2^(n1-1)) >> n1) + ((q_sc * M_sc + 2^(n2-1)) >> n2),
           0, 127
       )

4. Quy chuẩn Sắp xếp Bộ nhớ Phần cứng (Memory Layout Protocol):
   - Trọng số (layer*_w.hex & ddr_weights_pack.bin): Sắp xếp theo thứ tự khối Tiling
     [Out_Tile_Group, In_Tile_Group, Kernel_H, Kernel_W, Tm, Tn].
   - Xuất mã bù 2 (Two's Complement): 2 ký tự Hex/dòng cho INT8, 8 ký tự Hex/dòng cho INT32.
"""

import os
import json
import math
import argparse
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# 1. CÁC HÀM XUẤT FILE .HEX & RE-QUANTIZATION CHUẨN CÔNG NGHIỆP
# =====================================================================
def export_int8_to_hex(tensor_int8: np.ndarray, filepath: str) -> None:
    """
    Xuất mảng numpy int8 ra file .hex theo chuẩn bù 2 (2 ký tự Hex / dòng)
    Phù hợp trực tiếp với lệnh $readmemh("...", mem_array) trong Verilog.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    flat_data = tensor_int8.flatten().astype(np.int8)
    # Chuyển sang uint8 để lấy đúng mã bù 2 cho số âm (ví dụ -5 -> 251 -> 0xFB)
    uint8_view = flat_data.view(np.uint8)
    with open(filepath, 'w', encoding='utf-8') as f:
        for val in uint8_view:
            f.write(f"{val:02X}\n")


def export_int32_to_hex(tensor_int32: np.ndarray, filepath: str) -> None:
    """
    Xuất mảng numpy int32 (Bias / Accumulator Golden) ra file .hex (8 ký tự Hex / dòng)
    theo chuẩn bù 2 (ví dụ -120 -> 0xFFFFFF88).
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    flat_data = tensor_int32.flatten().astype(np.int32)
    uint32_view = flat_data.view(np.uint32)
    with open(filepath, 'w', encoding='utf-8') as f:
        for val in uint32_view:
            f.write(f"{val:08X}\n")


def hw_requantize_int32_to_int8(
    acc_int32: np.ndarray,
    M0: int,
    shift_n: int,
    relu: bool = False
) -> np.ndarray:
    """
    Hàm mô phỏng Bit-Accurate khối Re-quantization trên FPGA
    Sử dụng phép nhân 64-bit trung gian để tránh tràn số trước khi dịch phải.
    """
    acc_64 = acc_int32.astype(np.int64)
    rounding_offset = (1 << (shift_n - 1)) if shift_n > 0 else 0
    scaled = (acc_64 * int(M0) + rounding_offset) >> shift_n

    min_val = 0 if relu else -128
    clipped = np.clip(scaled, min_val, 127)
    return clipped.astype(np.int8)


# =====================================================================
# 2. CÁC HÀM TÍNH TOÁN LƯỢNG TỬ HÓA & TILING TRỌNG SỐ
# =====================================================================
def quantize_weight_int8(w_fp32: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Lượng tử hóa đối xứng trọng số sang INT8 [-127, 127]:
        S_w = max(|W_fold|) / 127
        q_w = clip(round(W_fold / S_w), -127, 127)
    """
    max_abs = float(np.max(np.abs(w_fp32)))
    s_w = max_abs / 127.0 if max_abs > 1e-12 else 1e-8
    q_w = np.clip(np.round(w_fp32 / s_w), -127, 127).astype(np.int8)
    return q_w, s_w


def quantize_bias_int32(b_fp32: np.ndarray, s_x: float, s_w: float) -> Tuple[np.ndarray, float]:
    """
    Lượng tử hóa hệ số chệch (Bias) sang INT32 với hệ số tỷ lệ S_b = S_x * S_w:
        q_b = round(b_fold / (S_x * S_w))
    """
    s_b = float(s_x * s_w)
    if s_b < 1e-16:
        s_b = 1e-16
    q_b = np.round(b_fp32 / s_b)
    q_b = np.clip(q_b, -2**31, 2**31 - 1).astype(np.int32)
    return q_b, s_b


def compute_multiplier_and_shift(
    m_real: float,
    m0_bits: int = 15,
    min_shift: int = 1,
    max_shift: int = 30
) -> Tuple[int, int]:
    """
    Tìm cặp số nguyên dương (M0, n) sao cho:
        M_real ≈ M0 / (2^n)   với M0 = round(M_real * 2^n)
    Trong đó M0 nằm trong dải không dấu m0_bits (mặc định 15-bit: [1, 32767] để khớp
    ngõ vào 18-bit có dấu của khối DSP48E1 trên FPGA Xilinx Zynq-7020 / PYNQ-Z2).
    """
    if m_real <= 0.0:
        return 0, 0

    max_m0 = (1 << m0_bits) - 1
    best_n = min_shift
    best_m0 = 1
    best_err = float("inf")

    for n in range(min_shift, max_shift + 1):
        candidate_m0 = int(round(m_real * (1 << n)))
        if 1 <= candidate_m0 <= max_m0:
            approx = candidate_m0 / float(1 << n)
            err = abs(m_real - approx)
            # Ưu tiên sai số nhỏ nhất và tận dụng tối đa độ phân giải bit của M0
            if err <= best_err:
                best_err = err
                best_m0 = candidate_m0
                best_n = n

    return best_m0, best_n


def pack_weights_tiled(
    w_int8: np.ndarray,
    tm: int = 16,
    tn: int = 16
) -> np.ndarray:
    """
    Sắp xếp lại mảng trọng số từ chuẩn PyTorch [C_out, C_in, K_h, K_w]
    sang chuẩn Tiling của phần cứng FPGA:
        [Out_Tile_Group, In_Tile_Group, Kernel_H, Kernel_W, Tm, Tn]
    Giúp bộ điều khiển FSM trên Verilog chỉ cần tăng bộ đếm địa chỉ tuyến tính (addr <= addr + 1).
    Nếu C_out không chia hết cho Tm hoặc C_in không chia hết cho Tn (ví dụ lớp đầu C_in = 1,
    hoặc lớp FC C_out = 9), hàm tự động chèn 0 (Zero-Padding) vào các khe PE dư.
    """
    if w_int8.ndim == 2:
        # Tầng FC [C_out, C_in] -> xem như Kernel 1x1: [C_out, C_in, 1, 1]
        w_4d = w_int8[:, :, np.newaxis, np.newaxis]
    else:
        w_4d = w_int8

    c_out, c_in, k_h, k_w = w_4d.shape
    out_groups = math.ceil(c_out / tm)
    in_groups = math.ceil(c_in / tn)

    padded_c_out = out_groups * tm
    padded_c_in = in_groups * tn

    w_padded = np.zeros((padded_c_out, padded_c_in, k_h, k_w), dtype=np.int8)
    w_padded[:c_out, :c_in, :, :] = w_4d

    # Reshape thành [Out_Tile_Group, Tm, In_Tile_Group, Tn, K_h, K_w]
    w_reshaped = w_padded.reshape(out_groups, tm, in_groups, tn, k_h, k_w)
    # Hoán vị trục về [Out_Tile_Group, In_Tile_Group, K_h, K_w, Tm, Tn]
    w_tiled = np.transpose(w_reshaped, (0, 2, 4, 5, 1, 3))
    return np.ascontiguousarray(w_tiled, dtype=np.int8)


# =====================================================================
# 3. ĐỊNH NGHĨA MÔ HÌNH ĐÃ GỘP BATCHNORM ĐỂ CALIBRATE ACTIVATION SCALES
# =====================================================================
class FoldedBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.has_shortcut_conv = (stride != 1 or in_planes != planes * self.expansion)
        self.shortcut_conv = (
            nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=True)
            if self.has_shortcut_conv else None
        )
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut_conv(x) if self.shortcut_conv is not None else x
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        out = self.relu_out(out + identity)
        return out


class ResNet18ModifiedFolded(nn.Module):
    def __init__(self, num_classes: int = 9, in_channels: int = 1):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes, bias=True)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(FoldedBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.sum(dim=(2, 3)) / 64.0
        out = self.fc(out)
        return out


# =====================================================================
# 4. HIỆU CHUẨN DẢI ĐỘNG ACTIVATION (CALIBRATION) & TRÍCH XUẤT THAM SỐ
# =====================================================================
@torch.no_grad()
def calibrate_activation_scales(
    model: ResNet18ModifiedFolded,
    calib_inputs_fp32: torch.Tensor,
    device: torch.device,
    batch_size: int = 64
) -> Dict[str, float]:
    """
    Chạy tập dữ liệu hiệu chuẩn (Calibration Set) qua mô hình FP32 đã gộp BN
    để đo giá trị cực đại max(|y|) tại đầu ra mỗi tầng, từ đó tính S_y = max(|y|) / 127.0.
    """
    model.eval()
    max_activations: Dict[str, float] = {}

    def update_max(name: str, tensor: torch.Tensor):
        val = float(torch.max(torch.abs(tensor)).item())
        if name not in max_activations or val > max_activations[name]:
            max_activations[name] = val

    n_samples = calib_inputs_fp32.size(0)
    for start in range(0, n_samples, batch_size):
        x = calib_inputs_fp32[start:start + batch_size].to(device)

        # Stem
        out = model.relu(model.conv1(x))
        update_max("stem_out", out)

        # 4 Stages
        for s_idx, stage_name in enumerate(["layer1", "layer2", "layer3", "layer4"], start=1):
            stage = getattr(model, stage_name)
            for b_idx, block in enumerate(stage):
                prefix = f"s{s_idx}b{b_idx}"
                c1_out = block.relu1(block.conv1(out))
                update_max(f"{prefix}_conv1_out", c1_out)

                c2_out = block.conv2(c1_out)
                sc_out = block.shortcut_conv(out) if block.has_shortcut_conv else out
                out = block.relu_out(c2_out + sc_out)
                update_max(f"{prefix}_block_out", out)

        # Global Average Pooling (8x8 -> chia 64)
        gap_out = out.sum(dim=(2, 3)) / 64.0
        update_max("gap_out", gap_out)

        # FC
        fc_out = model.fc(gap_out)
        update_max("fc_out", fc_out)

    scales: Dict[str, float] = {}
    for k, max_val in max_activations.items():
        scales[k] = (max_val / 127.0) if max_val > 1e-12 else (1.0 / 127.0)

    # Đảm bảo tính nhất quán của GAP (dịch phải >> 6 giữ nguyên hệ số tỷ lệ với đầu ra Stage 4)
    scales["gap_out"] = scales["s4b1_block_out"]
    return scales


# =====================================================================
# 5. QUY TRÌNH LƯỢNG TỬ HÓA TOÀN MẠNG & XUẤT VECTOR PHẦN CỨNG
# =====================================================================
def quantize_and_export_all(
    model: ResNet18ModifiedFolded,
    act_scales: Dict[str, float],
    input_scale: float,
    hw_dir: str,
    ckpt_dir: str,
    tm: int = 16,
    tn: int = 16,
    m0_bits: int = 15
) -> Dict[str, Any]:
    """
    Thực hiện lượng tử hóa toàn bộ 20 lớp Conv (17 nhánh chính + 3 nhánh tắt 1x1) và 1 lớp FC.
    Xuất đầy đủ:
      - hw_vectors/weights_hex/layer*_w.hex
      - hw_vectors/bias_hex/layer*_b.hex
      - hw_vectors/quant_config.hex & quant_config.vh
      - hw_vectors/ddr_weights_pack.bin & ddr_memory_map.json
      - checkpoints/quantized_resnet18_int8.npz (dùng cho 04_bit_accurate_emulator.py)
    """
    weights_hex_dir = os.path.join(hw_dir, "weights_hex")
    bias_hex_dir = os.path.join(hw_dir, "bias_hex")
    os.makedirs(weights_hex_dir, exist_ok=True)
    os.makedirs(bias_hex_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    npz_dict: Dict[str, Any] = {}
    layer_manifest: List[Dict[str, Any]] = []
    block_quant_configs: List[Dict[str, Any]] = []
    ddr_binary_chunks: List[bytes] = []
    ddr_offset = 0

    layer_id = 1

    def register_and_export_layer(
        layer_tag: str,
        w_fp32: np.ndarray,
        b_fp32: np.ndarray,
        s_in: float,
        s_out: float,
        M0: int,
        shift_n: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        nonlocal layer_id, ddr_offset
        q_w, s_w = quantize_weight_int8(w_fp32)
        q_b, s_b = quantize_bias_int32(b_fp32, s_in, s_w)

        # Sắp xếp trọng số theo chuẩn Tiling [Out_Tile_Group, In_Tile_Group, K_h, K_w, Tm, Tn]
        q_w_tiled = pack_weights_tiled(q_w, tm=tm, tn=tn)

        w_hex_name = f"layer{layer_id:02d}_{layer_tag}_w.hex"
        b_hex_name = f"layer{layer_id:02d}_{layer_tag}_b.hex"
        export_int8_to_hex(q_w_tiled, os.path.join(weights_hex_dir, w_hex_name))
        export_int32_to_hex(q_b, os.path.join(bias_hex_dir, b_hex_name))

        # Đóng gói nối tiếp vào bộ nhớ DDR3 binary
        raw_bytes = q_w_tiled.flatten().astype(np.int8).tobytes()
        ddr_binary_chunks.append(raw_bytes)
        byte_len = len(raw_bytes)

        layer_info = {
            "layer_index": layer_id,
            "name": layer_tag,
            "orig_weight_shape": list(q_w.shape),
            "tiled_weight_shape": list(q_w_tiled.shape),
            "bias_shape": list(q_b.shape),
            "s_in": s_in,
            "s_w": s_w,
            "s_b": s_b,
            "s_out": s_out,
            "M0": int(M0),
            "shift_n": int(shift_n),
            "w_hex_file": w_hex_name,
            "b_hex_file": b_hex_name,
            "ddr_byte_offset": ddr_offset,
            "ddr_byte_length": byte_len
        }
        layer_manifest.append(layer_info)
        ddr_offset += byte_len

        # Lưu vào dictionary cho bộ mô phỏng Bit-Accurate ở Bước 4
        npz_dict[f"{layer_tag}_w"] = q_w
        npz_dict[f"{layer_tag}_w_tiled"] = q_w_tiled
        npz_dict[f"{layer_tag}_b"] = q_b
        npz_dict[f"{layer_tag}_M0"] = np.int32(M0)
        npz_dict[f"{layer_tag}_shift_n"] = np.int32(shift_n)

        print(f"  [Layer {layer_id:02d}] {layer_tag:<18} | W: {str(list(q_w.shape)):<18} "
              f"-> Tiled: {str(list(q_w_tiled.shape)):<22} | M0={M0:<6d}, n={shift_n:<2d}")

        layer_id += 1
        return q_w, q_b, s_w

    # -----------------------------------------------------------------
    # A. TẦNG STEM (conv1)
    # -----------------------------------------------------------------
    w_conv1 = model.conv1.weight.detach().cpu().numpy()
    b_conv1 = model.conv1.bias.detach().cpu().numpy()
    _, s_w_conv1 = quantize_weight_int8(w_conv1)
    s_stem_out = act_scales["stem_out"]
    m_real_stem = (input_scale * s_w_conv1) / s_stem_out
    m0_stem, n_stem = compute_multiplier_and_shift(m_real_stem, m0_bits=m0_bits)

    register_and_export_layer(
        layer_tag="stem_conv1",
        w_fp32=w_conv1,
        b_fp32=b_conv1,
        s_in=input_scale,
        s_out=s_stem_out,
        M0=m0_stem,
        shift_n=n_stem
    )

    block_quant_configs.append({
        "unit_name": "STEM_CONV1",
        "M_conv1": m0_stem, "s_conv1": n_stem,
        "M_conv2": 0, "s_conv2": 0,
        "M_sc": 0, "s_sc": 0
    })

    # -----------------------------------------------------------------
    # B. 4 STAGES (8 BASIC BLOCKS)
    # -----------------------------------------------------------------
    curr_s_in = s_stem_out
    for s_idx, stage_name in enumerate(["layer1", "layer2", "layer3", "layer4"], start=1):
        stage = getattr(model, stage_name)
        for b_idx, block in enumerate(stage):
            prefix = f"s{s_idx}b{b_idx}"
            s_c1_out = act_scales[f"{prefix}_conv1_out"]
            s_block_out = act_scales[f"{prefix}_block_out"]

            # 1. Nhánh chính Conv1 (kèm ReLU)
            w_c1 = block.conv1.weight.detach().cpu().numpy()
            b_c1 = block.conv1.bias.detach().cpu().numpy()
            _, s_w_c1 = quantize_weight_int8(w_c1)
            m_real_c1 = (curr_s_in * s_w_c1) / s_c1_out
            m0_c1, n_c1 = compute_multiplier_and_shift(m_real_c1, m0_bits=m0_bits)

            register_and_export_layer(
                layer_tag=f"{prefix}_conv1",
                w_fp32=w_c1,
                b_fp32=b_c1,
                s_in=curr_s_in,
                s_out=s_c1_out,
                M0=m0_c1,
                shift_n=n_c1
            )

            # 2. Nhánh chính Conv2 (Re-quantize về thẳng S_block_out trước khi cộng nhánh tắt)
            w_c2 = block.conv2.weight.detach().cpu().numpy()
            b_c2 = block.conv2.bias.detach().cpu().numpy()
            _, s_w_c2 = quantize_weight_int8(w_c2)
            m_real_c2 = (s_c1_out * s_w_c2) / s_block_out
            m0_c2, n_c2 = compute_multiplier_and_shift(m_real_c2, m0_bits=m0_bits)

            register_and_export_layer(
                layer_tag=f"{prefix}_conv2",
                w_fp32=w_c2,
                b_fp32=b_c2,
                s_in=s_c1_out,
                s_out=s_block_out,
                M0=m0_c2,
                shift_n=n_c2
            )

            # 3. Nhánh tắt (Residual Shortcut Alignment về S_block_out)
            if block.has_shortcut_conv:
                w_sc = block.shortcut_conv.weight.detach().cpu().numpy()
                b_sc = block.shortcut_conv.bias.detach().cpu().numpy()
                _, s_w_sc = quantize_weight_int8(w_sc)
                m_real_sc = (curr_s_in * s_w_sc) / s_block_out
                m0_sc, n_sc = compute_multiplier_and_shift(m_real_sc, m0_bits=m0_bits)

                register_and_export_layer(
                    layer_tag=f"{prefix}_shortcut",
                    w_fp32=w_sc,
                    b_fp32=b_sc,
                    s_in=curr_s_in,
                    s_out=s_block_out,
                    M0=m0_sc,
                    shift_n=n_sc
                )
                npz_dict[f"{prefix}_has_shortcut_conv"] = np.bool_(True)
            else:
                # Nhánh tắt Identity: căn chỉnh trực tiếp từ S_in về S_block_out
                m_real_sc = curr_s_in / s_block_out
                m0_sc, n_sc = compute_multiplier_and_shift(m_real_sc, m0_bits=m0_bits)
                npz_dict[f"{prefix}_has_shortcut_conv"] = np.bool_(False)
                npz_dict[f"{prefix}_shortcut_M0"] = np.int32(m0_sc)
                npz_dict[f"{prefix}_shortcut_shift_n"] = np.int32(n_sc)

            block_quant_configs.append({
                "unit_name": prefix.upper(),
                "M_conv1": m0_c1, "s_conv1": n_c1,
                "M_conv2": m0_c2, "s_conv2": n_c2,
                "M_sc": m0_sc, "s_sc": n_sc
            })

            curr_s_in = s_block_out

    # -----------------------------------------------------------------
    # C. TẦNG PHÂN LOẠI CUỐI (FC 512 -> 9)
    # -----------------------------------------------------------------
    s_gap_out = act_scales["gap_out"]
    s_fc_out = act_scales["fc_out"]
    w_fc = model.fc.weight.detach().cpu().numpy()
    b_fc = model.fc.bias.detach().cpu().numpy()
    _, s_w_fc = quantize_weight_int8(w_fc)
    m_real_fc = (s_gap_out * s_w_fc) / s_fc_out
    m0_fc, n_fc = compute_multiplier_and_shift(m_real_fc, m0_bits=m0_bits)

    register_and_export_layer(
        layer_tag="fc",
        w_fp32=w_fc,
        b_fp32=b_fc,
        s_in=s_gap_out,
        s_out=s_fc_out,
        M0=m0_fc,
        shift_n=n_fc
    )

    block_quant_configs.append({
        "unit_name": "FC_OUT",
        "M_conv1": m0_fc, "s_conv1": n_fc,
        "M_conv2": 0, "s_conv2": 0,
        "M_sc": 0, "s_sc": 0
    })

    # -----------------------------------------------------------------
    # D. XUẤT FILE CẤU HÌNH LƯỢNG TỬ HÓA: quant_config.hex & quant_config.vh
    # -----------------------------------------------------------------
    quant_hex_path = os.path.join(hw_dir, "quant_config.hex")
    quant_vh_path = os.path.join(hw_dir, "quant_config.vh")

    with open(quant_hex_path, "w", encoding="utf-8") as f_hex:
        for cfg in block_quant_configs:
            # Mỗi dòng đóng gói 6 thông số 16-bit (M_conv1, s_conv1, M_conv2, s_conv2, M_sc, s_sc)
            line = (
                f"{cfg['M_conv1']:04X}_{cfg['s_conv1']:02X}_"
                f"{cfg['M_conv2']:04X}_{cfg['s_conv2']:02X}_"
                f"{cfg['M_sc']:04X}_{cfg['s_sc']:02X}"
            )
            f_hex.write(line + "\n")

    with open(quant_vh_path, "w", encoding="utf-8") as f_vh:
        f_vh.write("// =====================================================================\n")
        f_vh.write("// Auto-generated by 03_quantizer_export.py\n")
        f_vh.write("// Re-quantization Multipliers (M0) and Right-Shift (s) for Verilog RTL\n")
        f_vh.write("// =====================================================================\n")
        f_vh.write("`ifndef QUANT_CONFIG_VH\n")
        f_vh.write("`define QUANT_CONFIG_VH\n\n")
        f_vh.write(f"localparam integer PE_TM = {tm};\n")
        f_vh.write(f"localparam integer PE_TN = {tn};\n")
        f_vh.write("localparam integer GAP_SHIFT_BITS = 6; // 8x8 Global Average Pool (>> 6)\n\n")
        for cfg in block_quant_configs:
            u = cfg["unit_name"]
            f_vh.write(f"// --- {u} ---\n")
            f_vh.write(f"localparam [15:0] {u}_M_CONV1 = 16'd{cfg['M_conv1']};\n")
            f_vh.write(f"localparam [5:0]  {u}_S_CONV1 = 6'd{cfg['s_conv1']};\n")
            f_vh.write(f"localparam [15:0] {u}_M_CONV2 = 16'd{cfg['M_conv2']};\n")
            f_vh.write(f"localparam [5:0]  {u}_S_CONV2 = 6'd{cfg['s_conv2']};\n")
            f_vh.write(f"localparam [15:0] {u}_M_SC    = 16'd{cfg['M_sc']};\n")
            f_vh.write(f"localparam [5:0]  {u}_S_SC    = 6'd{cfg['s_sc']};\n\n")
        f_vh.write("`endif // QUANT_CONFIG_VH\n")

    # -----------------------------------------------------------------
    # E. XUẤT FILE ĐÓNG GÓI TRỌNG SỐ DDR3: ddr_weights_pack.bin
    # -----------------------------------------------------------------
    ddr_bin_path = os.path.join(hw_dir, "ddr_weights_pack.bin")
    with open(ddr_bin_path, "wb") as f_bin:
        for chunk in ddr_binary_chunks:
            f_bin.write(chunk)

    ddr_map_path = os.path.join(hw_dir, "ddr_memory_map.json")
    with open(ddr_map_path, "w", encoding="utf-8") as f_json:
        json.dump({
            "tiling_tm": tm,
            "tiling_tn": tn,
            "total_bytes": ddr_offset,
            "layers": layer_manifest,
            "block_quant_configs": block_quant_configs
        }, f_json, indent=4)

    # Lưu toàn bộ gói tham số lượng tử hóa để Bước 4 (Bit-Accurate Emulator) nạp trực tiếp
    quant_npz_path = os.path.join(ckpt_dir, "quantized_resnet18_int8.npz")
    np.savez_compressed(quant_npz_path, **npz_dict)

    print(f"\n  Đã xuất thành công:")
    print(f"    - 21 file Trọng số INT8 (.hex): {weights_hex_dir}")
    print(f"    - 21 file Bias INT32 (.hex):    {bias_hex_dir}")
    print(f"    - Cấu hình Verilog Header:      {quant_vh_path} & {quant_hex_path}")
    print(f"    - File DDR3 Binary ({ddr_offset:,} bytes): {ddr_bin_path}")
    print(f"    - Tham số cho Bit-Accurate Sim: {quant_npz_path}")

    return npz_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Bước 3: Lượng tử hóa INT8/INT32 & Xuất Vector Phần cứng")
    parser.add_argument(
        "--folded_ckpt",
        type=str,
        default=None,
        help="Đường dẫn tới checkpoints/resnet18_fp32_folded.pth"
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        default=None,
        help="Đường dẫn tới file dữ liệu wm811k_64x64_int8.npz"
    )
    parser.add_argument(
        "--hw_dir",
        type=str,
        default=None,
        help="Thư mục gốc hw_vectors/ để xuất .hex, .vh, .bin"
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=1024,
        help="Số lượng mẫu dùng để calibrate hệ số tỷ lệ activation S_y (mặc định: 1024)"
    )
    parser.add_argument(
        "--tm",
        type=int,
        default=16,
        help="Kích thước Tile kênh đầu ra Tm của mảng PE (mặc định: 16)"
    )
    parser.add_argument(
        "--tn",
        type=int,
        default=16,
        help="Kích thước Tile kênh đầu vào Tn của mảng PE (mặc định: 16)"
    )
    parser.add_argument(
        "--m0_bits",
        type=int,
        default=15,
        help="Độ rộng bit dương của hệ số nhân M0 (mặc định: 15-bit cho DSP48E1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị chạy calibration (cuda hoặc cpu)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(os.path.dirname(script_dir)).lower() == "software":
        base_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "phase1_golden_model"))
    else:
        base_dir = script_dir

    ckpt_dir = os.path.join(base_dir, "checkpoints")
    folded_ckpt = args.folded_ckpt if args.folded_ckpt else os.path.join(ckpt_dir, "resnet18_fp32_folded.pth")
    npz_path = args.npz_path if args.npz_path else os.path.join(base_dir, "data", "wm811k_64x64_int8.npz")
    hw_dir = args.hw_dir if args.hw_dir else os.path.join(base_dir, "hw_vectors")

    device = torch.device(args.device)
    print("=" * 72)
    print("  BƯỚC 3: LƯỢNG TỬ HÓA ĐỐI XỨNG INT8/INT32 & XUẤT VECTOR PHẦN CỨNG")
    print("=" * 72)

    if not os.path.isfile(folded_ckpt):
        raise FileNotFoundError(
            f"Không tìm thấy trọng số đã gộp BN tại: {folded_ckpt}\n"
            f"Hãy chạy 02_resnet18_Modified_train.py trước."
        )
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu tại: {npz_path}\n"
            f"Hãy chạy 01_wm811k_preprocess.py trước."
        )

    # 1. Nạp mô hình FP32 đã gộp BatchNorm
    ckpt = torch.load(folded_ckpt, map_location=device)
    input_scale = float(ckpt.get("input_scale", 1.0 / 127.0))
    model = ResNet18ModifiedFolded(num_classes=9, in_channels=1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Nạp dữ liệu Calibration và đo dải động Activation
    data = np.load(npz_path, allow_pickle=True)
    X_train_int8 = data["X_train"][:args.calib_samples]
    calib_fp32 = torch.from_numpy(X_train_int8).float() * input_scale

    print(f"[1/2] Đang hiệu chuẩn hệ số tỷ lệ Activation (S_y) trên {len(X_train_int8)} mẫu...")
    act_scales = calibrate_activation_scales(model, calib_fp32, device=device)

    # 3. Lượng tử hóa toàn mạng, tính (M0, n) và xuất file .hex / .vh / .bin
    print(f"[2/2] Đang lượng tử hóa và đóng gói Trọng số theo Tiling (Tm={args.tm}, Tn={args.tn})...\n")
    quantize_and_export_all(
        model=model,
        act_scales=act_scales,
        input_scale=input_scale,
        hw_dir=hw_dir,
        ckpt_dir=ckpt_dir,
        tm=args.tm,
        tn=args.tn,
        m0_bits=args.m0_bits
    )

    # Đảm bảo 9 mẫu test cũng có mặt trong hw_vectors/test_samples/
    test_samples_dir = os.path.join(hw_dir, "test_samples")
    os.makedirs(test_samples_dir, exist_ok=True)
    X_test_int8 = data["X_test"]
    y_test = data["y_test"]
    classes = data["classes"].tolist()
    for cls_idx, cls_name in enumerate(classes):
        matches = np.where(y_test == cls_idx)[0]
        if len(matches) > 0:
            wafer_nhwc = np.transpose(X_test_int8[matches[0]], (1, 2, 0))
            hex_path = os.path.join(test_samples_dir, f"input_wafer_{cls_idx}_{cls_name}.hex")
            if not os.path.isfile(hex_path):
                export_int8_to_hex(wafer_nhwc, hex_path)

    print("\nHoàn tất Bước 3!")


if __name__ == "__main__":
    main()
