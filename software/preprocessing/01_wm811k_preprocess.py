"""
01_wm811k_preprocess.py
=======================
Bước 1: Tiền xử lý dữ liệu WM-811K giữ nguyên bản chất vật lý cho phần cứng FPGA.

Đặc tả kỹ thuật:
1. Đọc dữ liệu gốc từ LSWMD.pkl / wafers.pkl (xử lý tương thích Pandas/Python mới).
2. Lọc các tấm wafer có nhãn thuộc 9 lớp lỗi chuẩn:
   Center (0), Donut (1), Edge-Loc (2), Edge-Ring (3), Loc (4),
   Random (5), Scratch (6), Near-full (7), none (8).
3. Mã hóa nhãn dạng Categorical (0..8) và One-Hot vector (9 chiều).
4. Resize về kích thước 64x64 BẮT BUỘC dùng nội suy Nearest-Neighbor (cv2.INTER_NEAREST)
   để bảo toàn tuyệt đối 3 giá trị vật lý rời rạc của die:
   - 0: Không phải vùng wafer (Background / Ngoài rìa)
   - 1: Die đạt chuẩn (Normal / Passing Die)
   - 2: Die bị lỗi (Defective / Failing Die)
5. Ánh xạ trực tiếp sang miền lượng tử hóa đầu vào INT8 (1 kênh: 64x64x1):
   {0, 1, 2} -> {0, 64, 127} (tận dụng tối đa dải động dương INT8, tiết kiệm 66% BRAM).
6. Xuất 9 file mẫu test chuẩn (.hex) đại diện cho 9 lớp lỗi tại:
   hw_vectors/test_samples/input_wafer_*.hex
"""

import os
import sys
import types
import pickle
import argparse
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# =====================================================================
# 1. ĐỊNH NGHĨA 9 LỚP LỖI WM-811K & BẢNG ÁNH XẠ VẬT LÝ INT8
# =====================================================================
CLASS_TO_IDX: Dict[str, int] = {
    'Center': 0,
    'Donut': 1,
    'Edge-Loc': 2,
    'Edge-Ring': 3,
    'Loc': 4,
    'Random': 5,
    'Scratch': 6,
    'Near-full': 7,
    'none': 8
}

IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}
CLASSES: List[str] = [IDX_TO_CLASS[i] for i in range(len(CLASS_TO_IDX))]

# Bảng tra cứu (LUT) ánh xạ giá trị die vật lý {0, 1, 2} sang INT8 {0, 64, 127}
PHYSICAL_TO_INT8_LUT = np.array([0, 64, 127], dtype=np.int8)


# =====================================================================
# 2. CÁC HÀM TIỆN ÍCH XUẤT HEX & NỘI SUY NEAREST-NEIGHBOR
# =====================================================================
def export_int8_to_hex(tensor_int8: np.ndarray, filepath: str) -> None:
    """
    Xuất mảng numpy int8 ra file .hex theo chuẩn bù 2 (2 ký tự Hex / dòng).
    Phù hợp trực tiếp với lệnh $readmemh("...", mem_array) trong Verilog.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    flat_data = tensor_int8.flatten().astype(np.int8)
    uint8_view = flat_data.view(np.uint8)
    with open(filepath, 'w', encoding='utf-8') as f:
        for val in uint8_view:
            f.write(f"{val:02X}\n")


def resize_wafer_nearest(wafer_map: np.ndarray, target_size: int = 64) -> np.ndarray:
    """
    Resize tấm wafer về (target_size, target_size) BẮT BUỘC dùng Nearest-Neighbor.
    Tuyệt đối không dùng Bilinear/Bicubic để tránh sinh ra giá trị thập phân giả.
    """
    wafer_clean = np.clip(np.asarray(wafer_map, dtype=np.uint8), 0, 2)
    if HAS_CV2:
        resized = cv2.resize(
            wafer_clean,
            (target_size, target_size),
            interpolation=cv2.INTER_NEAREST
        )
    else:
        pil_img = Image.fromarray(wafer_clean, mode='L')
        resample_mode = getattr(Image, 'Resampling', Image).NEAREST
        resized = np.asarray(pil_img.resize((target_size, target_size), resample=resample_mode), dtype=np.uint8)
    return resized


def map_physical_to_int8(wafer_64x64: np.ndarray) -> np.ndarray:
    """
    Ánh xạ giá trị rời rạc {0, 1, 2} sang INT8 {0, 64, 127}.
    Đầu ra có kiểu np.int8, kích thước (64, 64).
    """
    indices = np.clip(wafer_64x64, 0, 2).astype(np.int32)
    return PHYSICAL_TO_INT8_LUT[indices]


def to_one_hot(labels: np.ndarray, num_classes: int = 9) -> np.ndarray:
    """Mã hóa mảng nhãn nguyên (N,) sang ma trận One-Hot (N, num_classes)."""
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


# =====================================================================
# 3. ĐỌC VÀ LỌC DỮ LIỆU TỪ LSWMD.PKL / WAFERS.PKL
# =====================================================================
def _patch_legacy_pandas_pickle() -> None:
    """Khắc phục tương thích ngược khi đọc file pickle tạo từ Pandas cũ trên Python 3.10+."""
    import pandas.core.indexes
    sys.modules['pandas.indexes'] = pandas.core.indexes
    if not hasattr(pandas.core.indexes, 'numeric'):
        numeric_mod = types.ModuleType('pandas.core.indexes.numeric')
        numeric_mod.Int64Index = pd.Index
        numeric_mod.UInt64Index = pd.Index
        numeric_mod.Float64Index = pd.Index
        sys.modules['pandas.core.indexes.numeric'] = numeric_mod
        sys.modules['pandas.indexes.numeric'] = numeric_mod


def resolve_default_pkl_path() -> str:
    """Tự động dò tìm đường dẫn file wafers.pkl hoặc LSWMD.pkl trong dự án."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "data", "LSWMD.pkl"),
        os.path.join(script_dir, "..", "data", "wafers.pkl"),
        os.path.join(script_dir, "data", "LSWMD.pkl"),
        os.path.join(script_dir, "data", "wafers.pkl"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return os.path.abspath(candidates[1])


def extract_label_str(raw_val) -> Optional[str]:
    """Trích xuất chuỗi tên lớp lỗi từ trường failureType của WM-811K."""
    if raw_val is None:
        return None
    if isinstance(raw_val, str):
        return raw_val if raw_val in CLASS_TO_IDX else None
    if isinstance(raw_val, (list, tuple, np.ndarray)):
        if len(raw_val) == 0:
            return None
        inner = raw_val[0]
        if isinstance(inner, (list, tuple, np.ndarray)):
            if len(inner) == 0:
                return None
            inner = inner[0]
        inner_str = str(inner).strip()
        return inner_str if inner_str in CLASS_TO_IDX else None
    return None


def augment_wafer_int8(wafer: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Tăng cường dữ liệu bảo toàn vật lý (chỉ dùng xoay 90/180/270 độ và lật gương).
    Tuyệt đối không làm thay đổi tập giá trị {0, 64, 127}.
    """
    k_rot = int(rng.integers(0, 4))
    out = np.rot90(wafer, k=k_rot)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    return np.ascontiguousarray(out, dtype=np.int8)


def load_and_preprocess_pkl(
    pkl_path: str,
    img_size: int = 64,
    target_per_class: Optional[int] = 6500,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Đọc file pickle WM-811K, lọc mẫu có nhãn, resize Nearest-Neighbor về 64x64,
    ánh xạ sang INT8 {0, 64, 127} và cân bằng số lượng mẫu giữa các lớp.
    """
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu gốc tại: {pkl_path}\n"
            f"Vui lòng đặt LSWMD.pkl hoặc wafers.pkl vào software/data/ hoặc truyền --pkl_path."
        )

    print(f"[1/4] Đang đọc dữ liệu từ: {pkl_path} ...")
    _patch_legacy_pandas_pickle()
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f, encoding='latin1')

    print(f"      Tổng số tấm wafer trong file gốc: {len(df):,}")

    # Lọc các tấm wafer có nhãn hợp lệ
    print("[2/4] Đang lọc nhãn và thực hiện Nearest-Neighbor resize về 64x64...")
    class_buckets: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(CLASS_TO_IDX))}

    wafer_maps = df['waferMap'].values
    failure_types = df['failureType'].values

    for wmap, ftype in zip(wafer_maps, failure_types):
        label_str = extract_label_str(ftype)
        if label_str is None:
            continue
        cls_idx = CLASS_TO_IDX[label_str]
        w_resized = resize_wafer_nearest(wmap, target_size=img_size)
        w_int8 = map_physical_to_int8(w_resized)
        class_buckets[cls_idx].append(w_int8)

    del df, wafer_maps, failure_types

    print("\n      Phân bố dữ liệu gốc sau khi lọc:")
    for idx in range(len(CLASS_TO_IDX)):
        print(f"        - Lớp {idx} ({IDX_TO_CLASS[idx]:<10}): {len(class_buckets[idx]):,} mẫu")

    # Cân bằng dữ liệu (Undersample lớp 'none' và Augment các lớp thiểu số bằng phép xoay/lật bảo toàn vật lý)
    rng = np.random.default_rng(seed)
    all_X: List[np.ndarray] = []
    all_y: List[int] = []

    if target_per_class is not None and target_per_class > 0:
        print(f"\n[3/4] Cân bằng các lớp về ~{target_per_class:,} mẫu/lớp (bảo toàn giá trị {0, 64, 127})...")
        for cls_idx, samples in class_buckets.items():
            n_curr = len(samples)
            if n_curr == 0:
                continue
            samples_arr = np.stack(samples, axis=0)
            if n_curr >= target_per_class:
                chosen_idx = rng.choice(n_curr, size=target_per_class, replace=False)
                selected = samples_arr[chosen_idx]
            else:
                # Giữ toàn bộ mẫu gốc và bổ sung bằng xoay 90 độ / lật gương
                extra_needed = target_per_class - n_curr
                extra_idx = rng.choice(n_curr, size=extra_needed, replace=True)
                augmented = np.stack([augment_wafer_int8(samples_arr[i], rng) for i in extra_idx], axis=0)
                selected = np.concatenate([samples_arr, augmented], axis=0)

            all_X.append(selected)
            all_y.append(np.full(len(selected), cls_idx, dtype=np.int64))
    else:
        for cls_idx, samples in class_buckets.items():
            if len(samples) == 0:
                continue
            all_X.append(np.stack(samples, axis=0))
            all_y.append(np.full(len(samples), cls_idx, dtype=np.int64))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Thêm chiều kênh (N, 1, 64, 64) cho chuẩn NCHW của PyTorch
    X = np.expand_dims(X, axis=1).astype(np.int8)
    return X, y


def export_9_test_sample_vectors(X_test: np.ndarray, y_test: np.ndarray, hw_test_dir: str) -> None:
    """
    Trích xuất 9 tấm wafer đại diện cho 9 lớp lỗi từ tập Test và xuất ra file .hex
    theo chuẩn NHWC (64x64x1 = 4096 dòng Hex 2 ký tự) để nạp vào Testbench Verilog.
    """
    os.makedirs(hw_test_dir, exist_ok=True)
    selected_wafers = []
    selected_labels = []

    print(f"\n[4/4] Xuất 9 file mẫu Wafer INT8 (.hex) cho RTL Testbench tại: {hw_test_dir}")
    for cls_idx in range(len(CLASS_TO_IDX)):
        cls_name = IDX_TO_CLASS[cls_idx]
        matches = np.where(y_test == cls_idx)[0]
        if len(matches) == 0:
            continue
        sample_idx = matches[0]
        # X_test[sample_idx] có shape (1, 64, 64) -> chuyển sang NHWC (64, 64, 1)
        wafer_nchw = X_test[sample_idx]
        wafer_nhwc = np.transpose(wafer_nchw, (1, 2, 0))

        hex_filename = f"input_wafer_{cls_idx}_{cls_name}.hex"
        hex_filepath = os.path.join(hw_test_dir, hex_filename)
        export_int8_to_hex(wafer_nhwc, hex_filepath)

        selected_wafers.append(wafer_nchw)
        selected_labels.append(cls_idx)
        print(f"      -> Đã xuất: {hex_filename:<32} | Shape NHWC: {wafer_nhwc.shape} | Values: {np.unique(wafer_nhwc).tolist()}")

    # Lưu kèm bản .npz của 9 mẫu chuẩn để Bước 3 & Bước 4 tự động tạo Golden Feature Maps
    if len(selected_wafers) > 0:
        npz_samples_path = os.path.join(hw_test_dir, "test_samples_9classes.npz")
        np.savez_compressed(
            npz_samples_path,
            X_samples=np.stack(selected_wafers, axis=0).astype(np.int8),
            y_samples=np.array(selected_labels, dtype=np.int64),
            classes=np.array(CLASSES)
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Bước 1: Tiền xử lý WM-811K 64x64 INT8 cho FPGA")
    parser.add_argument(
        "--pkl_path",
        type=str,
        default=None,
        help="Đường dẫn tới file LSWMD.pkl hoặc wafers.pkl"
    )
    parser.add_argument(
        "--output_npz",
        type=str,
        default=None,
        help="Đường dẫn lưu file dữ liệu đã tiền xử lý (.npz)"
    )
    parser.add_argument(
        "--hw_test_dir",
        type=str,
        default=None,
        help="Thư mục lưu 9 file input_wafer_*.hex cho Verilog Testbench"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=64,
        help="Kích thước ảnh sau khi resize bằng Nearest-Neighbor (mặc định: 64)"
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=8000,
        help="Số lượng mẫu mục tiêu trên mỗi lớp sau cân bằng (mặc định: 8000, đặt 0 nếu giữ nguyên)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (mặc định: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    software_dir = os.path.abspath(os.path.join(script_dir, ".."))

    pkl_path = args.pkl_path if args.pkl_path else resolve_default_pkl_path()
    output_npz = args.output_npz if args.output_npz else os.path.join(software_dir, "data", "wm811k_64x64_int8.npz")
    hw_test_dir = args.hw_test_dir if args.hw_test_dir else os.path.join(software_dir, "Explicit_Model", "hw_vectors", "test_samples")

    print("=" * 72)
    print("  BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU WM-811K (64x64 NEAREST-NEIGHBOR INT8)")
    print("=" * 72)

    X, y = load_and_preprocess_pkl(
        pkl_path=pkl_path,
        img_size=args.img_size,
        target_per_class=args.samples_per_class if args.samples_per_class > 0 else None,
        seed=args.seed
    )

    # Kiểm tra tính toàn vẹn vật lý: chỉ tồn tại đúng 3 giá trị {0, 64, 127}
    unique_vals = np.unique(X)
    assert set(unique_vals.tolist()).issubset({0, 64, 127}), (
        f"Lỗi: Phát hiện giá trị ngoài tập vật lý {{0, 64, 127}}: {unique_vals}"
    )

    # Chia tập Train (70%), Validation (15%), Test (15%) có phân tầng (Stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=args.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=args.seed
    )

    # Mã hóa One-Hot
    y_train_onehot = to_one_hot(y_train, len(CLASS_TO_IDX))
    y_val_onehot = to_one_hot(y_val, len(CLASS_TO_IDX))
    y_test_onehot = to_one_hot(y_test, len(CLASS_TO_IDX))

    os.makedirs(os.path.dirname(os.path.abspath(output_npz)), exist_ok=True)
    np.savez_compressed(
        output_npz,
        X_train=X_train,
        y_train=y_train,
        y_train_onehot=y_train_onehot,
        X_val=X_val,
        y_val=y_val,
        y_val_onehot=y_val_onehot,
        X_test=X_test,
        y_test=y_test,
        y_test_onehot=y_test_onehot,
        classes=np.array(CLASSES)
    )

    print(f"\n      Đã lưu bộ dữ liệu chuẩn hóa tại: {output_npz}")
    print(f"        - Train:      {X_train.shape} | One-hot: {y_train_onehot.shape}")
    print(f"        - Validation: {X_val.shape} | One-hot: {y_val_onehot.shape}")
    print(f"        - Test:       {X_test.shape} | One-hot: {y_test_onehot.shape}")
    print(f"        - Tập giá trị INT8 hợp lệ: {unique_vals.tolist()}")

    # Xuất 9 mẫu test ra file .hex cho RTL Testbench
    export_9_test_sample_vectors(X_test, y_test, hw_test_dir)
    print("\nHoàn tất Bước 1!")


if __name__ == "__main__":
    main()
