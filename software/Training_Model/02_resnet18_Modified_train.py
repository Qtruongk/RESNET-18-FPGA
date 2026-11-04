"""
02_resnet18_Modified_train.py
=============================
Bước 2: Chỉnh sửa Kiến trúc ResNet-18 cho ảnh Wafer 64x64x1, Huấn luyện FP32
        và thực hiện Gộp lớp Batch Normalization (BatchNorm Folding).

Đặc tả kỹ thuật:
A. Sửa đổi cấu trúc mạng ResNet-18 cho ảnh 64x64x1:
   - Thay Conv1 gốc (7x7, stride=2) bằng:
     Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
   - Xóa bỏ hoàn toàn tầng MaxPool2d ngay sau Conv1 (giữ nguyên độ phân giải 64x64 để không
     làm mất các vết xước mảnh Scratch).
   - Kích thước Feature Map qua các Stage:
     + Conv1 + Stage 1 (2 BasicBlocks, 64 kênh):  64 x 64
     + Stage 2 (2 BasicBlocks, 128 kênh, s=2):    32 x 32
     + Stage 3 (2 BasicBlocks, 256 kênh, s=2):    16 x 16
     + Stage 4 (2 BasicBlocks, 512 kênh, s=2):     8 x 8
     + Global Average Pooling (8x8): Tổng 64 điểm ảnh chia 64 (tương đương >> 6 trên FPGA).
     + FC Layer: 512 -> 9 lớp lỗi WM-811K.

B. Gộp lớp Batch Normalization (BatchNorm Folding):
   - Gộp (gamma, beta, running_mean, running_var) vào thẳng W và b của lớp Conv trước nó:
     W_fold = W * (gamma / sqrt(var + eps))
     b_fold = (b - mean) * (gamma / sqrt(var + eps)) + beta
   - Kiểm chứng sai số tuyệt đối giữa mô hình trước và sau khi gộp (< 1e-5).
   - Xuất trọng số đã gộp ra: checkpoints/resnet18_fp32_folded.pth
"""

import os
import time
import json
import copy
import argparse
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable


# =====================================================================
# 1. ĐỊNH NGHĨA NHÃN & THÔNG SỐ CHUẨN
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

# Hệ số tỷ lệ đầu vào: x_fp32 = q_x0 * INPUT_SCALE, với q_x0 thuộc {0, 64, 127}
INPUT_SCALE: float = 1.0 / 127.0


# =====================================================================
# 2. KIẾN TRÚC RESNET-18 SỬA ĐỔI (64x64x1) - TRƯỚC KHI GỘP BATCHNORM
# =====================================================================
class ModifiedBasicBlock(nn.Module):
    """
    Khối BasicBlock chuẩn cho ResNet-18 (trước khi gộp BatchNorm).
    Nhánh chính: Conv3x3 -> BN1 -> ReLU -> Conv3x3 -> BN2
    Nhánh tắt:   Identity hoặc Conv1x1(stride=2) -> BN_sc
    Đầu ra:      ReLU(Nhánh chính + Nhánh tắt)
    """
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * self.expansion)
            )

        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu_out(out)
        return out


class ResNet18Modified(nn.Module):
    """
    Kiến trúc ResNet-18 sửa đổi chuyên biệt cho ảnh Wafer Map 64x64x1:
    - Conv1: 3x3, stride=1, padding=1, in_channels=1, out_channels=64
    - Không có MaxPool2d sau Conv1
    - 4 Stage giữ kích thước: 64x64 -> 32x32 -> 16x16 -> 8x8
    - Global Average Pooling (8x8): Chia 64 (tương đương dịch phải >> 6 trên phần cứng)
    """
    def __init__(self, num_classes: int = 9, in_channels: int = 1):
        super().__init__()
        self.in_planes = 64

        # Tầng đầu vào đã sửa đổi (bỏ 7x7 s=2 và bỏ MaxPool)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 4 Stages (tổng cộng 8 BasicBlocks = 16 lớp Conv3x3 + 3 lớp Shortcut Conv1x1)
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)   # 64 x 64 x 64
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)  # 128 x 32 x 32
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)  # 256 x 16 x 16
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)  # 512 x 8 x 8

        # Tầng phân loại cuối (FC)
        self.fc = nn.Linear(512, num_classes, bias=True)

        self._initialize_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ModifiedBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * ModifiedBasicBlock.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem: (B, 1, 64, 64) -> (B, 64, 64, 64)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 4 Residual Stages
        out = self.layer1(out)  # (B, 64, 64, 64)
        out = self.layer2(out)  # (B, 128, 32, 32)
        out = self.layer3(out)  # (B, 256, 16, 16)
        out = self.layer4(out)  # (B, 512, 8, 8)

        # Global Average Pooling 8x8: tổng 64 phần tử chia 64 (trên FPGA là >> 6)
        out = out.sum(dim=(2, 3)) / 64.0  # (B, 512)

        # Classifier
        out = self.fc(out)  # (B, 9)
        return out


# =====================================================================
# 3. KIẾN TRÚC RESNET-18 ĐÃ GỘP BATCHNORM (BATCHNORM FOLDED MODEL)
# =====================================================================
class FoldedBasicBlock(nn.Module):
    """
    Khối BasicBlock sau khi đã gộp hoàn toàn BatchNorm vào trọng số W_fold và b_fold của Conv2d.
    Mọi lớp Conv2d trong khối đều có bias=True và không còn tầng BatchNorm2d nào.
    """
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.has_shortcut_conv = (stride != 1 or in_planes != planes * self.expansion)
        if self.has_shortcut_conv:
            self.shortcut_conv = nn.Conv2d(
                in_planes, planes * self.expansion,
                kernel_size=1, stride=stride, bias=True
            )
        else:
            self.shortcut_conv = None

        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut_conv(x) if self.shortcut_conv is not None else x

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)

        out = out + identity
        out = self.relu_out(out)
        return out


class ResNet18ModifiedFolded(nn.Module):
    """
    Mô hình ResNet-18 64x64x1 đã thực hiện BatchNorm Folding.
    Gồm đúng 20 lớp Conv2d (17 lớp Conv nhánh chính + 3 lớp Conv 1x1 nhánh tắt) và 1 lớp FC.
    """
    def __init__(self, num_classes: int = 9, in_channels: int = 1):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.fc = nn.Linear(512, num_classes, bias=True)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(FoldedBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * FoldedBasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.sum(dim=(2, 3)) / 64.0
        out = self.fc(out)
        return out


# =====================================================================
# 4. THUẬT TOÁN GỘP BATCH NORMALIZATION (BATCHNORM FOLDING)
# =====================================================================
@torch.no_grad()
def compute_folded_conv_params(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tính toán trọng số W_fold và hệ số chệch b_fold theo công thức toán học:
        scale = gamma / sqrt(var + eps)
        W_fold = W * scale
        b_fold = (b - mean) * scale + beta
    """
    w = conv.weight.detach().clone().float()
    if conv.bias is not None:
        b = conv.bias.detach().clone().float()
    else:
        b = torch.zeros(conv.out_channels, dtype=torch.float32, device=w.device)

    gamma = bn.weight.detach().clone().float()
    beta = bn.bias.detach().clone().float()
    mean = bn.running_mean.detach().clone().float()
    var = bn.running_var.detach().clone().float()
    eps = bn.eps

    scale = gamma / torch.sqrt(var + eps)
    w_fold = w * scale.view(-1, 1, 1, 1)
    b_fold = (b - mean) * scale + beta

    return w_fold, b_fold


@torch.no_grad()
def fold_resnet18_batchnorm(unfolded_model: ResNet18Modified) -> ResNet18ModifiedFolded:
    """
    Chuyển đổi toàn bộ mô hình ResNet18Modified (có BN) sang ResNet18ModifiedFolded (đã gộp BN).
    """
    unfolded_model.eval()
    device = next(unfolded_model.parameters()).device
    folded_model = ResNet18ModifiedFolded(
        num_classes=unfolded_model.fc.out_features,
        in_channels=unfolded_model.conv1.in_channels
    ).to(device)

    # 1. Gộp tầng Stem: conv1 + bn1
    w_fold, b_fold = compute_folded_conv_params(unfolded_model.conv1, unfolded_model.bn1)
    folded_model.conv1.weight.copy_(w_fold)
    folded_model.conv1.bias.copy_(b_fold)

    # 2. Gộp 4 Stages (layer1 .. layer4)
    for stage_name in ["layer1", "layer2", "layer3", "layer4"]:
        src_stage = getattr(unfolded_model, stage_name)
        dst_stage = getattr(folded_model, stage_name)
        for src_block, dst_block in zip(src_stage, dst_stage):
            # Gộp conv1 + bn1 trong BasicBlock
            w1, b1 = compute_folded_conv_params(src_block.conv1, src_block.bn1)
            dst_block.conv1.weight.copy_(w1)
            dst_block.conv1.bias.copy_(b1)

            # Gộp conv2 + bn2 trong BasicBlock
            w2, b2 = compute_folded_conv_params(src_block.conv2, src_block.bn2)
            dst_block.conv2.weight.copy_(w2)
            dst_block.conv2.bias.copy_(b2)

            # Gộp nhánh tắt (Shortcut 1x1 Conv + BN) nếu có
            if dst_block.has_shortcut_conv:
                sc_conv = src_block.shortcut[0]
                sc_bn = src_block.shortcut[1]
                w_sc, b_sc = compute_folded_conv_params(sc_conv, sc_bn)
                dst_block.shortcut_conv.weight.copy_(w_sc)
                dst_block.shortcut_conv.bias.copy_(b_sc)

    # 3. Sao chép tầng FC cuối
    folded_model.fc.weight.copy_(unfolded_model.fc.weight.detach())
    folded_model.fc.bias.copy_(unfolded_model.fc.bias.detach())

    folded_model.eval()
    return folded_model


# =====================================================================
# 5. DATASET LOADER CHO ẢNH WAFER 64x64x1 INT8
# =====================================================================
class Wafer64Dataset(Dataset):
    """
    Dataset cung cấp ảnh Wafer (1, 64, 64) ở định dạng FP32 đã nhân với INPUT_SCALE (1/127.0)
    để huấn luyện mô hình FP32 đồng nhất tuyệt đối với miền INT8 {0, 64, 127}.
    """
    def __init__(self, X_int8: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X_int8 = X_int8.astype(np.int8)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        wafer = self.X_int8[idx]  # shape (1, 64, 64), giá trị thuộc {0, 64, 127}

        if self.augment:
            # Xoay 90/180/270 độ và lật gương ngẫu nhiên (bảo toàn 100% giá trị vật lý)
            k = np.random.randint(0, 4)
            wafer = np.rot90(wafer, k=k, axes=(1, 2))
            if np.random.rand() < 0.5:
                wafer = np.flip(wafer, axis=2)
            if np.random.rand() < 0.5:
                wafer = np.flip(wafer, axis=1)
            wafer = np.ascontiguousarray(wafer)

        # Chuyển sang FP32 bằng cách nhân với INPUT_SCALE = 1 / 127.0
        # Khi lượng tử hóa ở Bước 3 với S_x0 = 1/127.0: round(x_fp32 / S_x0) = {0, 64, 127} chính xác!
        x_fp32 = torch.from_numpy(wafer).float() * INPUT_SCALE
        label = int(self.y[idx])
        return x_fp32, label


def load_preprocessed_data(npz_path: str, batch_size: int = 64, num_workers: int = 0):
    """Đọc file wm811k_64x64_int8.npz từ Bước 1 và khởi tạo DataLoader."""
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu đã tiền xử lý tại: {npz_path}\n"
            f"Hãy chạy 01_wm811k_preprocess.py trước để tạo file này."
        )
    data = np.load(npz_path, allow_pickle=True)
    train_ds = Wafer64Dataset(data["X_train"], data["y_train"], augment=True)
    val_ds = Wafer64Dataset(data["X_val"], data["y_val"], augment=False)
    test_ds = Wafer64Dataset(data["X_test"], data["y_test"], augment=False)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    return train_loader, val_loader, test_loader


# =====================================================================
# 6. VÒNG LẶP HUẤN LUYỆN & KIỂM ĐỊNH BATCHNORM FOLDING
# =====================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]")
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct / total) * 100:.2f}%"})

    return running_loss / total, (correct / total) * 100.0


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, (correct / total) * 100.0


def parse_args():
    parser = argparse.ArgumentParser(description="Bước 2: Huấn luyện Modified ResNet-18 (64x64x1) & BatchNorm Folding")
    parser.add_argument(
        "--npz_path",
        type=str,
        default=None,
        help="Đường dẫn tới file wm811k_64x64_int8.npz (mặc định: ./data/wm811k_64x64_int8.npz)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Số lượng epochs huấn luyện FP32 (mặc định: 25)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (mặc định: 64)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate khởi tạo cho AdamW (mặc định: 1e-3)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Thư mục lưu checkpoints/resnet18_fp32_folded.pth"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị chạy (cuda hoặc cpu)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(os.path.dirname(script_dir)).lower() == "software":
        base_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "phase1_golden_model"))
    else:
        base_dir = script_dir

    npz_path = args.npz_path if args.npz_path else os.path.join(base_dir, "data", "wm811k_64x64_int8.npz")
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir else os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device(args.device)
    print("=" * 72)
    print("  BƯỚC 2: HUẤN LUYỆN MODIFIED RESNET-18 (64x64x1) & BATCHNORM FOLDING")
    print("=" * 72)
    print(f"  Device:          {device}")
    print(f"  Input Format:    64x64x1 (INT8 {{0, 64, 127}} * INPUT_SCALE=1/127)")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch Size:      {args.batch_size}")
    print(f"  Learning Rate:   {args.lr}")
    print("=" * 72)

    # 1. Nạp dữ liệu
    train_loader, val_loader, test_loader = load_preprocessed_data(
        npz_path=npz_path,
        batch_size=args.batch_size
    )

    # 2. Khởi tạo mô hình Modified ResNet-18
    model = ResNet18Modified(num_classes=len(CLASS_TO_IDX), in_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Khởi tạo thành công ResNet18Modified | Tổng tham số: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    unfolded_ckpt_path = os.path.join(ckpt_dir, "resnet18_fp32_unfolded.pth")
    folded_ckpt_path = os.path.join(ckpt_dir, "resnet18_fp32_folded.pth")

    # 3. Huấn luyện FP32
    print("\n  [1/3] Bắt đầu huấn luyện FP32...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_model_weights,
                "best_val_acc": best_val_acc,
                "input_scale": INPUT_SCALE,
                "class_to_idx": CLASS_TO_IDX
            }, unfolded_ckpt_path)

    # Khôi phục trọng số tốt nhất
    model.load_state_dict(best_model_weights)
    _, test_acc_unfolded = evaluate_model(model, test_loader, criterion, device)
    print(f"\n  [2/3] Kết quả FP32 (Trước khi gộp BN) trên tập Test: {test_acc_unfolded:.2f}%")

    # 4. Thực hiện BatchNorm Folding
    print("\n  [3/3] Đang thực hiện BatchNorm Folding (gộp BN vào 20 lớp Conv)...")
    folded_model = fold_resnet18_batchnorm(model)

    # Kiểm tra độ lệch tuyệt đối giữa mô hình gốc và mô hình đã gộp BN
    sample_x, _ = next(iter(test_loader))
    sample_x = sample_x.to(device)
    with torch.no_grad():
        out_unfolded = model(sample_x)
        out_folded = folded_model(sample_x)
        max_diff = torch.max(torch.abs(out_unfolded - out_folded)).item()

    _, test_acc_folded = evaluate_model(folded_model, test_loader, criterion, device)
    print(f"        Sai số tuyệt đối lớn nhất (Max |FP32 - Folded|): {max_diff:.2e}")
    print(f"        Độ chính xác Test sau khi gộp BatchNorm:         {test_acc_folded:.2f}%")

    # Lưu deliverable #5: checkpoints/resnet18_fp32_folded.pth
    torch.save({
        "model_state_dict": folded_model.state_dict(),
        "test_acc_fp32_folded": test_acc_folded,
        "max_folding_diff": max_diff,
        "input_scale": INPUT_SCALE,
        "class_to_idx": CLASS_TO_IDX,
        "arch": "ResNet18ModifiedFolded_64x64x1"
    }, folded_ckpt_path)

    print(f"\n  Đã lưu trọng số đã gộp BatchNorm tại: {folded_ckpt_path}")
    print("  Hoàn tất Bước 2!")


if __name__ == "__main__":
    main()
