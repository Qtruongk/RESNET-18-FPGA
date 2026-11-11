+ Training Model --> Tìm ra dc model xịn nhất với dataset mình dùng
+ Low_Precision_Model --> Chỉnh lại model dùng fixed point hoặc quantized model
+ Explicit_Model --> Viết tường minh model để hỗ trợ Verification cho cái RTL
-----------

Ở đây đề tài này chọn Resnet18
Lựa chọn mô hình: ResNet18 được chọn nhờ sự cân bằng giữa độ chính xác và hiệu quả tính toán, phù hợp với các hạn chế của FPGA.

Dataset: WM-811K

---

## 🛠 Hướng dẫn cài đặt môi trường (Environment Setup)

### 1. Khởi tạo môi trường ảo Python (Python 3.10 - 3.12 khuyến nghị)
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo:
# Trên Windows (PowerShell/CMD):
.venv\Scripts\activate
# Trên Linux / macOS:
source .venv/bin/activate
```

### 2. Cài đặt các gói phụ thuộc (Dependencies)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Lưu ý về PyTorch (Nếu sử dụng GPU NVIDIA):**  
> Để huấn luyện nhanh hơn trên GPU với CUDA, bạn có thể cài đặt PyTorch phiên bản CUDA phù hợp từ [pytorch.org](https://pytorch.org/get-started/locally/):  
> Ví dụ với CUDA 12.1:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
