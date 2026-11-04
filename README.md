+ Training Model --> Tìm ra dc model xịn nhất với dataset mình dùng
+ Low_Precision_Model --> Chỉnh lại model dùng fixed point hoặc quantized model
+ Explicit_Model --> Viết tường minh model để hỗ trợ Verification cho cái RTL
-----------

Ở đây đề tài này chọn Resnet18
Lựa chọn mô hình: ResNet18 được chọn nhờ sự cân bằng giữa độ chính xác và hiệu quả tính toán, phù hợp với các hạn chế của FPGA.

Dataset: WM-811K