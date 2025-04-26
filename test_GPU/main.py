import torch
import sys
from packaging import version

def check_gpu_availability():
    print("=== Kiểm tra khả năng chạy BLIP trên GPU ===")
    
    # Kiểm tra PyTorch
    if not torch.__version__:
        print("Lỗi: PyTorch không được cài đặt. Vui lòng cài đặt PyTorch trước.")
        return False
    
    # Kiểm tra CUDA
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("Không tìm thấy GPU hoặc CUDA không được cài đặt.")
        print("BLIP sẽ chỉ chạy trên CPU, điều này có thể chậm hơn đáng kể.")
        print("Hướng dẫn khắc phục:")
        print("1. Cài đặt driver NVIDIA mới nhất: https://www.nvidia.com/Download/index.aspx")
        print("2. Cài đặt PyTorch với hỗ trợ CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    # Thông tin GPU
    print("GPU được phát hiện!")
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
    print(f"Số lượng GPU: {torch.cuda.device_count()}")
    
    # Kiểm tra bộ nhớ GPU
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    print(f"Bộ nhớ GPU tổng cộng: {total_memory:.2f} GB")
    
    # Đánh giá cho GTX 1650 Max-Q
    if total_memory < 4.5:
        print("Cảnh báo: Bộ nhớ GPU 4GB (GTX 1650 Max-Q) có thể hạn chế khi chạy BLIP.")
        print("Khuyến nghị: Sử dụng mô hình 'Salesforce/blip-image-captioning-base' và batch size = 1.")
        print("Tối ưu hóa: Đóng các ứng dụng khác để giải phóng VRAM.")
    else:
        print("Bộ nhớ GPU đủ để chạy BLIP base hiệu quả!")
    
    # Kiểm tra phiên bản CUDA
    cuda_version = torch.version.cuda
    print(f"Phiên bản CUDA: {cuda_version}")
    
    # Kiểm tra PyTorch
    pytorch_version = torch.__version__
    print(f"Phiên bản PyTorch: {pytorch_version}")
    
    # Kiểm tra tương thích CUDA
    if cuda_version and version.parse(cuda_version) >= version.parse("10.2"):
        print("Phiên bản CUDA tương thích với BLIP.")
    else:
        print("Cảnh báo: Phiên bản CUDA quá cũ. Cập nhật driver NVIDIA và CUDA.")
    
    return True

def check_blip_requirements():
    print("\n=== Kiểm tra yêu cầu để chạy BLIP ===")
    try:
        import transformers
        print(f"Thư viện transformers đã được cài đặt: phiên bản {transformers.__version__}")
    except ImportError:
        print("Thư viện transformers chưa được cài đặt.")
        print("Cài đặt bằng lệnh: pip install transformers")
        return False
    
    try:
        import PIL
        print("Thư viện Pillow (PIL) đã được cài đặt.")
    except ImportError:
        print("Thư viện Pillow (PIL) chưa được cài đặt.")
        print("Cài đặt bằng lệnh: pip install Pillow")
        return False
    
    print("Tất cả các thư viện cần thiết đã được cài đặt!")
    return True

def provide_blip_setup_instructions():
    print("\n=== Hướng dẫn chạy BLIP trên GTX 1650 Max-Q ===")
    print("1. Driver NVIDIA đã được cài đặt (phiên bản 572.60).")
    print("2. PyTorch với CUDA 12.1 đã được cài đặt:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("3. Đã cài đặt transformers và Pillow:")
    print("   pip install transformers Pillow")
    print("4. Chạy BLIP với mô hình base:")
    print("   ```python")
    print("   from transformers import BlipProcessor, BlipForConditionalGeneration")
    print("   import torch")
    print("   processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')")
    print("   model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')")
    print("   model = model.to('cuda')")
    print("   model.eval()  # Chuyển sang chế độ đánh giá")
    print("   torch.cuda.empty_cache()  # Giải phóng bộ nhớ GPU")
    print("   ```")
    print("5. Tối ưu hóa cho 4GB VRAM:")
    print("   - Sử dụng batch size = 1.")
    print("   - Đóng các ứng dụng sử dụng GPU (ví dụ: Brave Browser).")
    print("   - Sử dụng model.half() để chạy ở FP16.")
    print("   - Tránh mô hình 'blip-large' (>6GB VRAM).")

if __name__ == "__main__":
    # Kiểm tra GPU
    gpu_ok = check_gpu_availability()
    
    # Kiểm tra yêu cầu BLIP
    blip_ok = check_blip_requirements()
    
    # Cung cấp hướng dẫn
    if gpu_ok and blip_ok:
        print("\nHệ thống sẵn sàng chạy BLIP trên GPU!")
    else:
        print("\nVui lòng khắc phục các vấn đề được liệt kê.")
    
    provide_blip_setup_instructions()