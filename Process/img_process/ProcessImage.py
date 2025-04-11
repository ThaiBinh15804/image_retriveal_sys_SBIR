import torch
from fastapi import FastAPI, File, UploadFile
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import spacy
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# Khởi tạo FastAPI
app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Giới hạn bộ nhớ GPU
torch.cuda.empty_cache()
torch.set_grad_enabled(False)

yolo_model = YOLO("yolov8n.pt")
nlp = spacy.load("en_core_web_sm")

# Chọn thiết bị chạy mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải mô hình BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", local_files_only=True)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", local_files_only=True).to(device)

def detect_main_entity(image: Image.Image):
    """Dùng YOLOv8 để nhận diện tất cả các thực thể và trả về chuỗi tên class dạng 'a, b, c'"""
    results = yolo_model(image)
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        main_entity = []  # Danh sách để lưu tất cả tên lớp
        
        # Duyệt qua tất cả các hộp giới hạn
        for box in boxes:
            class_id = int(box.cls)  # Lấy ID lớp của đối tượng
            class_name = yolo_model.names[class_id]  # Lấy tên lớp từ ID
            main_entity.append(class_name)  # Thêm tên lớp vào danh sách
        
        # Chuyển danh sách thành chuỗi, nối bằng ", "
        return ", ".join(main_entity)
    return ""


def extract_key_phrases(text):
    """Trích xuất các từ quan trọng từ mô tả"""
    doc = nlp(text)
    keywords = []

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}:  # Chỉ giữ danh từ, động từ, tính từ
            keywords.append(token.lemma_)  # Lấy dạng gốc (lemma)

    return " ".join(keywords)

def generate_description(image: Image.Image) -> str:
    """Sinh mô tả tập trung vào thực thể chính trong ảnh"""
    # prompt = "describe the most important objects and actions in the image in a simple sentence."  # Hướng mô hình tập trung vào thực thể chính

    # inputs = blip_processor(image, text=prompt, return_tensors="pt").to(device)
    inputs = blip_processor(image, return_tensors="pt").to(device)

    output = blip_model.generate(
        **inputs, 
        max_length=50,  # Giới hạn độ dài mô tả
        num_beams=2,  # Beam search để tăng độ chính xác
        repetition_penalty=1.2
    )

    description = blip_processor.decode(output[0], skip_special_tokens=True)

    # if (prompt in description):
    #     description = description.replace(prompt, "").strip('., ')

    return description


@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    """API nhận ảnh và trả về mô tả tập trung vào thực thể"""
    image = Image.open(file.file).convert("RGB")

    # Sinh mô tả ảnh
    description = generate_description(image)

    # Nhận diện thực thể chính
    main_entity = detect_main_entity(image)

    return {
        "filename": file.filename,
        "description": description,
        "main_entity": main_entity,
    }