import torch
from fastapi import FastAPI, File, UploadFile
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import spacy
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import hashlib
from io import BytesIO
from torch.cuda.amp import autocast
import asyncio
import supervision as sv

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

# Tải mô hình YOLOv11n với đường dẫn tuyệt đối
yolo_model = YOLO("D:/Learn/InSchool/NCKH/image_retrieval_sys/Process/img_process/yolo12n.pt")
nlp = spacy.load("en_core_web_sm")

# Chọn thiết bị chạy mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải mô hình BLIP-base
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def preprocess_image(image: Image.Image, max_size=384):
    """Resize ảnh để tối ưu hóa tốc độ"""
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image

def get_image_hash(image: Image.Image) -> str:
    """Tính hash của ảnh"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.md5(buffer.getvalue()).hexdigest()

def detect_main_entity(image: Image.Image):
    """Dùng YOLOv11n để nhận diện tất cả các thực thể và trả về chuỗi tên class dạng 'a, b, c'"""
    # Chạy inference với YOLOv11n
    results = yolo_model(image, conf=0.25, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    if len(detections) > 0:
        entities = []
        img_width, img_height = image.size
        img_center_x, img_center_y = img_width / 2, img_height / 2
        
        for box, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
            class_name = yolo_model.names[class_id]
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            box_center_x = (x_min + x_max) / 2
            box_center_y = (y_min + y_max) / 2
            center_distance = ((box_center_x - img_center_x) ** 2 + (box_center_y - img_center_y) ** 2) ** 0.5
            entities.append({
                "class_name": class_name,
                "area": area,
                "center_distance": center_distance
            })

        # Sắp xếp theo diện tích (ưu tiên lớn) và khoảng cách đến tâm (ưu tiên gần)
        entities.sort(key=lambda x: (-x["area"], x["center_distance"]))
        seen = set()
        main_entity = []
        for entity in entities:
            class_name = entity["class_name"]
            if class_name not in seen:
                seen.add(class_name)
                main_entity.append(class_name)
        
        return ", ".join(main_entity)
    return ""

def extract_key_phrases(text):
    """Trích xuất các từ quan trọng từ mô tả"""
    doc = nlp(text)
    keywords = []
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}:
            keywords.append(token.lemma_)
    return " ".join(keywords)

def generate_description(image: Image.Image) -> str:
    """Sinh mô tả tập trung vào thực thể chính trong ảnh"""
    # Bỏ caching Redis
    inputs = blip_processor(image, return_tensors="pt").to(device)
    with autocast():  # Mixed precision
        output = blip_model.generate(
            **inputs,
            max_length=30,
            num_beams=1,  # Greedy decoding
            repetition_penalty=1.2,
            early_stopping=True
        )
    description = blip_processor.decode(output[0], skip_special_tokens=True)
    return description

async def detect_main_entity_async(image: Image.Image):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, detect_main_entity, image)

async def generate_description_async(image: Image.Image):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_description, image)

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    """API nhận ảnh và trả về mô tả tập trung vào thực thể"""
    image = Image.open(file.file).convert("RGB")
    image = preprocess_image(image)
    
    # Chạy YOLOv11n và BLIP song song
    main_entity_task = detect_main_entity_async(image)
    description_task = generate_description_async(image)
    main_entity, description = await asyncio.gather(main_entity_task, description_task)
    
    return {
        "filename": file.filename,
        "description": description,
        "main_entity": main_entity,
    }