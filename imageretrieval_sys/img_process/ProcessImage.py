import torch
from fastapi import FastAPI, File, UploadFile
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import spacy
from fastapi.middleware.cors import CORSMiddleware

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

nlp = spacy.load("en_core_web_sm")

# Chọn thiết bị chạy mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải mô hình BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", local_files_only=True)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", local_files_only=True).to(device)

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
        num_beams=7,  # Beam search để tăng độ chính xác
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

    return {
        "filename": file.filename,
        "description": description,
    }
