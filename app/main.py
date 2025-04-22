# app/main.py
from fastapi import FastAPI, File, UploadFile
from app.model_utils import load_models
from app.inference import predict_image

app = FastAPI()

print("🔁 Loading models...")
model_b7, model_xcp = load_models()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_image(image_bytes, model_b7, model_xcp)
    return result
