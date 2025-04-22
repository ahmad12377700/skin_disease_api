from fastapi import FastAPI, File, UploadFile
from app.model_utils import load_models
from app.inference import predict_image
import openai
import os

app = FastAPI()

print("🔁 Loading models...")
model_b7, model_xcp = load_models()

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_image(image_bytes, model_b7, model_xcp)
    return result

@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_image(image_bytes, model_b7, model_xcp)
    prediction = result['predicted_label']

    prompt = f"""
    You are Derma AI, a virtual assistant trained to help both patients and dermatologists understand skin lesion conditions detected by an AI classifier.

    A patient has uploaded a skin lesion image. The model predicted the class as: **{prediction.upper()}**.

    Please provide:
    1. An empathetic, easy-to-understand explanation of what {prediction.upper()} means for the patient.
    2. If and when they should see a dermatologist.
    3. General guidance and home care tips (if safe).
    4. For doctors: include a short clinical summary (causes, risks, dermatoscopic features).
    5. Make it clear this is not a diagnosis, but educational support from Derma AI.

    Be interactive, helpful, and supportive. Format it clearly.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are Derma AI — a helpful, safe, and friendly medical explainer for skin lesion predictions."},
            {"role": "user", "content": prompt}
        ]
    )

    explanation = response.choices[0].message.content.strip()
    result['derma_ai_explanation'] = explanation
    return result
