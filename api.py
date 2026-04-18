import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from inference import load_model, predict

app = FastAPI()

try:
    model, idx_to_label = load_model("weights/resnet50.pth", "class_mapping.json")
except Exception as e:
    print(
        f"Failed to load checkpoint on API startup. Please ensure models are fully trained in weights/ dir: {e}"
    )
    model, idx_to_label = None, {}


@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    if model is None:
        return {
            "error": "API Error: No trained weights detected locally. Execute train.py"
        }

    image = Image.open(file.file).convert("RGB")

    result = predict(image, model, idx_to_label)

    return result
