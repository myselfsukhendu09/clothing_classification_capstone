import io
import json

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import models, transforms

app = FastAPI(title="Professional Clothing Classification API")

# CONFIG
MODEL_PATH = "best_model_run_0.pth"  # Assume run 0 is the best for now
CLASS_MAPPING_FILE = "class_mapping.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classes
with open(CLASS_MAPPING_FILE, "r") as f:
    class_mapping = json.load(f)

NUM_CLASSES = len(class_mapping)

# Transform
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(in_features, NUM_CLASSES)
    )
    # Load weights safely
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# Global model instance for efficiency (Lazy loading)
model = None


@app.on_event("startup")
async def startup_event():
    global model
    import os

    if os.path.exists(MODEL_PATH):
        model = get_model()
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: Model NOT found at {MODEL_PATH}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    class_name = class_mapping.get(str(pred.item()), "Unknown")

    return {
        "class_id": pred.item(),
        "class_name": class_name,
        "confidence": float(conf.item()),
    }


@app.get("/health")
def health():
    return {"status": "alive", "device": str(DEVICE)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
