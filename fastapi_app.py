import io
import json
import os

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from checkpoint_compat import load_checkpoint_strict
from config import CLASS_MAPPING_FILE, IMAGE_SIZE, MODEL_SAVE_PATH
from model import get_model

app = FastAPI(title="Clothing Classification Production Inference API", version="1.0.0")

# Load global definitions on cold start natively
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
idx_to_label = {}


@app.on_event("startup")
def load_assets():
    global model, idx_to_label

    if not os.path.exists(CLASS_MAPPING_FILE) or not os.path.exists(MODEL_SAVE_PATH):
        print(
            "[WARNING] Required model artifacts missing. Ensure you have trained the model first."
        )
        return

    with open(CLASS_MAPPING_FILE, "r") as f:
        idx_to_label = json.load(f)

    num_classes = len(idx_to_label)

    model = get_model("resnet50", num_classes)
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    load_checkpoint_strict(model, checkpoint, source=MODEL_SAVE_PATH)
    model.to(device)
    model.eval()
    print(
        "[INFO] Model checkpoints loaded into VRAM/RAM for fast inference capabilities."
    )


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Classification Model Not Initialized. Re-train required.",
        )

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Image verification failed. Ensure valid image file. {e}",
        )

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probs, 1)
        topk_probs, topk_classes = torch.topk(probs, min(3, probs.size(1)))

    class_idx = str(top_class.item())
    label = idx_to_label.get(class_idx, "Unknown")
    confidence = float(top_prob.item())

    topk_results = []
    for i in range(topk_probs.size(1)):
        c_idx = str(topk_classes[0, i].item())
        c_prob = float(topk_probs[0, i].item())
        topk_results.append(
            {
                "label": idx_to_label.get(c_idx, "Unknown"),
                "confidence": round(c_prob * 100, 2),
            }
        )

    if confidence < 0.6:
        label = f"Uncertain prediction ({label})"

    return {
        "filename": file.filename,
        "prediction_target": label,
        "confidence_ratio": round(confidence * 100, 2),
        "top_k": topk_results,
        "status": "Success",
    }


if __name__ == "__main__":
    print(f"Deploying FastAPI Enterprise Endpoints Natively on Port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
