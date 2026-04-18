from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import io
import base64
import cv2
import json

# Internal Logic Imports
from ensemble import load_models, ensemble_predict
from gradcam_utils import generate_gradcam

app = FastAPI(title="Clothing Classification CV Architecture API")

# Allow Cross-Origin connectivity for React dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration
MAP_PATH = "../class_mapping.json"
if os.path.exists(MAP_PATH):
    with open(MAP_PATH) as f:
        CLASS_NAMES = list(json.load(f).values())
else:
    CLASS_NAMES = [f"Class_{i}" for i in range(20)]

# Initialize Architecture Suite (using pre-trained weights/ cluster)
# Note: Ensure you run training first or have established checkpoints
models = load_models(num_classes=len(CLASS_NAMES))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def encode_image(img_array):
    """Auxiliary to convert architectural heatmaps to base64 for web transport."""
    # Convert RGB to BGR for OpenCV
    bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', bgr_img)
    return base64.b64encode(buffer).decode("utf-8")

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "nodes": [name for name, _ in models],
        "device": str(DEVICE)
    }

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    """Inference endpoint for synchronized ensemble consensus."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    pred_idx, conf, _ = ensemble_predict(models, image_tensor)
    
    return {
        "prediction": CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "Unknown",
        "confidence": float(conf)
    }

@app.post("/explain")
async def explain_api(file: UploadFile = File(...)):
    """Explanability endpoint for multi-model architectural saliency (Grad-CAM)."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    rgb_image = np.array(image.resize((224, 224)))
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    explanations = {}

    for name, model in models:
        # Clone tensor for gradient safety
        cam_tensor = transform(image).unsqueeze(0).to(DEVICE)
        cam_img = generate_gradcam(model, name, cam_tensor, rgb_image)
        explanations[name] = encode_image(cam_img)

    return explanations
