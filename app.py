import streamlit as st
import torch
import numpy as np
import os
import json
from PIL import Image
from torchvision import transforms

# Architectural Logic Imports
from config import BASE_DIR
from ensemble import load_models, ensemble_predict
from gradcam_utils import generate_gradcam

# --- Hardware Allocation ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="👗 Multi-Model Architectural Clothing Classification Dashboard", layout="wide")

st.title("🖥️ Multi-Model Architectural Clothing Classification Dashboard")
st.markdown(
    "Industry-Level Inference Mapping featuring Ensemble Averaging & Real-Time Grad-CAM Saliency heatmaps!"
)

# --- Load Dataset Topology ---
MAP_PATH = "class_mapping.json"
if os.path.exists(MAP_PATH):
    with open(MAP_PATH) as f:
        idx_to_label = json.load(f)
    NUM_CLASSES = len(idx_to_label)
else:
    # Safe fallback if first run
    NUM_CLASSES = 20
    idx_to_label = {str(i): f"Class_{i}" for i in range(20)}

@st.cache_resource
def load_all_models():
    """Consolidated model loading for all architectural nodes."""
    models_dict = {}
    idx_map = {}
    map_path = os.path.join(BASE_DIR, "class_mapping.json")

    for name in ["resnet50", "efficientnet", "mobilenet", "vgg16", "ann", "custom_cnn", "keras_cnn"]:
        # Standardize search across multiple local node structures
        potential_paths = [
            os.path.join(BASE_DIR, f"best_{name}.pth"),
            os.path.join(BASE_DIR, "weights", f"{name}.pth"),
            os.path.join(BASE_DIR, f"best_{name}.h5"),   # Added for Keras
        ]
        
        # Additional heuristics for Keras CNN if specifically named in train_keras.py
        if name == "keras_cnn":
            potential_paths.append(os.path.join(BASE_DIR, "best_keras_model.h5"))

        loaded = False
        for model_path in potential_paths:
            if os.path.exists(model_path) and os.path.exists(map_path):
                try:
                    if model_path.endswith(".h5"):
                        from inference import load_keras_model
                        m, idx_map = load_keras_model(model_path, map_path)
                    else:
                        from inference import load_model
                        m, idx_map = load_model(model_path, map_path, internal_name=name)
                    
                    models_dict[name] = m
                    loaded = True
                    break
                except Exception as e:
                    st.sidebar.error(f"Node conflict on {name}: {e}")

    return models_dict, idx_map

models_dict, idx_to_label = load_all_models()

# --- Sidebar Logic Map ---
st.sidebar.header("🛠️ Local Logic Deploy Setup")
st.sidebar.success(f"ACTIVE HARDWARE COMPUTE ON GPU: {str(torch.cuda.is_available()).upper()}")
st.sidebar.text(f"Mapping Matrix -> {str(DEVICE).upper()}")

st.sidebar.markdown("### 🧬 Architecture Node Status")
for name in ["resnet50", "efficientnet", "mobilenet", "vgg16", "ann", "custom_cnn", "keras_cnn"]:
    display_name = "Keras Standard CNN" if name == "keras_cnn" else name.upper()
    status = "🟢 ACTIVE" if name in models_dict else "🔴 INACTIVE"
    st.sidebar.markdown(f"**{display_name}**: {status}")

model_choice = st.sidebar.selectbox(
    "Select Structural Backend Processor:",
    [
        "Ensemble (ResNet + EfficientNet + MobileNet)",
        "ResNet-50",
        "EfficientNet-B3",
        "MobileNet-V3",
        "VGG-16",
        "Keras Standard CNN",
        "Legacy ANN",
        "Legacy Custom CNN",
    ],
)

# --- Transform Pipeline ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- MAIN UI ENTRY POINT ---
st.subheader("💡 Dynamic Native Target Identification")
uploaded_file = st.file_uploader("Upload Image to Architecture Suite", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Raw Pre-Inference Canvas Input", use_container_width=True)
    
    with col2:
        with st.spinner(f"Initiating evaluation sequence bounds natively on {DEVICE}..."):
            internal_keys = {
                "ResNet-50": "resnet50",
                "EfficientNet-B3": "efficientnet",
                "MobileNet-V3": "mobilenet",
                "VGG-16": "vgg16",
                "Keras Standard CNN": "keras_cnn",
                "Legacy ANN": "ann",
                "Legacy Custom CNN": "custom_cnn",
            }

            if "Ensemble" in model_choice:
                ensemble_keys = ["resnet50", "efficientnet", "mobilenet"]
                ensemble_targets = [models_dict[k] for k in ensemble_keys if k in models_dict]
                
                if not ensemble_targets:
                    st.warning("Ensemble Matrix Failed: All candidate architecture block states are uninitialized. Await `train.py` convergence.")
                else:
                    try:
                        from inference import ensemble_predict
                        results = ensemble_predict(image, ensemble_targets, idx_to_label)
                        st.success("Synchronized Ensemble Sequence Completed Successfully!")
                        st.metric(label="Consensus Predicted Output Target:", value=str(results["class"]).upper())
                        st.metric(label="Weighted Component Confidence Levels:", value=f"{results['confidence'] * 100:.2f}%")
                    except Exception as e:
                        st.error(f"Backend Node Failure: {e}")
            else:
                target_key = internal_keys[model_choice]
                if target_key not in models_dict:
                    st.warning(f"**Node Isolation Detected**: The weights for `{target_key}` are not yet initialized. Please ensure `train.py` or `train_keras.py` has completed.")
                else:
                    try:
                        model_to_use = models_dict[target_key]
                        if target_key == "keras_cnn":
                            from inference import keras_predict
                            results = keras_predict(image, model_to_use, idx_to_label)
                        else:
                            from inference import predict
                            results = predict(image, model_to_use, idx_to_label, generate_cam=True)
                        
                        st.success(f"Inference resolved cleanly over `{model_choice}`.")
                        st.metric(label="Probability Class Assessment Output:", value=str(results["class"]).upper())
                        st.metric(label="Architecture Internal Confidence:", value=f"{results['confidence'] * 100:.2f}%")
                        
                        if "cam_image" in results:
                            st.markdown("#### Dynamic Feature Explainability Map (Grad-CAM)")
                            st.image(results["cam_image"], caption=f"Model Spatial Target Attention ({model_choice})", use_container_width=True)
                    except Exception as e:
                        st.error(f"Execution mapping pipeline blocked conditionally: {e}")

st.markdown("---")
st.markdown("*Note: All architectural results are derived from the local weights/ cluster.*")
