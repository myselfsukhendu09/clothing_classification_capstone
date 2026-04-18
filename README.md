# 👗 Multi-Model Architectural Clothing Classification Dashboard

**Industry-Grade Computer Vision Ecosystem for Automated Apparel Categorization & Explainable AI**

---

## 🚀 Overview
This project delivers a comprehensive, full-stack Computer Vision solution for high-accuracy clothing classification. Utilizing a **synchronized multi-model architectural cluster**, the system leverages consensus-based inference (Ensemble) and real-time saliency mapping (Grad-CAM) to provide both stable and explainable results.

### 🍱 Architectural Features
*   **Ensemble Consensus Integration**: Combines ResNet-50, EfficientNet-B0, and MobileNet-V3 for a robust, redundant classification engine.
*   **Real-Time Explainability (Grad-CAM)**: High-fidelity spatial attention maps to debug and validate model focus regions.
*   **Dataset Intelligence (EDA)**: Automated topology mapping including distribution analysis and imbalance detection.
*   **Deployment-Ready Backend**: GPU-accelerated FastAPI server for low-latency external connectivity.
*   **Production Dashboard**: Streamlit-based UI featuring live inference mapping and saliency parallel views.

---

## 🛠️ Project Structure
```text
project/
│
├── app.py                      # Multi-Model Production Dashboard (Streamlit)
├── train.py                    # Architectural Synchronization Logic (Retraining)
├── models.py                   # Model Cluster Definitions (ResNet, EfficientNet, MobileNet)
├── eda.py                      # Dataset Intelligence Sequence
├── ensemble.py                 # Multi-model Consensus Predictor
├── gradcam_utils.py            # Real-time Saliency Mapping (Explainability)
├── utils.py                    # Optimized Data Pipline Loaders
├── backend/
│   └── main.py                 # GPU-Accelerated FastAPI Logic
├── outputs/                    # Benchmarks, EDA Plots, and Distributions
└── weights/                    # Active Architectural Nodes (Model Weights)
```

---

## 📈 Performance Engineering
The system is optimized for high-throughput GPU inference and focuses on architectural generalization:
*   **Augmentation Policy**: Random Resized Crops, Horizontal Flips, and Color Jittering.
*   **Loss Strategy**: Cross-Entropy with Label Smoothing (0.1) to mitigate dataset imbalance effects.
*   **Optimization**: AdamW with weight decay (1e-2) for superior generalization on edge-cases.

---

## 🚦 Execution Sequence
To deploy the system locally, follow this professional sequence:

### 1️⃣ Dataset Intelligence
Scan the topology and validate the input distribution.
```bash
python eda.py
```

### 2️⃣ Architectural Synchronization
Train/Update the model cluster (ResNet, EfficientNet, MobileNet).
```bash
python model_comparison_chart.py
```

### 3️⃣ Local Dashboard Deployment
Launch the interactive production interface.
```bash
streamlit run app.py
```

### 4️⃣ API Deployment (Optional)
Deploy the GPU-accelerated inference backend.
```bash
uvicorn backend.main:app --reload
```

---

## 🔬 Explainable AI (XAI)
The dashboard features **Grad-CAM (Gradient-weighted Class Activation Mapping)**. This allows users to visualize which spatial features (collars, sleeves, patterns) the architecture prioritized for its consensus decision, moving beyond the "black box" model paradigm.

---
**Architectural Verification Logic Powered by PyTorch & FastAPI.**
