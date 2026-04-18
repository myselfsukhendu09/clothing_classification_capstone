import torch
import numpy as np
import cv2
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    try:
        from grad_cam import GradCAM
        from grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("[Warning] Grad-CAM visualization library not detected. Feature suppressed.")
        GradCAM = None

def get_target_layer(model, model_name):
    """Dynamically target architecture-specific saliency layers."""
    if model_name == "resnet50":
        # Final residual block layer
        return model.layer4[-1]
    elif model_name == "efficientnet":
        # Final functional feature extractor block
        return model.features[-1]
    elif model_name == "mobilenet":
        # Final inverted residual block
        return model.features[-1]
    else:
        raise ValueError(f"Unknown architecture target: {model_name}")

def generate_gradcam(model, model_name, image_tensor, rgb_image):
    """Generate industrial-grade Grad-CAM attention heatmap."""
    if GradCAM is None:
        return rgb_image

    try:
        model.eval()
        target_layer = get_target_layer(model, model_name)

        cam = GradCAM(model=model, target_layers=[target_layer])

        # Enable grad logic contextually for saliency mapping
        with torch.enable_grad():
            grayscale_cam = cam(input_tensor=image_tensor)[0]

        # Rescale heatmap and blend with original image
        visualization = show_cam_on_image(
            rgb_image.astype(np.float32) / 255.0,
            grayscale_cam,
            use_rgb=True
        )

        return visualization
    except Exception as e:
        print(f"[Warning] Failed to generate Explanatory Grad-CAM trace for {model_name}: {e}")
        return rgb_image
