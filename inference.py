import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from checkpoint_compat import load_checkpoint_strict
from models import get_model

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    pass

try:
    import tensorflow as tf
except ImportError:
    tf = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logic mapping removed as get_model handles it internally
def load_model(model_path, class_map_path, internal_name="resnet50"):
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    with open(class_map_path) as f:
        idx_to_label = json.load(f)

    num_classes = len(idx_to_label)
    model = get_model(internal_name, num_classes)

    load_checkpoint_strict(model, checkpoint, source=model_path)

    model.to(DEVICE)
    if DEVICE.type == "cuda":
        model.half()
    model.eval()

    return model, idx_to_label


def load_keras_model(model_path, class_map_path):
    if tf is None:
        raise ImportError("TensorFlow/Keras is not installed.")

    model = tf.keras.models.load_model(model_path)

    with open(class_map_path) as f:
        idx_to_label = json.load(f)

    return model, idx_to_label


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict(image_source, model, idx_to_label, generate_cam=False):
    if isinstance(image_source, str):
        image = Image.open(image_source).convert("RGB")
    else:
        # Assuming PIL Image
        image = image_source.convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    if DEVICE.type == "cuda":
        image_tensor = image_tensor.half()

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)

    conf, pred = torch.max(probs, 1)

    result = {
        "class": idx_to_label.get(str(pred.item()), "Unknown"),
        "confidence": conf.item(),
    }

    if generate_cam:
        try:
            target_layers = None
            # ResNet-50
            if hasattr(model, "layer4"):
                target_layers = [model.layer4[-1]]
            # EfficientNet / MobileNet / CustomCNN
            elif hasattr(model, "features"):
                if isinstance(model.features, torch.nn.Sequential):
                    # For CustomCNN, we want a Conv layer, usually [6]
                    if len(model.features) > 6 and isinstance(model.features[6], torch.nn.Conv2d):
                        target_layers = [model.features[6]]
                    else:
                        target_layers = [model.features[-1]]
                else:
                    target_layers = [model.features[-1]]

            if target_layers:
                cam = GradCAM(model=model, target_layers=target_layers)
                # Need gradient capabilities, so turn on grad explicitly for CAM
                # Use a specific copy for CAM to avoid float16 issues on some backends
                cam_input = transform(image).unsqueeze(0).to(DEVICE)
                with torch.enable_grad():
                    grayscale_cam = cam(input_tensor=cam_input)[0, :]

                rgb_img = np.float32(image.resize((224, 224))) / 255
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                result["cam_image"] = cam_image
        except Exception as e:
            print(f"[Warning] Failed to generate Explanatory Grad-CAM trace: {e}")

    return result


def ensemble_predict(image_source, models, idx_to_label, generate_cam=True):
    if isinstance(image_source, str):
        image = Image.open(image_source).convert("RGB")
    else:
        image = image_source.convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    if DEVICE.type == "cuda":
        image_tensor = image_tensor.half()
    outputs = []
    cam_images = []

    for model in models:
        model.eval()
        with torch.no_grad():
            outputs.append(torch.softmax(model(image_tensor), dim=1))
        
        if generate_cam:
            try:
                # Reuse predict logic for CAM generation
                res = predict(image, model, idx_to_label, generate_cam=True)
                if "cam_image" in res:
                    cam_images.append(res["cam_image"].astype(np.float32))
            except:
                pass

    avg_output = torch.mean(torch.stack(outputs), dim=0)
    conf, pred = torch.max(avg_output, 1)

    result = {
        "class": idx_to_label.get(str(pred.item()), "Unknown"),
        "confidence": conf.item(),
    }

    if cam_images:
        # Blend multiple Grad-CAM heatmaps
        avg_cam = np.mean(cam_images, axis=0).astype(np.uint8)
        result["cam_image"] = avg_cam

    return result


def keras_predict(image_source, model, idx_to_label, generate_cam=True):
    if isinstance(image_source, str):
        image = Image.open(image_source).convert("RGB")
    else:
        image = image_source.convert("RGB")

    # Keras typically expects [0, 1] range and specific size
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(predictions[0])
    confidence = predictions[0][pred_idx]

    result = {
        "class": idx_to_label.get(str(pred_idx), "Unknown"),
        "confidence": float(confidence),
    }

    if generate_cam and tf is not None:
        try:
            # Simple Keras Grad-CAM implementation
            # Find the last convolutional layer
            last_conv_layer_name = None
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break
            
            if last_conv_layer_name:
                grad_model = tf.keras.models.Model(
                    [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
                )
                
                with tf.GradientTape() as tape:
                    last_conv_layer_output, preds = grad_model(img_array)
                    class_channel = preds[:, pred_idx]
                
                grads = tape.gradient(class_channel, last_conv_layer_output)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                
                last_conv_layer_output = last_conv_layer_output[0]
                heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                
                heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
                heatmap = heatmap.numpy()
                
                import cv2
                heatmap = cv2.resize(heatmap, (224, 224))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                rgb_img = np.array(image_resized)
                superimposed_img = cv2.addWeighted(rgb_img, 0.6, heatmap, 0.4, 0)
                result["cam_image"] = superimposed_img
        except Exception as e:
            print(f"[Warning] Keras Grad-CAM Error: {e}")

    return result
