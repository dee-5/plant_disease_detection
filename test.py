import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import timm

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load Class Names that were saved during training
# ──────────────────────────────────────────────────────────────────────────────
try:
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
except FileNotFoundError:
    print("ERROR: class_names.json not found. Please run the Training block first!")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same standard normalization used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────────────────────────────
# 2. Model Loading Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_mobilenet():
    """Load MobileNetV3-Small with trained weights."""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(torch.load("mobilenet_v3.pth", map_location=device))
    model = model.to(device).eval()
    print("Loaded MobileNetV3-Small from 'mobilenet_v3.pth'")
    return model


def load_efficientnet():
    """Load EfficientNet-Lite0 with trained weights."""
    model = timm.create_model('tf_efficientnet_lite0', pretrained=False, num_classes=len(class_names))
    model.load_state_dict(torch.load("efficientnet_lite.pth", map_location=device))
    model = model.to(device).eval()
    print("Loaded EfficientNet-Lite0 from 'efficientnet_lite.pth'")
    return model


def get_gradcam_target_layers(model, model_name):
    """Return the appropriate GradCAM target layer for each architecture."""
    if model_name == 'efficientnet':
        return [model.conv_head]
    else:  # mobilenet
        return [model.features[-1]]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Test Function
# ──────────────────────────────────────────────────────────────────────────────

def test_image(image_path, model_name='mobilenet'):
    """
    Test a single image with the specified model.
    model_name: 'mobilenet' or 'efficientnet'
    """
    print(f"\nAnalyzing custom image: {image_path}")
    print(f"Using model: {model_name}")

    # Load the selected model
    if model_name == 'efficientnet':
        try:
            model = load_efficientnet()
        except FileNotFoundError:
            print("ERROR: efficientnet_lite.pth not found. Run train.py first!")
            return
    else:
        try:
            model = load_mobilenet()
        except FileNotFoundError:
            print("ERROR: mobilenet_v3.pth not found. Run train.py first!")
            return

    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    input_tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = class_names[predicted_idx.item()]
    display_model = "MobileNetV3-Small" if model_name == 'mobilenet' else "EfficientNet-Lite0"
    
    print(f"==========================================")
    print(f"--> Model     : {display_model}")
    print(f"--> Prediction: {predicted_class}")
    print(f"--> Confidence: {confidence.item()*100:.2f}%")
    print(f"==========================================")

    print("\n[🔍 XAI DIAGNOSTIC EXPLANATION]")
    print("The image on the right is a Grad-CAM 'Heatmap'.")
    print("It explains exactly *why* the AI made this diagnosis by highlighting the pixels it relied on.")
    print(" - RED / YELLOW areas: The AI focused heavily here (e.g., at the exact spot of a leaf lesion or mold).")
    print(" - BLUE / COOL areas: The AI ignored these areas (e.g., the healthy leaf parts or background).")
    print("If the RED areas directly align with the physical disease symptoms on the leaf, you can highly trust this diagnosis!\n")

    # Explainable AI: Grad-CAM
    target_layers = get_gradcam_target_layers(model, model_name)
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_idx.item())]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # Normalize for displaying
    rgb_img = np.array(img.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Display the beautiful comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Your Uploaded Image")
    axes[0].axis('off')

    axes[1].imshow(visualization)
    axes[1].set_title(f"Grad-CAM Heatmap ({display_model})\nDiagnosis: {predicted_class}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def compare_models(image_path):
    """Run both models on the same image and display side-by-side comparison."""
    print(f"\n{'='*60}")
    print(f"  COMPARING BOTH MODELS ON: {image_path}")
    print(f"{'='*60}")

    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    input_tensor = transform(img).unsqueeze(0).to(device)
    rgb_img = np.array(img.resize((224, 224))) / 255.0

    models_config = [
        ("MobileNetV3-Small", load_mobilenet, "mobilenet"),
        ("EfficientNet-Lite0", load_efficientnet, "efficientnet"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i, (display_name, loader, key) in enumerate(models_config):
        try:
            model = loader()
        except FileNotFoundError:
            print(f"ERROR: Weights not found for {display_name}. Run train.py first!")
            continue

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        predicted_class = class_names[predicted_idx.item()]

        # Grad-CAM
        target_layers = get_gradcam_target_layers(model, key)
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(predicted_idx.item())]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        axes[i + 1].imshow(visualization)
        axes[i + 1].set_title(f"{display_name}\n{predicted_class}\n{confidence.item()*100:.1f}%")
        axes[i + 1].axis('off')

        print(f"\n  {display_name}: {predicted_class} ({confidence.item()*100:.2f}%)")

    plt.suptitle("Model Comparison — Grad-CAM Heatmaps", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =======================================================
# RUN YOUR TEST HERE
# Replace "YOUR_IMAGE_PATH_HERE.JPG" with the actual path
# =======================================================
your_image_path = "YOUR_IMAGE_PATH_HERE.JPG"

# Remove the hashtag from the line below to test with a single model:
# test_image(your_image_path, model_name='mobilenet')      # MobileNetV3-Small
# test_image(your_image_path, model_name='efficientnet')   # EfficientNet-Lite0

# Remove the hashtag from the line below to compare both models:
# compare_models(your_image_path)
