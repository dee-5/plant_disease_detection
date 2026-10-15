import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import gradio as gr
import qrcode
import time
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. Initialization and Loading
try:
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
except FileNotFoundError:
    print("ERROR: class_names.json not found! Run train.py first.")
    exit()

device = torch.device("cpu") # For mobile web server, CPU is fine and prevents driver blocks

# --- Load MobileNetV3-Small ---
mobilenet_model = None
try:
    mobilenet_model = models.mobilenet_v3_small(weights=None) 
    in_features = mobilenet_model.classifier[3].in_features
    mobilenet_model.classifier[3] = nn.Linear(in_features, len(class_names))
    mobilenet_model.load_state_dict(torch.load("mobilenet_v3.pth", map_location=device))
    mobilenet_model = mobilenet_model.eval().to(device)
    print("[MODEL] MobileNetV3-Small loaded successfully")
except FileNotFoundError:
    print("[MODEL] WARNING: mobilenet_v3.pth not found. MobileNetV3 unavailable.")

# --- Load EfficientNet-Lite0 ---
efficientnet_model = None
try:
    efficientnet_model = timm.create_model(
        'tf_efficientnet_lite0', pretrained=False, num_classes=len(class_names)
    )
    efficientnet_model.load_state_dict(torch.load("efficientnet_lite.pth", map_location=device))
    efficientnet_model = efficientnet_model.eval().to(device)
    print("[MODEL] EfficientNet-Lite0 loaded successfully")
except FileNotFoundError:
    print("[MODEL] WARNING: efficientnet_lite.pth not found. EfficientNet-Lite unavailable.")

if mobilenet_model is None and efficientnet_model is None:
    print("ERROR: No model weights found! Run train.py first.")
    exit()

# Model registry
MODELS = {}
if mobilenet_model:
    MODELS['MobileNetV3-Small'] = {
        'model': mobilenet_model,
        'target_layers': [mobilenet_model.features[-1]],
        'transform': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }
if efficientnet_model:
    MODELS['EfficientNet-Lite0'] = {
        'model': efficientnet_model,
        'target_layers': [efficientnet_model.conv_head],
        'transform': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]),
    }

model_choices = list(MODELS.keys())

# 2. Prediction Core
def predict_leaf(img, model_choice):
    if img is None:
        return "Please upload an image.", None
    
    if model_choice not in MODELS:
        model_choice = model_choices[0]

    model_info = MODELS[model_choice]
    model = model_info['model']
    target_layers = model_info['target_layers']
    model_transform = model_info['transform']
    
    # Gradio sends a PIL Image
    img = img.convert('RGB')
    input_tensor = model_transform(img).unsqueeze(0).to(device)
    
    # Forward Pass
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_class = class_names[predicted_idx.item()]
    
    # Gather top 3 confidences for a pretty chart
    top_prob, top_indices = torch.topk(probabilities, 3)
    confidences = {class_names[idx.item()]: float(prob.item()) for prob, idx in zip(top_prob, top_indices)}
    
    # Grad-CAM Heatmap
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_idx.item())]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    rgb_img = np.array(img.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Convert back to PIL for Gradio
    heatmap_pil = Image.fromarray(cam_image)
    
    return confidences, heatmap_pil

# 3. User Interface (Gradio)
custom_css = """
.container { max-width: 900px; margin: auto; }
.output-markdown { text-align: center; font-size: 1.2rem; }
.grad-cam-box { border: 2px solid #ff4b4b; border-radius: 10px; padding: 5px; }
"""

with gr.Blocks(title="PlantXAI") as demo:
    gr.Markdown("# 🌿 Plant Disease XAI Detector")
    gr.Markdown("Take a photo of a sick leaf, and our lightweight Edge-AI will instantly diagnose it and show you *exactly* where the disease is located using a heat map!")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Leaf Photo or Use Camera")
            model_selector = gr.Radio(
                choices=model_choices,
                value=model_choices[0],
                label="🧠 Select Model Architecture",
                info="Compare different AI architectures on the same leaf image"
            )
            analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")
            
        with gr.Column(scale=1):
            output_confidences = gr.Label(label="AI Diagnosis Results")
            output_heatmap = gr.Image(type="pil", label="Explainable AI Heatmap (Red = Disease Areas)", elem_classes="grad-cam-box")

    analyze_btn.click(fn=predict_leaf, inputs=[input_image, model_selector], outputs=[output_confidences, output_heatmap])
    
    gr.Markdown("---")
    gr.Markdown("**How to read the map:** If the highlighted red zones accurately align with visible spots or mold on your physical leaf, you can trust this diagnosis.")

# 4. Boot-up Server and QR Generator
print("\nStarting App Server...")
# Launch with share=True to bypass Firewalls. Prevent thread lock so we can print the QR.
app_data = demo.launch(share=True, prevent_thread_lock=True, quiet=True, css=custom_css, theme=gr.themes.Soft())

public_url = app_data[1] if len(app_data) > 1 else app_data[0] # The Public Gradio Share URL or local URL

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

if public_url:
    print("\n" + "="*60)
    print("  SCAN THIS QR CODE WITH YOUR PHONE'S CAMERA ")
    print("="*60 + "\n")
    
    qr = qrcode.QRCode(version=1, box_size=1, border=2)
    qr.add_data(public_url)
    qr.make(fit=True)
    
    # Save the QR code as an image file so it's easy to view
    img = qr.make_image(fill_color="black", back_color="white")
    img.save("app_qrcode.png")
    
    # We use invert=True because most terminals are dark background, and QR codes need white boundaries
    qr.print_ascii(invert=True)
    
    print("\n" + "="*60)
    print(f"URL: {public_url}")
    print(f"QR Code image saved as: app_qrcode.png")
    print("Keep this terminal running! Press Ctrl+C to shut down.")
    print("="*60 + "\n")
else:
    print("\nFailed to generate a public URL. Try checking your internet connection.")

# Keep the script running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down server...")
