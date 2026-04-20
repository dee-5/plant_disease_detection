"""
PlantXAI — Flask Server
A lightweight, mobile-friendly plant disease detection server with:
- Live camera capture, photo capture, and image upload
- Dual-model inference: MobileNetV3-Small AND EfficientNet-Lite0
- GradCAM heatmap generation for both architectures
- Intelligent per-image GradCAM explanations
- Side-by-side model comparison mode
- SQLite scan history database
- QR code for phone access
"""

import os
import json
import uuid
import base64
import socket
from io import BytesIO
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
import timm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import database as db
from disease_info import generate_gradcam_explanation, DISEASE_INFO

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
GRADCAM_DIR = os.path.join(BASE_DIR, 'static', 'gradcam')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────
print("\n[MODEL] Loading PlantXAI models...")

try:
    with open(os.path.join(BASE_DIR, 'class_names.json'), 'r') as f:
        class_names = json.load(f)
except FileNotFoundError:
    print("ERROR: class_names.json not found! Run train.py first.")
    exit()

device = torch.device("cpu")  # Edge-optimized: CPU only

# --- Model 1: MobileNetV3-Small ---
mobilenet_model = None
try:
    mobilenet_model = models.mobilenet_v3_small(weights=None)
    in_features = mobilenet_model.classifier[3].in_features
    mobilenet_model.classifier[3] = nn.Linear(in_features, len(class_names))
    mobilenet_model.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "mobilenet_v3.pth"), map_location=device
    ))
    mobilenet_model = mobilenet_model.eval().to(device)
    print(f"[MODEL] MobileNetV3-Small loaded — {len(class_names)} classes on {device}")
except FileNotFoundError:
    print("[MODEL] WARNING: mobilenet_v3.pth not found. MobileNetV3 will be unavailable.")
    mobilenet_model = None

# --- Model 2: EfficientNet-Lite0 ---
efficientnet_model = None
try:
    efficientnet_model = timm.create_model(
        'tf_efficientnet_lite0', pretrained=False, num_classes=len(class_names)
    )
    efficientnet_model.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "efficientnet_lite.pth"), map_location=device
    ))
    efficientnet_model = efficientnet_model.eval().to(device)
    print(f"[MODEL] EfficientNet-Lite0 loaded — {len(class_names)} classes on {device}")
except FileNotFoundError:
    print("[MODEL] WARNING: efficientnet_lite.pth not found. EfficientNet-Lite will be unavailable.")
    efficientnet_model = None

# Check at least one model is available
if mobilenet_model is None and efficientnet_model is None:
    print("ERROR: No model weights found! Run train.py first.")
    exit()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────────────────────────────────────

MODELS = {}
if mobilenet_model:
    MODELS['mobilenet'] = {
        'model': mobilenet_model,
        'display_name': 'MobileNetV3-Small',
        'target_layers': [mobilenet_model.features[-1]],
    }
if efficientnet_model:
    MODELS['efficientnet'] = {
        'model': efficientnet_model,
        'display_name': 'EfficientNet-Lite0',
        'target_layers': [efficientnet_model.conv_head],
    }


def get_available_models():
    """Return list of available model keys."""
    return list(MODELS.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Prediction Engine
# ──────────────────────────────────────────────────────────────────────────────

def run_prediction(pil_image, model_key='mobilenet'):
    """
    Run full prediction pipeline: inference + GradCAM + explanation.
    Returns dict with all results.
    """
    if model_key not in MODELS:
        # Fallback to first available model
        model_key = get_available_models()[0]

    model_info = MODELS[model_key]
    model = model_info['model']
    display_name = model_info['display_name']
    target_layers = model_info['target_layers']

    img = pil_image.convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Forward Pass
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = class_names[predicted_idx.item()]
    conf_value = float(confidence.item())

    # Top 3
    top_prob, top_indices = torch.topk(probabilities, min(3, len(class_names)))
    top3 = [
        {"class": class_names[idx.item()], "confidence": round(float(prob.item()) * 100, 1)}
        for prob, idx in zip(top_prob, top_indices)
    ]

    # GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_idx.item())]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    rgb_img = np.array(img.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    heatmap_pil = Image.fromarray(cam_image)

    # Intelligent Explanation
    explanation = generate_gradcam_explanation(
        grayscale_cam, predicted_class, conf_value, top3
    )

    # Save images
    file_id = str(uuid.uuid4())[:8]
    img_filename = f"{file_id}_original.jpg"
    cam_filename = f"{file_id}_gradcam_{model_key}.jpg"

    img.resize((224, 224)).save(os.path.join(UPLOAD_DIR, img_filename), quality=90)
    heatmap_pil.save(os.path.join(GRADCAM_DIR, cam_filename), quality=90)

    # Save to database
    explanation_text = (
        explanation['summary'] + "\n\n" +
        explanation['spatial_detail'] + "\n\n" +
        explanation.get('disease_detail', '') + "\n\n" +
        explanation['recommendation']
    )

    scan_id = db.save_scan(
        image_filename=img_filename,
        gradcam_filename=cam_filename,
        predicted_class=predicted_class,
        confidence=conf_value,
        top3=top3,
        explanation=explanation_text,
        model_name=model_key
    )

    return {
        "scan_id": scan_id,
        "model_key": model_key,
        "model_name": display_name,
        "predicted_class": predicted_class,
        "display_name": explanation['display_name'],
        "confidence": round(conf_value * 100, 1),
        "top3": top3,
        "is_healthy": explanation['is_healthy'],
        "severity": explanation['severity'],
        "image_url": f"/static/uploads/{img_filename}",
        "gradcam_url": f"/static/gradcam/{cam_filename}",
        "explanation": {
            "summary": explanation['summary'],
            "spatial_detail": explanation['spatial_detail'],
            "disease_detail": explanation['disease_detail'],
            "recommendation": explanation['recommendation'],
            "trust_assessment": explanation['trust_assessment'],
            "stats": explanation['stats'],
        }
    }


def _extract_image_from_request():
    """Extract a PIL image from the incoming Flask request (file or base64)."""
    pil_image = None

    # Check for file upload
    if 'image' in request.files:
        file = request.files['image']
        if file.filename:
            pil_image = Image.open(file.stream)

    # Check for base64
    elif request.is_json:
        data = request.get_json()
        if 'image_base64' in data:
            b64 = data['image_base64']
            if ',' in b64:
                b64 = b64.split(',')[1]
            img_bytes = base64.b64decode(b64)
            pil_image = Image.open(BytesIO(img_bytes))

    return pil_image


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    available = get_available_models()
    return render_template('index.html', available_models=available)


@app.route('/history')
def history():
    scans = db.get_all_scans()
    stats = db.get_stats()
    # Parse top3_json for each scan
    for scan in scans:
        if scan.get('top3_json'):
            scan['top3'] = json.loads(scan['top3_json'])
    return render_template('history.html', scans=scans, stats=stats)


@app.route('/api/models')
def api_models():
    """Return list of available models."""
    models_list = []
    for key, info in MODELS.items():
        models_list.append({
            "key": key,
            "display_name": info['display_name'],
        })
    return jsonify({"models": models_list})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Accept image via file upload or base64 and return prediction."""
    try:
        pil_image = _extract_image_from_request()

        if pil_image is None:
            return jsonify({"error": "No image provided"}), 400

        # Get model selection from query param or JSON body
        model_key = request.args.get('model', 'mobilenet')
        if request.is_json:
            data = request.get_json()
            model_key = data.get('model', model_key)

        result = run_prediction(pil_image, model_key=model_key)
        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """Run both models on the same image and return side-by-side results."""
    try:
        pil_image = _extract_image_from_request()

        if pil_image is None:
            return jsonify({"error": "No image provided"}), 400

        results = {}
        for model_key in get_available_models():
            results[model_key] = run_prediction(pil_image, model_key=model_key)

        return jsonify({
            "comparison": True,
            "results": results,
            "models_compared": list(results.keys()),
        })

    except Exception as e:
        print(f"[ERROR] Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/history')
def api_history():
    """Get scan history as JSON."""
    scans = db.get_all_scans()
    stats = db.get_stats()
    for scan in scans:
        if scan.get('top3_json'):
            scan['top3'] = json.loads(scan['top3_json'])
    return jsonify({"scans": scans, "stats": stats})


@app.route('/api/scan/<int:scan_id>')
def api_scan_detail(scan_id):
    """Get a single scan detail."""
    scan = db.get_scan(scan_id)
    if not scan:
        return jsonify({"error": "Scan not found"}), 404
    if scan.get('top3_json'):
        scan['top3'] = json.loads(scan['top3_json'])
    return jsonify(scan)


@app.route('/api/scan/<int:scan_id>', methods=['DELETE'])
def api_delete_scan(scan_id):
    """Delete a scan."""
    scan = db.get_scan(scan_id)
    if scan:
        # Delete image files
        img_path = os.path.join(UPLOAD_DIR, scan['image_filename'])
        cam_path = os.path.join(GRADCAM_DIR, scan['gradcam_filename'])
        if os.path.exists(img_path):
            os.remove(img_path)
        if os.path.exists(cam_path):
            os.remove(cam_path)
        db.delete_scan(scan_id)
    return jsonify({"success": True})


# ──────────────────────────────────────────────────────────────────────────────
# Server Launch
# ──────────────────────────────────────────────────────────────────────────────

def get_local_ip():
    """Get the local network IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == '__main__':
    import sys
    import io
    # Fix Windows terminal encoding for unicode
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    port = 5000
    local_ip = get_local_ip()
    url = f"http://{local_ip}:{port}"

    loaded_models = ", ".join([MODELS[k]['display_name'] for k in MODELS])

    print("\n" + "=" * 60)
    print("  PlantXAI -- Plant Disease Detection Server")
    print("=" * 60)
    print(f"\n  Models:  {loaded_models}")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: {url}")
    print(f"\n  Open the Network URL on your phone")
    print(f"  (Make sure phone is on the same WiFi)\n")

    # Generate QR code for easy phone access
    try:
        import qrcode
        qr = qrcode.QRCode(version=1, box_size=1, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        print("  Scan this QR code with your phone camera:")
        print("  " + "-" * 40)
        qr.print_ascii(invert=True)
        print("  " + "-" * 40)
    except ImportError:
        print("  (Install 'qrcode' package for QR code display)")

    print(f"\n  Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=port, debug=False)
