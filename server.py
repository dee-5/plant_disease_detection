"""
PlantXAI — Flask Server (Ensemble Edition)
A lightweight, mobile-friendly plant disease detection server with:
- Live camera capture, photo capture, and image upload
- Multi-split ensemble inference: averages predictions from models trained on
  different train-test splits (80:20, 75:25, 70:30, 65:35, 60:40) for both
  MobileNetV3-Small AND EfficientNet-Lite0
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
# Model Loading — Ensemble Support
# ──────────────────────────────────────────────────────────────────────────────
print("\n[MODEL] Loading PlantXAI models (Ensemble Mode)...")

try:
    with open(os.path.join(BASE_DIR, 'class_names.json'), 'r') as f:
        class_names = json.load(f)
except FileNotFoundError:
    print("ERROR: class_names.json not found! Run train.py first.")
    exit()

device = torch.device("cpu")  # Edge-optimized: CPU only

# Load ensemble configuration
ensemble_config = None
try:
    with open(os.path.join(BASE_DIR, 'ensemble_config.json'), 'r') as f:
        ensemble_config = json.load(f)
    print(f"[ENSEMBLE] Config loaded — Splits: {ensemble_config['split_ratios']}")
except FileNotFoundError:
    print("[INFO] ensemble_config.json not found. Falling back to single-model mode.")


def _build_mobilenet():
    """Build MobileNetV3-Small architecture (no weights)."""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(class_names))
    return model


def _build_efficientnet():
    """Build EfficientNet-Lite0 architecture (no weights)."""
    return timm.create_model('tf_efficientnet_lite0', pretrained=False, num_classes=len(class_names))


def load_model_weights(weight_path, model_type):
    """Load a single model from a weight file."""
    if model_type == 'mobilenet':
        model = _build_mobilenet()
    else:
        model = _build_efficientnet()

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.eval().to(device)
    return model


def load_ensemble_models(model_type):
    """Load all split-trained models for the given architecture.
    Returns list of loaded model objects.
    Falls back to legacy single weight file if no ensemble config.
    """
    loaded = []

    if ensemble_config:
        config = ensemble_config['models'].get(model_type)
        if config:
            for weight_file in config['weight_files']:
                full_path = os.path.join(BASE_DIR, weight_file)
                if os.path.exists(full_path):
                    model = load_model_weights(full_path, model_type)
                    loaded.append(model)
                    print(f"  [ENSEMBLE] Loaded {weight_file}")
                else:
                    print(f"  [WARNING] Weight file not found: {weight_file}")

    # Fallback: use legacy single weight file
    if not loaded:
        legacy = "mobilenet_v3.pth" if model_type == 'mobilenet' else "efficientnet_lite.pth"
        legacy_path = os.path.join(BASE_DIR, legacy)
        if os.path.exists(legacy_path):
            model = load_model_weights(legacy_path, model_type)
            loaded.append(model)
            print(f"  [FALLBACK] Loaded single model: {legacy}")

    return loaded


# Load all ensembles at startup
mobilenet_ensemble = load_ensemble_models('mobilenet')
efficientnet_ensemble = load_ensemble_models('efficientnet')

if not mobilenet_ensemble and not efficientnet_ensemble:
    print("ERROR: No model weights found! Run train.py first.")
    exit()

print(f"[MODEL] MobileNetV3 ensemble: {len(mobilenet_ensemble)} model(s)")
print(f"[MODEL] EfficientNet ensemble: {len(efficientnet_ensemble)} model(s)")

# Issue #3 fix: Per-architecture preprocessing
# MobileNetV3-Small uses standard ImageNet normalization.
# EfficientNet-Lite0 (via timm tf_efficientnet_lite0) uses mean=0.5, std=0.5.
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EFFNET_MEAN, EFFNET_STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

mobilenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

efficientnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=EFFNET_MEAN, std=EFFNET_STD),
])


# ──────────────────────────────────────────────────────────────────────────────
# Model Registry (Ensemble-aware)
# ──────────────────────────────────────────────────────────────────────────────

MODELS = {}
if mobilenet_ensemble:
    MODELS['mobilenet'] = {
        'ensemble': mobilenet_ensemble,
        'display_name': f'MobileNetV3-Small (Ensemble ×{len(mobilenet_ensemble)})',
        # GradCAM target layers from the first model in the ensemble
        'target_layers': [mobilenet_ensemble[0].features[-1]],
        'gradcam_model': mobilenet_ensemble[0],
        'transform': mobilenet_transform,
    }
if efficientnet_ensemble:
    MODELS['efficientnet'] = {
        'ensemble': efficientnet_ensemble,
        'display_name': f'EfficientNet-Lite0 (Ensemble ×{len(efficientnet_ensemble)})',
        'target_layers': [efficientnet_ensemble[0].conv_head],
        'gradcam_model': efficientnet_ensemble[0],
        'transform': efficientnet_transform,
    }


def get_available_models():
    """Return list of available model keys."""
    return list(MODELS.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble Prediction Engine
# ──────────────────────────────────────────────────────────────────────────────

def ensemble_predict(input_tensor, models_list):
    """Average softmax probabilities across all models in the ensemble."""
    all_probs = []

    with torch.no_grad():
        for model in models_list:
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            all_probs.append(probs)

    stacked = torch.stack(all_probs, dim=0)
    avg_probabilities = torch.mean(stacked, dim=0)

    confidence, predicted_idx = torch.max(avg_probabilities, 0)
    return avg_probabilities, predicted_idx, confidence


def run_prediction(pil_image, model_key='mobilenet'):
    """
    Run full ensemble prediction pipeline: inference + GradCAM + explanation.
    Returns dict with all results.
    """
    if model_key not in MODELS:
        # Fallback to first available model
        model_key = get_available_models()[0]

    model_info = MODELS[model_key]
    models_list = model_info['ensemble']
    display_name = model_info['display_name']
    target_layers = model_info['target_layers']
    gradcam_model = model_info['gradcam_model']
    model_transform = model_info['transform']

    img = pil_image.convert('RGB')
    input_tensor = model_transform(img).unsqueeze(0).to(device)

    # Ensemble Forward Pass
    avg_probabilities, predicted_idx, confidence = ensemble_predict(input_tensor, models_list)

    predicted_class = class_names[predicted_idx.item()]
    conf_value = float(confidence.item())

    # Top 3
    top_prob, top_indices = torch.topk(avg_probabilities, min(3, len(class_names)))
    top3 = [
        {"class": class_names[idx.item()], "confidence": round(float(prob.item()) * 100, 1)}
        for prob, idx in zip(top_prob, top_indices)
    ]

    # GradCAM (use the first model in the ensemble for visualization)
    cam = GradCAM(model=gradcam_model, target_layers=target_layers)
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
        "ensemble_size": len(models_list),
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
    """Return list of available models with ensemble info."""
    models_list = []
    for key, info in MODELS.items():
        models_list.append({
            "key": key,
            "display_name": info['display_name'],
            "ensemble_size": len(info['ensemble']),
        })
    return jsonify({"models": models_list})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Accept image via file upload or base64 and return ensemble prediction."""
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
    """Run both model ensembles on the same image and return side-by-side results."""
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

    port = int(os.environ.get('PORT', 5001))
    local_ip = get_local_ip()
    url = f"http://{local_ip}:{port}"

    loaded_models = ", ".join([MODELS[k]['display_name'] for k in MODELS])
    total_models = sum(len(MODELS[k]['ensemble']) for k in MODELS)

    print("\n" + "=" * 60)
    print("  PlantXAI -- Plant Disease Detection Server (Ensemble)")
    print("=" * 60)
    print(f"\n  Models:  {loaded_models}")
    print(f"  Total ensemble models loaded: {total_models}")
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
