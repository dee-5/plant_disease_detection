import json
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
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

# ──────────────────────────────────────────────────────────────────────────────
# Issue #3 fix: Per-architecture preprocessing
# ──────────────────────────────────────────────────────────────────────────────
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

TRANSFORMS = {
    'mobilenet': mobilenet_transform,
    'efficientnet': efficientnet_transform,
}


# ──────────────────────────────────────────────────────────────────────────────
# 2. Load Ensemble Configuration
# ──────────────────────────────────────────────────────────────────────────────

ensemble_config = None
try:
    with open('ensemble_config.json', 'r') as f:
        ensemble_config = json.load(f)
    print(f"[ENSEMBLE] Loaded config — Splits: {ensemble_config['split_ratios']}")
except FileNotFoundError:
    print("[INFO] ensemble_config.json not found. Falling back to single-model mode.")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Model Building & Loading Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_mobilenet():
    """Build MobileNetV3-Small architecture (no weights)."""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(class_names))
    return model


def _build_efficientnet():
    """Build EfficientNet-Lite0 architecture (no weights)."""
    return timm.create_model('tf_efficientnet_lite0', pretrained=False, num_classes=len(class_names))


def load_single_model(weight_path, model_type):
    """Load a single model from a weight file.
    model_type: 'mobilenet' or 'efficientnet'
    """
    if model_type == 'mobilenet':
        model = _build_mobilenet()
    else:
        model = _build_efficientnet()

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device).eval()
    return model


def load_ensemble(model_type):
    """Load all split-trained models for the given architecture.
    Returns a list of (model, weight_file_name) tuples.
    
    Falls back to the single legacy weight file if no ensemble config exists.
    """
    models_list = []

    if ensemble_config:
        config = ensemble_config['models'].get(model_type)
        if config:
            for weight_file in config['weight_files']:
                if os.path.exists(weight_file):
                    model = load_single_model(weight_file, model_type)
                    models_list.append((model, weight_file))
                    print(f"  [ENSEMBLE] Loaded {weight_file}")
                else:
                    print(f"  [WARNING] Weight file not found: {weight_file}")

    # Fallback: use legacy single weight file
    if not models_list:
        legacy_file = "mobilenet_v3.pth" if model_type == 'mobilenet' else "efficientnet_lite.pth"
        if os.path.exists(legacy_file):
            model = load_single_model(legacy_file, model_type)
            models_list.append((model, legacy_file))
            print(f"  [FALLBACK] Loaded single model: {legacy_file}")
        else:
            print(f"  [ERROR] No weights found for {model_type}!")

    return models_list


def ensemble_predict(input_tensor, models_list):
    """Run ensemble prediction by averaging softmax probabilities across all models.
    
    Returns:
        avg_probabilities: Averaged probability tensor across all split-models
        predicted_idx: Index of the predicted class
        confidence: Confidence of the prediction
    """
    all_probs = []
    
    with torch.no_grad():
        for model, _ in models_list:
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            all_probs.append(probs)

    # Stack and average across all split-models
    stacked = torch.stack(all_probs, dim=0)  # shape: [num_models, num_classes]
    avg_probabilities = torch.mean(stacked, dim=0)  # shape: [num_classes]

    confidence, predicted_idx = torch.max(avg_probabilities, 0)
    return avg_probabilities, predicted_idx, confidence


def get_gradcam_target_layers(model, model_type):
    """Return the appropriate GradCAM target layer for each architecture."""
    if model_type == 'efficientnet':
        return [model.conv_head]
    else:  # mobilenet
        return [model.features[-1]]


# ──────────────────────────────────────────────────────────────────────────────
# 4. Test Functions
# ──────────────────────────────────────────────────────────────────────────────

def test_image(image_path, model_name='mobilenet'):
    """
    Test a single image with the ensemble of models for the specified architecture.
    model_name: 'mobilenet' or 'efficientnet'
    """
    print(f"\nAnalyzing custom image: {image_path}")
    print(f"Using model: {model_name} (ensemble mode)")

    # Load the ensemble
    models_list = load_ensemble(model_name)
    if not models_list:
        print(f"ERROR: No weights found for {model_name}. Run train.py first!")
        return

    print(f"  Ensemble size: {len(models_list)} model(s)")

    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Issue #3 fix: use the correct transform for this architecture
    tf = TRANSFORMS[model_name]
    input_tensor = tf(img).unsqueeze(0).to(device)

    # Ensemble Predict
    avg_probabilities, predicted_idx, confidence = ensemble_predict(input_tensor, models_list)

    predicted_class = class_names[predicted_idx.item()]
    display_model = "MobileNetV3-Small" if model_name == 'mobilenet' else "EfficientNet-Lite0"

    print(f"==========================================")
    print(f"--> Model     : {display_model} (Ensemble of {len(models_list)} splits)")
    print(f"--> Prediction: {predicted_class}")
    print(f"--> Confidence: {confidence.item()*100:.2f}%")
    print(f"==========================================")

    # Show individual model predictions for transparency
    print(f"\n[📊 PER-SPLIT BREAKDOWN]")
    with torch.no_grad():
        for model, weight_file in models_list:
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, idx = torch.max(probs, 0)
            split_pred = class_names[idx.item()]
            print(f"  {weight_file:<35} → {split_pred} ({conf.item()*100:.1f}%)")

    print(f"\n[🔍 XAI DIAGNOSTIC EXPLANATION]")
    print("The image on the right is a Grad-CAM 'Heatmap'.")
    print("It explains exactly *why* the AI made this diagnosis by highlighting the pixels it relied on.")
    print(" - RED / YELLOW areas: The AI focused heavily here (e.g., at the exact spot of a leaf lesion or mold).")
    print(" - BLUE / COOL areas: The AI ignored these areas (e.g., the healthy leaf parts or background).")
    print("If the RED areas directly align with the physical disease symptoms on the leaf, you can highly trust this diagnosis!\n")

    # Explainable AI: Grad-CAM (use the best-performing / first model for visualization)
    best_model = models_list[0][0]
    target_layers = get_gradcam_target_layers(best_model, model_name)
    cam = GradCAM(model=best_model, target_layers=target_layers)
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
    axes[1].set_title(f"Grad-CAM Heatmap ({display_model} Ensemble)\nDiagnosis: {predicted_class}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def compare_models(image_path):
    """Run both model ensembles on the same image and display side-by-side comparison.
    
    Also computes and prints the dual-model consensus verdict:
      - HIGH: Both architectures agree AND both have >80% confidence
      - MEDIUM: Both architectures agree but lower confidence
      - LOW: The two architectures disagree
    """
    print(f"\n{'='*60}")
    print(f"  COMPARING BOTH ENSEMBLES ON: {image_path}")
    print(f"{'='*60}")

    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    rgb_img = np.array(img.resize((224, 224))) / 255.0

    models_config = [
        ("MobileNetV3-Small", "mobilenet"),
        ("EfficientNet-Lite0", "efficientnet"),
    ]

    predictions = {}  # store per-architecture results for consensus

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i, (display_name, model_type) in enumerate(models_config):
        models_list = load_ensemble(model_type)
        if not models_list:
            print(f"ERROR: Weights not found for {display_name}. Run train.py first!")
            continue

        # Issue #3 fix: use the correct transform for this architecture
        tf = TRANSFORMS[model_type]
        input_tensor = tf(img).unsqueeze(0).to(device)

        # Ensemble prediction
        avg_probabilities, predicted_idx, confidence = ensemble_predict(input_tensor, models_list)
        predicted_class = class_names[predicted_idx.item()]

        predictions[model_type] = {
            'class': predicted_class,
            'idx': predicted_idx.item(),
            'confidence': confidence.item(),
        }

        # Grad-CAM (use first model in ensemble)
        best_model = models_list[0][0]
        target_layers = get_gradcam_target_layers(best_model, model_type)
        cam = GradCAM(model=best_model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(predicted_idx.item())]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        axes[i + 1].imshow(visualization)
        axes[i + 1].set_title(f"{display_name}\n(Ensemble of {len(models_list)})\n{predicted_class}\n{confidence.item()*100:.1f}%")
        axes[i + 1].axis('off')

        print(f"\n  {display_name} Ensemble ({len(models_list)} splits): {predicted_class} ({confidence.item()*100:.2f}%)")

    # ── Issue #2: Dual-model consensus verdict ────────────────────────────
    if 'mobilenet' in predictions and 'efficientnet' in predictions:
        mob = predictions['mobilenet']
        eff = predictions['efficientnet']
        agree = mob['class'] == eff['class']
        both_confident = mob['confidence'] > 0.8 and eff['confidence'] > 0.8

        if agree and both_confident:
            verdict = "HIGH"
            color = "🟢"
            explanation = "Both architectures agree with high confidence (>80%)."
        elif agree:
            verdict = "MEDIUM"
            color = "🟡"
            explanation = "Both architectures agree, but at least one has lower confidence."
        else:
            verdict = "LOW"
            color = "🔴"
            explanation = f"Architectures DISAGREE: MobileNet→{mob['class']}, EfficientNet→{eff['class']}."

        print(f"\n  {color} CONSENSUS VERDICT: {verdict}")
        print(f"     {explanation}")
        print(f"     MobileNet: {mob['class']} ({mob['confidence']*100:.1f}%)")
        print(f"     EfficientNet: {eff['class']} ({eff['confidence']*100:.1f}%)")

    plt.suptitle("Ensemble Model Comparison — Grad-CAM Heatmaps", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Batch Consensus Evaluation (from saved predictions)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_consensus(results_dir="results"):
    """Compute dual-model consensus (High/Medium/Low) from saved prediction files.
    
    This loads the .npz files saved by train.py and computes consensus statistics
    for each split ratio. Prints a summary table and saves to CSV.
    
    High: both architectures agree AND both >80% confident
    Medium: both agree, lower confidence
    Low: they disagree
    """
    from sklearn.metrics import accuracy_score

    print(f"\n{'='*70}")
    print(f"  DUAL-MODEL CONSENSUS EVALUATION")
    print(f"{'='*70}")

    consensus_rows = []

    # Try to load ensemble config for split ratios, else scan results dir
    ratios = []
    if ensemble_config:
        ratios = ensemble_config['split_ratios']
    else:
        import re
        for f in os.listdir(results_dir):
            m = re.match(r'preds_mobilenet_v3_(\d+)\.npz', f)
            if m:
                ratios.append(int(m.group(1)))
        ratios.sort(reverse=True)

    if not ratios:
        print("  No prediction files found. Run train.py first!")
        return

    for split_pct in ratios:
        mob_path = os.path.join(results_dir, f"preds_mobilenet_v3_{split_pct}.npz")
        eff_path = os.path.join(results_dir, f"preds_efficientnet_lite_{split_pct}.npz")

        if not (os.path.exists(mob_path) and os.path.exists(eff_path)):
            print(f"  [SKIP] Missing predictions for split {split_pct}")
            continue

        mob = np.load(mob_path)
        eff = np.load(eff_path)

        # Verify same test set
        if not np.array_equal(mob['labels'], eff['labels']):
            print(f"  [ERROR] Label mismatch for split {split_pct}!")
            continue

        mob_preds, eff_preds = mob['preds'], eff['preds']
        mob_probs, eff_probs = mob['probs'], eff['probs']
        mob_conf = mob_probs.max(axis=1)
        eff_conf = eff_probs.max(axis=1)
        labels = mob['labels']

        n_total = len(labels)
        agree = mob_preds == eff_preds
        high = agree & (mob_conf > 0.8) & (eff_conf > 0.8)
        medium = agree & ~high
        low = ~agree

        # Accuracy within each tier (using MobileNet prediction where they agree,
        # since by definition both predict the same class in High/Medium tiers)
        high_acc = accuracy_score(labels[high], mob_preds[high]) * 100 if high.sum() > 0 else float('nan')
        med_acc = accuracy_score(labels[medium], mob_preds[medium]) * 100 if medium.sum() > 0 else float('nan')
        low_n = int(low.sum())

        row = dict(
            split=f"{split_pct}:{100 - split_pct}",
            total=n_total,
            high_n=int(high.sum()), high_pct=round(high.sum() / n_total * 100, 1),
            high_accuracy=round(high_acc, 2),
            medium_n=int(medium.sum()), medium_pct=round(medium.sum() / n_total * 100, 1),
            medium_accuracy=round(med_acc, 2),
            low_n=low_n, low_pct=round(low_n / n_total * 100, 1),
        )
        consensus_rows.append(row)

        print(f"\n  Split {row['split']} ({n_total} test samples):")
        print(f"    🟢 HIGH   : {row['high_n']:>5} ({row['high_pct']:>5.1f}%) — accuracy {row['high_accuracy']:.1f}%")
        print(f"    🟡 MEDIUM : {row['medium_n']:>5} ({row['medium_pct']:>5.1f}%) — accuracy {row['medium_accuracy']:.1f}%")
        print(f"    🔴 LOW    : {row['low_n']:>5} ({row['low_pct']:>5.1f}%)")

    if consensus_rows:
        df = pd.DataFrame(consensus_rows)
        out_path = os.path.join(results_dir, "consensus_summary.csv")
        df.to_csv(out_path, index=False)
        print(f"\n  Consensus table saved → {out_path}")
        print(f"\n{df.to_string(index=False)}")

    return consensus_rows


# =======================================================
# RUN YOUR TEST HERE
# Replace "YOUR_IMAGE_PATH_HERE.JPG" with the actual path
# =======================================================
your_image_path = "YOUR_IMAGE_PATH_HERE.JPG"

# Remove the hashtag from the line below to test with a single model ensemble:
# test_image(your_image_path, model_name='mobilenet')      # MobileNetV3-Small Ensemble
# test_image(your_image_path, model_name='efficientnet')   # EfficientNet-Lite0 Ensemble

# Remove the hashtag from the line below to compare both model ensembles:
# compare_models(your_image_path)

# Remove the hashtag below to compute consensus statistics from saved predictions:
# evaluate_consensus()
