import os
import json
import hashlib
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import timm

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility: Fixed Seed (Section III-A)
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────────────────────────────────────────────
# Configuration: Multiple Train-Test Split Ratios for Ensemble Training
# ──────────────────────────────────────────────────────────────────────────────
# Each model is trained on every split ratio below. The saved weights from all
# splits are later loaded together during inference and their softmax outputs
# are averaged (probability-level ensemble) for a more robust prediction.
# ──────────────────────────────────────────────────────────────────────────────

SPLIT_RATIOS = [0.80, 0.75, 0.70, 0.65, 0.60]  # train fractions

# ──────────────────────────────────────────────────────────────────────────────
# Dataset Loading (shared across all splits & models)
# ──────────────────────────────────────────────────────────────────────────────
print("Step 1: Downloading & Loading Dataset...")
path = kagglehub.dataset_download("arjuntejaswi/plant-village")
data_dir = path
for root, dirs, files in os.walk(path):
    if len(dirs) > 10:
        data_dir = root
        break

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Fix: Custom loader that forces RGBA → RGB conversion ─────────────────────
# Some images in PlantVillage are RGBA (4-channel). Without this, Normalize
# will crash because it expects exactly 3 channels.
def rgba_safe_loader(path):
    """Load image and convert to RGB, handling RGBA/palette images safely."""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


full_dataset = datasets.ImageFolder(
    root=data_dir, transform=transform, loader=rgba_safe_loader
)
class_names = full_dataset.classes
num_classes = len(class_names)

# Save class names for the Testing block to use later
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

print(f"  Dataset loaded: {len(full_dataset)} images, {num_classes} classes")


# ── Fix: Remove duplicate images by file hash ────────────────────────────────
# 14 known duplicates exist in PlantVillage. If a duplicate lands in both train
# and test, it inflates accuracy. We keep only the first occurrence of each
# unique file hash.
print("\nStep 2: Removing duplicate images by content hash...")

def compute_file_hash(filepath):
    """Return MD5 hex digest of file contents."""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

seen_hashes = {}
duplicate_indices = set()
for idx in range(len(full_dataset)):
    filepath, _ = full_dataset.samples[idx]
    file_hash = compute_file_hash(filepath)
    if file_hash in seen_hashes:
        duplicate_indices.add(idx)
    else:
        seen_hashes[file_hash] = idx

clean_indices = [i for i in range(len(full_dataset)) if i not in duplicate_indices]
print(f"  Found {len(duplicate_indices)} duplicate(s). Using {len(clean_indices)} unique images.")

# Build clean labels array for stratified splitting
clean_labels = [full_dataset.targets[i] for i in clean_indices]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10  # Balanced: good convergence + efficient on light hardware


# ──────────────────────────────────────────────────────────────────────────────
# Reusable Model Builders
# ──────────────────────────────────────────────────────────────────────────────

def build_mobilenet_v3(num_classes):
    """Build MobileNetV3-Small with custom head."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_lite(num_classes):
    """Build EfficientNet-Lite0 (via timm) with custom head."""
    model = timm.create_model('tf_efficientnet_lite0', pretrained=True, num_classes=num_classes)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Reusable Training & Evaluation Functions
# ──────────────────────────────────────────────────────────────────────────────

def train_model(model, model_name, train_loader, device, criterion, num_epochs):
    """Train a model and return it."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Epochs: {num_epochs} | Device: {device}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 50 == 0:
                print(f"  [{model_name}] Epoch {epoch+1}/{num_epochs}, Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(train_loader)
        print(f"  [{model_name}] Epoch {epoch+1}/{num_epochs} complete — Avg Loss: {avg_loss:.4f}")

    return model


def evaluate_model(model, model_name, val_loader, device):
    """Evaluate a model and return metrics dict."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics (weighted average handles class imbalances)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'all_preds': all_preds,
        'all_labels': all_labels,
    }

    print(f"\n  --- {model_name} METRICS ---")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {precision*100:.2f}%")
    print(f"  Recall    : {recall*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")

    return metrics


def plot_confusion_matrix(all_labels, all_preds, class_names, model_name):
    """Generate and display a confusion matrix."""
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('True Disease')
    plt.xlabel('Predicted Disease')
    plt.xticks(rotation=90)
    plt.tight_layout()
    fname = f"confusion_matrix_{model_name.lower().replace(' ', '_').replace('-', '_').replace(':', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {fname}")


def get_weight_filename(model_base_name, split_ratio):
    """Generate a consistent weight filename for a given model and split ratio.
    
    Examples:
        mobilenet_v3_split80.pth
        efficientnet_lite_split70.pth
    """
    split_pct = int(split_ratio * 100)
    return f"{model_base_name}_split{split_pct}.pth"


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Split Ensemble Training Loop
# ──────────────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

# Stores per-split, per-model metrics for the final summary
all_results = {}  # key: "ModelName (XX:YY)" -> metrics dict

# Track saved weight files for ensemble config
ensemble_config = {
    "split_ratios": [int(r * 100) for r in SPLIT_RATIOS],
    "models": {
        "mobilenet": {
            "display_name": "MobileNetV3-Small",
            "weight_files": [],
        },
        "efficientnet": {
            "display_name": "EfficientNet-Lite0",
            "weight_files": [],
        }
    }
}

for split_idx, train_ratio in enumerate(SPLIT_RATIOS):
    val_ratio = 1.0 - train_ratio
    split_label = f"{int(train_ratio*100)}:{int(val_ratio*100)}"
    
    print(f"\n\n{'#'*60}")
    print(f"  SPLIT {split_idx+1}/{len(SPLIT_RATIOS)} — Train:Val = {split_label}")
    print(f"{'#'*60}")

    # ── Fix: Stratified split with fixed seed ─────────────────────────────
    # StratifiedShuffleSplit preserves class proportions in both train and
    # test sets (critical for small classes like Potato_healthy = 152 images).
    # The fixed random_state ensures identical splits across runs.
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=SEED + split_idx
    )
    train_idx, val_idx = next(sss.split(clean_indices, clean_labels))

    # Map back to original dataset indices
    train_subset_indices = [clean_indices[i] for i in train_idx]
    val_subset_indices = [clean_indices[i] for i in val_idx]

    train_dataset = Subset(full_dataset, train_subset_indices)
    val_dataset = Subset(full_dataset, val_subset_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"  Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"  (Stratified, deduplicated, seed={SEED + split_idx})")

    # ── Model 1: MobileNetV3-Small ──
    model_label = f"MobileNetV3-Small ({split_label})"
    print(f"\n  Training {model_label}...")
    mobilenet = build_mobilenet_v3(num_classes)
    mobilenet = train_model(mobilenet, model_label, train_loader, device, criterion, num_epochs)

    print(f"\n  Evaluating {model_label}...")
    metrics = evaluate_model(mobilenet, model_label, val_loader, device)
    all_results[model_label] = metrics
    plot_confusion_matrix(metrics['all_preds'], metrics['all_labels'], class_names, model_label)

    weight_file = get_weight_filename("mobilenet_v3", train_ratio)
    torch.save(mobilenet.state_dict(), weight_file)
    ensemble_config["models"]["mobilenet"]["weight_files"].append(weight_file)
    print(f"  Saved weights → {weight_file}")

    # ── Model 2: EfficientNet-Lite0 ──
    model_label = f"EfficientNet-Lite0 ({split_label})"
    print(f"\n  Training {model_label}...")
    efficientnet = build_efficientnet_lite(num_classes)
    efficientnet = train_model(efficientnet, model_label, train_loader, device, criterion, num_epochs)

    print(f"\n  Evaluating {model_label}...")
    metrics = evaluate_model(efficientnet, model_label, val_loader, device)
    all_results[model_label] = metrics
    plot_confusion_matrix(metrics['all_preds'], metrics['all_labels'], class_names, model_label)

    weight_file = get_weight_filename("efficientnet_lite", train_ratio)
    torch.save(efficientnet.state_dict(), weight_file)
    ensemble_config["models"]["efficientnet"]["weight_files"].append(weight_file)
    print(f"  Saved weights → {weight_file}")


# ──────────────────────────────────────────────────────────────────────────────
# Save Ensemble Configuration
# ──────────────────────────────────────────────────────────────────────────────
# This JSON file tells test.py and server.py which weight files to load for
# the probability-averaging ensemble.

with open('ensemble_config.json', 'w') as f:
    json.dump(ensemble_config, f, indent=2)
print("\n  Ensemble config saved → ensemble_config.json")


# ──────────────────────────────────────────────────────────────────────────────
# Also save the "best split" weights as the original filenames for backward
# compatibility with existing server.py / test.py (until they are updated).
# The best split is the one with the highest F1 score for each model.
# ──────────────────────────────────────────────────────────────────────────────

for model_arch, base_name, legacy_name in [
    ("MobileNetV3-Small", "mobilenet_v3", "mobilenet_v3.pth"),
    ("EfficientNet-Lite0", "efficientnet_lite", "efficientnet_lite.pth"),
]:
    best_f1 = -1
    best_split_file = None
    for split_ratio in SPLIT_RATIOS:
        split_label = f"{int(split_ratio*100)}:{int((1-split_ratio)*100)}"
        key = f"{model_arch} ({split_label})"
        if key in all_results and all_results[key]['f1'] > best_f1:
            best_f1 = all_results[key]['f1']
            best_split_file = get_weight_filename(base_name, split_ratio)

    if best_split_file:
        import shutil
        shutil.copy2(best_split_file, legacy_name)
        print(f"  Best {model_arch} weights ({best_split_file}) → copied to {legacy_name}")


# ──────────────────────────────────────────────────────────────────────────────
# Grand Comparison Summary — All Splits × Both Models
# ──────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 90)
print("  MULTI-SPLIT ENSEMBLE TRAINING — COMPLETE RESULTS")
print("=" * 90)

# Header
header = f"  {'Split':<12} {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
print(header)
print(f"  {'-' * (len(header) - 2)}")

for split_ratio in SPLIT_RATIOS:
    split_label = f"{int(split_ratio*100)}:{int((1-split_ratio)*100)}"
    for model_arch in ["MobileNetV3-Small", "EfficientNet-Lite0"]:
        key = f"{model_arch} ({split_label})"
        if key in all_results:
            m = all_results[key]
            print(f"  {split_label:<12} {model_arch:<22} {m['accuracy']*100:>9.2f}% {m['precision']*100:>9.2f}% {m['recall']*100:>9.2f}% {m['f1']*100:>9.2f}%")
    print(f"  {'-' * (len(header) - 2)}")

print("=" * 90)
print(f"\n  Total models trained: {len(SPLIT_RATIOS) * 2} ({len(SPLIT_RATIOS)} splits × 2 architectures)")
print(f"  Ensemble weight files saved for each architecture.")
print(f"  At inference time, all {len(SPLIT_RATIOS)} split-models per architecture are loaded")
print(f"  and their softmax probabilities are averaged for a stronger, ensemble prediction.")
print("\n  Training Phase Complete! Ensemble is ready.\n")
