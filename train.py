import os
import json
import copy
import time
import hashlib
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, classification_report)
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import timm

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
# Reviewer asked for 60:40, 70:30, 80:20. 75:25 and 65:35 are kept as extras.
# ── Quick-test: set to [0.80] with 1 epoch. Full run: all 5 with 10 epochs. ──
SPLIT_RATIOS = [0.80, 0.75, 0.70, 0.65, 0.60]  # Full run: all 5 splits
NUM_EPOCHS = 10                                   # Full run: 10 epochs
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Issue #1 fix: Explicitly define the expected 15 classes
# ──────────────────────────────────────────────────────────────────────────────
EXPECTED_CLASSES = sorted([
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
])

# ──────────────────────────────────────────────────────────────────────────────
# Issue #3 fix: Per-architecture preprocessing constants
# ──────────────────────────────────────────────────────────────────────────────
# MobileNetV3-Small uses standard ImageNet normalization.
# EfficientNet-Lite0 (via timm) uses mean=0.5, std=0.5 (from model.default_cfg).
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EFFNET_MEAN, EFFNET_STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def make_train_transform(mean, std):
    """Training transform: augmentation (random crop + horizontal flip)."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def make_eval_transform(mean, std):
    """Evaluation transform: NO augmentation (same as deployment inference)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
print("Step 1: Downloading & Loading Dataset...")
path = kagglehub.dataset_download("arjuntejaswi/plant-village")
data_dir = path
for root, dirs, files in os.walk(path):
    if len(dirs) > 10:
        data_dir = root
        break


def rgba_safe_loader(p):
    """Load image and convert to RGB, handling RGBA/palette images safely."""
    with open(p, 'rb') as f:
        return Image.open(f).convert('RGB')


# Use a neutral (no-norm) transform just for loading — we apply the actual
# architecture-specific transform via a wrapper later.
_identity_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load with ImageNet normalization (MobileNet). EfficientNet transform
# is built per-split when creating DataLoaders via TransformedSubset.
mobilenet_train_tf = make_train_transform(IMAGENET_MEAN, IMAGENET_STD)
mobilenet_eval_tf = make_eval_transform(IMAGENET_MEAN, IMAGENET_STD)
effnet_train_tf = make_train_transform(EFFNET_MEAN, EFFNET_STD)
effnet_eval_tf = make_eval_transform(EFFNET_MEAN, EFFNET_STD)

# Master dataset — we use the eval transform as default; train/eval-specific
# transforms are applied via TransformedSubset below.
master_dataset = datasets.ImageFolder(root=data_dir, transform=None, loader=rgba_safe_loader)
class_names = master_dataset.classes
num_classes = len(class_names)

# ── Issue #1 guard: assert exactly the expected 15 classes ────────────────────
assert sorted(class_names) == EXPECTED_CLASSES, (
    f"Class mismatch! Expected {len(EXPECTED_CLASSES)} classes:\n"
    f"  {EXPECTED_CLASSES}\n"
    f"Got {len(class_names)} classes:\n"
    f"  {sorted(class_names)}"
)

with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print(f"  Dataset loaded: {len(master_dataset)} images, {num_classes} classes")
print(f"  Classes verified: matches expected 15-class set ✓")


# ── TransformedSubset: apply a specific transform to a Subset ─────────────────
class TransformedSubset(torch.utils.data.Dataset):
    """Wraps a list of indices into the master dataset, applying a given transform."""
    def __init__(self, master, indices, transform):
        self.master = master
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, label = self.master.samples[real_idx]
        img = self.master.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Remove exact duplicates (before any split) ───────────────────────────────
print("\nStep 2: Removing duplicate images by content hash...")


def compute_file_hash(fp):
    h = hashlib.md5()
    with open(fp, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


seen, dup = {}, set()
for idx, (fp, _) in enumerate(master_dataset.samples):
    hsh = compute_file_hash(fp)
    if hsh in seen:
        dup.add(idx)
    else:
        seen[hsh] = idx
clean_indices = [i for i in range(len(master_dataset)) if i not in dup]
clean_labels = [master_dataset.targets[i] for i in clean_indices]
print(f"  Found {len(dup)} duplicate(s). Using {len(clean_indices)} unique images.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────
def build_mobilenet_v3(n):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n)
    return m


def build_efficientnet_lite(n):
    return timm.create_model('tf_efficientnet_lite0', pretrained=True, num_classes=n)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)


@torch.no_grad()
def measure_cpu_latency(model, warmup=10, runs=100):
    """Model-only forward-pass latency on CPU (batch=1). Does NOT include image
    loading, resizing, normalization, or Grad-CAM — see Issue #5."""
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, 3, 224, 224)
    for _ in range(warmup):
        m(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        m(x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


# ──────────────────────────────────────────────────────────────────────────────
# Train / evaluate
# ──────────────────────────────────────────────────────────────────────────────
def train_model(model, name, loader, criterion, epochs):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    print(f"\n{'=' * 60}\n  Training: {name} | Epochs: {epochs} | Device: {device}\n{'=' * 60}")
    for ep in range(epochs):
        model.train()
        run = 0.0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item()
            if (i + 1) % 50 == 0:
                print(f"  [{name}] Epoch {ep + 1}/{epochs}, Batch [{i + 1}/{len(loader)}], Loss: {loss.item():.4f}")
        print(f"  [{name}] Epoch {ep + 1}/{epochs} complete - Avg Loss: {run / len(loader):.4f}")
    return model


@torch.no_grad()
def evaluate_model(model, name, loader):
    model.eval()
    preds, labels, probs = [], [], []
    for x, y in loader:
        out = model(x.to(device))
        probs.append(torch.softmax(out, 1).cpu().numpy())
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    preds, labels, probs = np.array(preds), np.array(labels), np.concatenate(probs)

    m = dict(
        accuracy=accuracy_score(labels, preds),
        precision=precision_score(labels, preds, average='weighted', zero_division=0),
        recall=recall_score(labels, preds, average='weighted', zero_division=0),
        f1=f1_score(labels, preds, average='weighted', zero_division=0),
        macro_f1=f1_score(labels, preds, average='macro', zero_division=0),
        auroc=roc_auc_score(labels, probs, multi_class='ovr', average='macro',
                            labels=list(range(num_classes))),
        all_preds=preds, all_labels=labels, probs=probs,
    )
    print(f"\n  --- {name} ---")
    print(f"  Acc {m['accuracy'] * 100:.2f}% | Prec {m['precision'] * 100:.2f}% | "
          f"Rec {m['recall'] * 100:.2f}% | F1 {m['f1'] * 100:.2f}% | "
          f"MacroF1 {m['macro_f1'] * 100:.2f}% | AUROC {m['auroc']:.4f}")
    return m


def save_outputs(m, tag):
    """Confusion matrix (labels FIRST, predictions second — fixes original bug),
    per-class report, raw predictions."""
    cm = confusion_matrix(m['all_labels'], m['all_preds'], labels=list(range(num_classes)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(f"{RESULTS_DIR}/cm_{tag}.csv")
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 6})
    plt.title(f'Confusion Matrix - {tag}')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/cm_{tag}.png", dpi=300)
    plt.close()

    rep = classification_report(m['all_labels'], m['all_preds'], labels=list(range(num_classes)),
                                target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(f"{RESULTS_DIR}/per_class_{tag}.csv")
    # Keep probabilities so the consensus / Grad-CAM analysis can be done later without retraining.
    np.savez_compressed(f"{RESULTS_DIR}/preds_{tag}.npz", labels=m['all_labels'],
                        preds=m['all_preds'], probs=m['probs'])


def wfile(base, ratio):
    return f"{base}_split{int(ratio * 100)}.pth"


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
summary_rows = []
ensemble_config = {"split_ratios": [int(r * 100) for r in SPLIT_RATIOS],
                   "models": {"mobilenet": {"display_name": "MobileNetV3-Small", "weight_files": []},
                              "efficientnet": {"display_name": "EfficientNet-Lite0", "weight_files": []}}}

for split_idx, ratio in enumerate(SPLIT_RATIOS):
    val_ratio = 1.0 - ratio
    label = f"{int(ratio * 100)}:{int(round(val_ratio * 100))}"
    split_seed = SEED + split_idx
    print(f"\n\n{'#' * 60}\n  SPLIT {split_idx + 1}/{len(SPLIT_RATIOS)} - Train:Test = {label} (seed {split_seed})\n{'#' * 60}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=split_seed)
    tr, va = next(sss.split(clean_indices, clean_labels))
    tr_idx = [clean_indices[i] for i in tr]
    va_idx = [clean_indices[i] for i in va]
    json.dump({"train": tr_idx, "test": va_idx}, open(f"{RESULTS_DIR}/split_{int(ratio * 100)}.json", "w"))

    print(f"  Train: {len(tr_idx)} | Test: {len(va_idx)}")

    # Issue #4 note: Each split ratio uses a different seed AND different train/test
    # partition. Results are reported per-ratio individually — they are NOT five
    # replications of one fixed split protocol.

    for arch, builder, base, key, train_tf, eval_tf in [
        ("MobileNetV3-Small", build_mobilenet_v3, "mobilenet_v3", "mobilenet",
         mobilenet_train_tf, mobilenet_eval_tf),
        ("EfficientNet-Lite0", build_efficientnet_lite, "efficientnet_lite", "efficientnet",
         effnet_train_tf, effnet_eval_tf),
    ]:
        name = f"{arch} ({label})"

        # Issue #3 fix: each architecture gets its own transform
        g = torch.Generator().manual_seed(split_seed)
        train_ds = TransformedSubset(master_dataset, tr_idx, train_tf)
        val_ds = TransformedSubset(master_dataset, va_idx, eval_tf)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, generator=g)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

        model = train_model(builder(num_classes), name, train_loader, criterion, NUM_EPOCHS)
        m = evaluate_model(model, name, val_loader)
        tag = f"{base}_{int(ratio * 100)}"
        save_outputs(m, tag)

        # Issue #5 note: this is model-only forward pass latency, not end-to-end
        lat_mean, lat_std = measure_cpu_latency(model)
        torch.save(model.state_dict(), wfile(base, ratio))
        ensemble_config["models"][key]["weight_files"].append(wfile(base, ratio))
        summary_rows.append(dict(
            split=label, model=arch, train_n=len(tr_idx), test_n=len(va_idx),
            accuracy=m['accuracy'] * 100, precision=m['precision'] * 100, recall=m['recall'] * 100,
            f1=m['f1'] * 100, macro_f1=m['macro_f1'] * 100, auroc=m['auroc'],
            params_M=count_params(model) / 1e6, size_MB_fp32=model_size_mb(model),
            cpu_latency_ms_mean=lat_mean, cpu_latency_ms_std=lat_std,
            torch_threads=torch.get_num_threads(),
            note_latency="model-only forward pass, excludes I/O and preprocessing",
        ))
        pd.DataFrame(summary_rows).to_csv(f"{RESULTS_DIR}/metrics_summary.csv", index=False)

with open('ensemble_config.json', 'w') as f:
    json.dump(ensemble_config, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Issue #2: Dual-model consensus evaluation
# ──────────────────────────────────────────────────────────────────────────────
# For each split, compare MobileNetV3-Small vs EfficientNet-Lite0 predictions.
# High = both agree AND both confident (>0.8 max prob)
# Medium = both agree, lower confidence
# Low = they disagree
print("\n\nStep 4: Computing dual-model consensus (MobileNetV3 vs EfficientNet)...")
consensus_rows = []
for ratio in SPLIT_RATIOS:
    split_pct = int(ratio * 100)
    mob_path = f"{RESULTS_DIR}/preds_mobilenet_v3_{split_pct}.npz"
    eff_path = f"{RESULTS_DIR}/preds_efficientnet_lite_{split_pct}.npz"
    if not (os.path.exists(mob_path) and os.path.exists(eff_path)):
        print(f"  [SKIP] Missing predictions for split {split_pct}")
        continue

    mob = np.load(mob_path)
    eff = np.load(eff_path)
    # Both were evaluated on the same test set (same indices)
    assert np.array_equal(mob['labels'], eff['labels']), "Label mismatch between architectures!"

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

    # Accuracy within each consensus tier
    high_acc = accuracy_score(labels[high], mob_preds[high]) * 100 if high.sum() > 0 else float('nan')
    med_acc = accuracy_score(labels[medium], mob_preds[medium]) * 100 if medium.sum() > 0 else float('nan')
    low_n = low.sum()

    consensus_rows.append(dict(
        split=f"{split_pct}:{100 - split_pct}",
        total=n_total,
        high_n=int(high.sum()), high_pct=high.sum() / n_total * 100, high_acc=high_acc,
        medium_n=int(medium.sum()), medium_pct=medium.sum() / n_total * 100, medium_acc=med_acc,
        low_n=int(low_n), low_pct=low_n / n_total * 100,
    ))
    print(f"  Split {split_pct}:{100 - split_pct}: "
          f"High={high.sum()} ({high.sum() / n_total * 100:.1f}%, acc={high_acc:.1f}%), "
          f"Medium={medium.sum()} ({medium.sum() / n_total * 100:.1f}%, acc={med_acc:.1f}%), "
          f"Low={low_n} ({low_n / n_total * 100:.1f}%)")

if consensus_rows:
    pd.DataFrame(consensus_rows).to_csv(f"{RESULTS_DIR}/consensus_summary.csv", index=False)
    print(f"  Consensus table saved → {RESULTS_DIR}/consensus_summary.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Deployment convenience: best split by test F1. NOT an unbiased result.
# ──────────────────────────────────────────────────────────────────────────────
import shutil
for arch, base, legacy in [("MobileNetV3-Small", "mobilenet_v3", "mobilenet_v3.pth"),
                           ("EfficientNet-Lite0", "efficientnet_lite", "efficientnet_lite.pth")]:
    rows = [r for r in summary_rows if r['model'] == arch]
    best = max(rows, key=lambda r: r['f1'])
    shutil.copy2(wfile(base, int(best['split'].split(':')[0]) / 100), legacy)

print("\n" + "=" * 100)
print(pd.DataFrame(summary_rows).drop(columns=['note_latency']).round(3).to_string(index=False))
print("=" * 100)
print("All tables, confusion matrices (csv+png), per-class reports and predictions are in ./results/")
