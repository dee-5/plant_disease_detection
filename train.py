import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import timm

# ──────────────────────────────────────────────────────────────────────────────
# Dataset Loading (shared across both models)
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

full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = full_dataset.classes
num_classes = len(class_names)

# Save class names for the Testing block to use later
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 1  # Set to 10 or 20 for real results!


# ──────────────────────────────────────────────────────────────────────────────
# Reusable Training & Evaluation Functions
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
    plt.savefig(f"confusion_matrix_{model_name.lower().replace(' ', '_').replace('-', '_')}.png", dpi=150)
    print(f"  Confusion matrix saved to confusion_matrix_{model_name.lower().replace(' ', '_').replace('-', '_')}.png")


# ──────────────────────────────────────────────────────────────────────────────
# Train Both Models
# ──────────────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
results = {}

# --- Model 1: MobileNetV3-Small ---
print("\n\nStep 2a: Training MobileNetV3-Small...")
mobilenet = build_mobilenet_v3(num_classes)
mobilenet = train_model(mobilenet, "MobileNetV3-Small", train_loader, device, criterion, num_epochs)

print("\nStep 3a: Evaluating MobileNetV3-Small...")
results['MobileNetV3-Small'] = evaluate_model(mobilenet, "MobileNetV3-Small", val_loader, device)
plot_confusion_matrix(results['MobileNetV3-Small']['all_preds'], results['MobileNetV3-Small']['all_labels'], class_names, "MobileNetV3-Small")

torch.save(mobilenet.state_dict(), "mobilenet_v3.pth")
print("  Model saved to 'mobilenet_v3.pth'")

# --- Model 2: EfficientNet-Lite0 ---
print("\n\nStep 2b: Training EfficientNet-Lite0...")
efficientnet = build_efficientnet_lite(num_classes)
efficientnet = train_model(efficientnet, "EfficientNet-Lite0", train_loader, device, criterion, num_epochs)

print("\nStep 3b: Evaluating EfficientNet-Lite0...")
results['EfficientNet-Lite0'] = evaluate_model(efficientnet, "EfficientNet-Lite0", val_loader, device)
plot_confusion_matrix(results['EfficientNet-Lite0']['all_preds'], results['EfficientNet-Lite0']['all_labels'], class_names, "EfficientNet-Lite0")

torch.save(efficientnet.state_dict(), "efficientnet_lite.pth")
print("  Model saved to 'efficientnet_lite.pth'")


# ──────────────────────────────────────────────────────────────────────────────
# Comparison Summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n\n" + "="*60)
print("  MODEL COMPARISON SUMMARY")
print("="*60)
print(f"  {'Metric':<15} {'MobileNetV3-Small':>20} {'EfficientNet-Lite0':>20}")
print(f"  {'-'*55}")
for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
    m_val = results['MobileNetV3-Small'][metric_name] * 100
    e_val = results['EfficientNet-Lite0'][metric_name] * 100
    m_arrow = " *" if m_val > e_val else ""
    e_arrow = " *" if e_val > m_val else ""
    print(f"  {metric_name.capitalize():<15} {m_val:>18.2f}%{m_arrow:>1} {e_val:>18.2f}%{e_arrow:>1}")
print("="*60)
print("\nTraining Phase Complete! Both models are ready.")
