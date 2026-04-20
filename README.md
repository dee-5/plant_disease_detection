# PlantXAI: Edge-Optimized Plant Disease Detection with Explainable AI 🌿 🔍

A robust, full-stack machine learning pipeline that goes beyond standard classification. PlantXAI detects 15 different plant diseases, explains its reasoning using Grad-CAM heatmaps, and runs a fully functional mobile-friendly web server optimized for resource-constrained edge devices (CPU-only).

## 🌟 Key Features

1. **Dual-Model Edge Architecture**: Uses both **MobileNetV3-Small** (2.5M parameters) and **EfficientNet-Lite0**, enabling extremely fast, real-time inference strictly on a CPU without requiring cloud GPUs. You can run this natively on a Raspberry Pi or standard laptop.
2. **True Explainable AI (XAI)**: Unlike traditional implementations that just display a Grad-CAM heatmap, PlantXAI features an **Intelligent Explanation Engine**. It mathematically analyzes the heatmap's spatial focus, spread, and intensity, cross-referencing it with botanical rules to generate a human-readable explanation of *why* it made its diagnosis.
3. **Mobile-First Full-Stack App**: Comes with a built-in Flask server featuring a stunning dark "glassmorphism" user interface.
    - Three Input Modes: **Live Camera**, **Take Photo**, and **File Upload**.
    - **Side-by-Side Model Comparison**: Compare inferences and Grad-CAM heatmaps from both MobileNetV3 and EfficientNet-Lite0 simultaneously.
    - Auto-Analyze: The inference engine triggers instantly upon image selection.
4. **Scan History & Persistence**: Integrated SQLite database to track historical scans, calculate crop health ratios, and monitor disease trends over time.

---

## 🚀 Quick Start (Running the App Locally)

Want to run the app on your phone natively? The server will automatically generate a QR code!

### 1. Installation
Install the required packages.
```bash
pip install -r requirements.txt
```

### 2. Launch the Server
```bash
python server.py
```
*Look at your terminal! You will see a large QR Code. Simply point your smartphone camera at your screen to launch the web app instantly on your phone.* Ensure your phone and computer are on the same Wi-Fi network.

---

## 🧠 Training & Modifying the Model (Google Colab)

If you wish to retrain the underlying model or test the Python scripts independently of the web app, simply upload the `train.py` and `test.py` files to Google Colab. The codebase is fully modular.

### `train.py`
Downloads the massive PlantVillage dataset (54,000+ images) via KaggleHub, handles bounding/normalization, executes the training loop, evaluates metrics (Accuracy, Precision, Recall, F1), generates a Seaborn Confusion Matrix, and saves the trained `.pth` weights files.

### `test.py`
A lightweight testing script intended to be used in Colab alongside `train.py`. It explicitly loads the saved model weights and outputs the custom intelligent Grad-CAM explanations directly in the console.

## 📊 Dataset
This project automatically targets the **PlantVillage** dataset, capable of detecting and explaining diseases spanning crops like Potato, Tomato, and Bell Peppers.
