# 🌾 Rice Field State Identification Neural Network

A PyTorch deep learning system for satellite rice field state identification that classifies fields into **4 distinct target states**:

1. **🏜️ Dry**: Harvested, dry soil, or maturing/pale fields.
2. **💧 Flooded**: Water-filled paddies or freshly prepared wet fields.
3. **🌿 Planted**: Actively growing, lush green rice crops.
4. **🌳 Others**: Non-field areas, trees, bunds/walkways, shadows, and obstacles.

---

## 📁 Dataset Organization

The model trains directly on the dataset inside `Dataset/`:
```
Dataset/
├── Dry/       # Images of dry / harvested rice fields
├── Flood/     # Images of flooded / water-filled paddies
├── Planted/   # Images of actively planted green rice crops
└── Others/    # Non-field regions, tree mounds, and obstacles
```

---

## 🚀 Quick Start & Usage

### 1. Training the Model

To train the 4-class neural network on `Dataset/`:

```bash
python train.py
```

Outputs:
- Saves checkpoint to `rice_field_classifier.pth`.
- Displays training loss, validation loss, validation accuracy, and per-class performance.

---

### 2. Python Inference API

```python
from inference import predict_field_state, predict_input_directory
from src.inference import RiceFieldPredictor

# Single image classification:
result = predict_field_state("Dataset/Planted/planted_01.png")
print("Status:", result["status"])         # 'Dry', 'Flooded', 'Planted', or 'Others'
print("Confidence:", f"{result['confidence'] * 100:.1f}%")
print("Probabilities:", result["probabilities"])

# Batch classify all images in Input/ (including subdirectories):
predictor = RiceFieldPredictor()
batch = predictor.predict_directory("Input", recursive=True, save_csv="results.csv")
print("Total Classified:", batch["total_images"])
print("Summary:", batch["summary_counts"])
```

---

### 3. Command Line Interface (CLI)

```bash
# 1. Batch classify ALL pictures inside Input/ (and all subdirectories):
python inference.py

# 2. Classify pictures in a custom folder (recursively):
python inference.py --dir Input/my_subfolder

# 3. Classify and export results to CSV and JSON:
python inference.py --dir Input --csv results.csv --json results.json

# 4. Classify a single image:
python inference.py --image Dataset/Flood/flood_01.png
```

---

### 4. Interactive Graphical User Interface (GUI)

Launch the interactive desktop application:

```bash
python gui_app.py
```

**GUI Features:**
- ⚡ **Classify 'Input/' Folder**: Recursively scans all images across `Input/` and all nested subdirectories with one click.
- 📋 **Batch Results Viewer**: Interactive table showing relative paths, classifications, and confidence; double-click any row to view it.
- ⚡ **Quick Sample Buttons**: Instantly switch between Dry, Flooded, Planted, and Others samples.
- 📊 **Real-Time Probability Bars**: Live probability distribution across all 4 classes.
- 💾 **Export Results**: Save CSV/JSON reports or inspected images.

---

## 🧪 Automated Testing

Run the test suite:

```bash
python test_pipeline.py
```
