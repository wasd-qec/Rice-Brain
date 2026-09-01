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
from inference import predict_field_state

result = predict_field_state("Dataset/Planted/1.png")

print("Status:", result["status"])         # 'Dry', 'Flooded', 'Planted', or 'Others'
print("Confidence:", f"{result['confidence'] * 100:.1f}%")
print("Probabilities:", result["probabilities"])
```

Or query a specific coordinate on a satellite map:
```python
from src.inference import RiceFieldPredictor

predictor = RiceFieldPredictor()
res = predictor.predict_coordinate(
    full_image_path="Dataset/Planted/1.png",
    coordinate=(500, 400),
    output_annotated_path="output_annotated.png"
)
print("State at (500, 400):", res["status"])
```

---

### 3. Command Line Interface (CLI)

```bash
# Classify an image/crop:
python inference.py --image Dataset/Flood/1.png

# Query a specific (x, y) coordinate on a map:
python inference.py --image Dataset/Planted/1.png --x 500 --y 400 --output result.png
```

---

### 4. Interactive Graphical User Interface (GUI)

Launch the interactive desktop application:

```bash
python gui_app.py
```

**GUI Features:**
- 🎯 **Point-and-Click**: Click anywhere on the satellite image to analyze the rice field at that exact location.
- ⚡ **Quick Sample Buttons**: Instantly switch between Dry, Flooded, Planted, and Others samples.
- 📊 **Real-Time Probability Bars**: Live probability distribution across all 4 classes.
- 💾 **Export Results**: Save annotated visual outputs.

---

## 🧪 Automated Testing

Run the test suite:

```bash
python test_pipeline.py
```
