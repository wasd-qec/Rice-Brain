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
# Automatically creates an Output/ folder with a .json report for each subdirectory!
predictor = RiceFieldPredictor()
batch = predictor.predict_directory("Input", recursive=True, output_dir="Output")
print("Total Classified:", batch["total_images"])
print("Summary:", batch["summary_counts"])
print("Generated JSON reports:", batch["generated_json_files"])
```

---

### 3. Command Line Interface (CLI)

```bash
# 1. Batch classify ALL pictures inside Input/ (and all subdirectories):
# Generates Output/root.json, Output/<subdir>.json, and Output/all_results.json
python inference.py

# 2. Classify pictures in a custom folder and save JSONs to a custom output folder:
python inference.py --dir Input/my_subfolder --output_dir Output

# 3. Classify and export an additional CSV table:
python inference.py --dir Input --csv results.csv

# 4. Classify a single image:
python inference.py --image Dataset/Flood/flood_01.png
```

---

### 4. Interactive Graphical User Interface (GUI) & Operator Review

Launch the interactive desktop application:

```bash
python gui_app.py
```

**GUI & Operator Review Features:**
- ⚡ **Classify 'Input/' Folder**: Recursively scans all images across `Input/` and all nested subdirectories with one click.
- 📁 **Per-Subdirectory JSON Generation**: Automatically saves separate JSON reports into `Output/` for each subdirectory.
- ⚠️ **Low-Confidence Flagging**: Automatically flags any prediction with confidence $< 80\%$ so operators can inspect questionable fields first.
- ✏️ **1-Click Operator Overruling**:
  - Hotkeys **`[1]` Dry**, **`[2]` Flooded**, **`[3]` Planted**, **`[4]` Others**, or **`[Space]`** to overrule or reset any AI decision.
  - Overruled rows are highlighted as `[OVERRULED]`.
- 💾 **Save Reviewed Reports**: Updates `Output/` JSON reports with human review audit data (`is_overruled`, `operator_label`, `reviewed_at`).
- 📥 **Add Overruled to Dataset (Active Learning)**: 1-click copies corrected images into `Dataset/<Label>/` so running `python train.py` enables continuous learning from human feedback.
- 📂 **Open Output Folder Button**: Quick 1-click button to open `Output/` in Windows File Explorer.
- ⚡ **Quick Sample Buttons**: Instantly switch between Dry, Flooded, Planted, and Others samples.
- 📊 **Real-Time Probability Bars**: Live probability distribution across all 4 classes.

---

## 🧪 Automated Testing

Run the test suite (7 unit tests):

```bash

python test_pipeline.py
```
