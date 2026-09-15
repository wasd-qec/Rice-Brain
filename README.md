# 🌾 Rice Field AI — Neural Network, Parcel Database & Web Curation System

A production-grade PyTorch deep learning and GIS monitoring system for rice field state classification, interactive dataset curation, and parcel database management.

Classifies agricultural parcels into **4 distinct states**:
1. **🏜️ Dry**: Harvested, bare soil, or dry field conditions.
2. **💧 Flooded**: Water-filled paddies or wet soil prepared for planting.
3. **🌿 Planted**: Actively growing, lush green rice crops.
4. **🌳 Others**: Roads, houses, forest, tree mounds, canals, and obstacles.

---

## 🏗️ System Architecture & Components

```
NNetwork/
├── web/                       # Modern glassmorphic Web Dashboard (HTML5, Vanilla CSS, JS)
│   ├── index.html             # Dashboard UI with warning alerts & review modal
│   ├── styles.css             # Tailored dark theme, badges, and responsive layout
│   └── app.js                 # REST client, dynamic table rendering, keyboard shortcuts
├── web_server.py              # REST API server & lazy SQLite parcel database manager
├── farm_parcels.db            # SQLite database (lazily created upon first classification)
├── parcel_pictures/           # Dedicated directory storing retained parcel photos on disk
│
├── gui_app.py                 # Desktop Tkinter GUI: Dataset Curator & Training Manager
├── embed_coordinates.py       # Utility to embed/read standard EXIF GPS tags (Cambodia coordinates)
│
├── train.py                   # Root runner to train the PyTorch neural network
├── inference.py               # Root runner for CLI inference & JSON/CSV batch reporting
│
├── Dataset/                   # 4-class training dataset (JPEG images)
│   ├── Dry/                   # Dry field photos
│   ├── Flood/                 # Flooded paddy photos
│   ├── Planted/               # Green planted rice photos
│   └── Others/                # Non-field photos
├── Unsorted/                  # Raw incoming images waiting to be curated/labeled
├── Input/                     # Directory scanned by inference and the Web Dashboard
│
├── docs/                      # Documentation guides & cheat sheets
│   ├── Note.txt               # Day-to-day command cheat sheet
│   ├── Note to change.txt     # Feature checklist & operator notes
│   └── CODE_WALKTHROUGH_GUIDE.txt # Step-by-step code roadmap
│
├── src/                       # Core neural network & pipeline source code
│   ├── model.py               # Custom ResNet architecture (ResidualBlocks + Stem + Classifier)
│   ├── dataset.py             # PyTorch Dataset loader with online data augmentations
│   ├── train.py               # GPU-accelerated training engine with loss & accuracy tracking
│   └── inference.py           # Inference predictor, GPS extraction, and batch reporting
│
├── tests/                     # Automated test suites
│   ├── test_pipeline.py       # Core unit test suite (7 automated tests)
│   └── test_parcel_db.py      # Database & curation rule unit tests (3 automated tests)
```

---

## 🚀 Quick Start

### 0. Install Dependencies

Choose based on your hardware:

**Option A: Default Installation (CPU-Only, Lightweight ~200 MB)**
```bash
pip install -r requirements.txt
```
*(Automatically uses the official PyTorch CPU wheel configured inside `requirements.txt`)*

**Option B: NVIDIA GPU (CUDA Accelerated Training)**
Uncomment your matching GPU line inside `requirements.txt`, or install directly via pip:
- **Python 3.14 + RTX 50 Series (Blackwell - RTX 5060/5070/5080/5090 - CUDA 13.0):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
  pip install -r requirements.txt
  ```
- **Python 3.11 - 3.12 + RTX 50 / 40 / 30 Series (CUDA 12.4+):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  pip install -r requirements.txt
  ```
- **RTX 40 Series / 30 Series (CUDA 12.1):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt
  ```
- **Older GPUs / Legacy Drivers (CUDA 11.8):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  pip install -r requirements.txt
  ```

**Verify PyTorch & Hardware Detection:**
```bash
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```


### 1. Web Dashboard & Parcel Database (`web_server.py`)
Launch the web dashboard:
```bash
python web_server.py
```
Open your browser to: **`http://localhost:5000`**

**Key Curation & Database Rules:**
- **Lazy Database Initialization**: The SQLite database file `farm_parcels.db` is **not** created until you click **"Classify Input Directory"** on the website for the first time.
- **Coordinate as Primary Key**: Each parcel is indexed by GPS coordinates (`coordinate TEXT PRIMARY KEY`). Classifying new data with the same coordinate automatically updates the existing row without duplication.
- **Automatic Picture Retention / Discard Rule**:
  - If a parcel is **`Planted`** OR has **low confidence ($<80\%$)**: `flag = True`, and the JPEG image is saved to the `parcel_pictures/` directory while storing its relative file path in the database (`picture_path`).
  - Otherwise: `flag = False`, and `picture_path = NULL` (the picture is not stored to conserve disk and database space).
- **Warning Alert**: If any parcel is flagged, a prominent warning banner is displayed.
- **Operator Review Modal**:
  - Clicking **"Inspect & Review"** loads the picture from `parcel_pictures/` storage via the API.
  - The operator can confirm or re-assign the category (`Dry`, `Flooded`, `Planted`, `Others`).
  - **If confirmed as `Planted`**: Picture file and review flag remain preserved.
  - **If changed to `Dry`, `Flooded`, or `Others`**: Review flag is cleared and the picture file is permanently deleted from disk storage.

---

### 2. Desktop Dataset Curator & Training App (`gui_app.py`)
Launch the interactive desktop curation application:
```bash
python gui_app.py
```

**Key Features:**
- **Dual Dataset Pools**:
  - **`📥 Unsorted Pool`** (`Unsorted/`): Inspect and label incoming raw photos.
  - **`📁 Sorted Dataset`** (`Dataset/`): Inspect categorized classes (`Dry`, `Flood`, `Planted`, `Others`).
- **Rapid Keyboard Sorting & Auto-Advance**:
  - Press **`[1]` Dry**, **`[2]` Flooded**, **`[3]` Planted**, **`[4]` Others** to move the file instantly into the target folder.
  - Selection automatically advances to the next image.
- **Mis-sort Correction & Undo**:
  - Select any already-sorted image and press `[1-4]` to change its class folder on disk.
  - Press **`[U]`** to return the image to the `Unsorted/` pool.
  - Press **`[Space]`** to accept the AI suggestion.
- **Vertical Tips & Controls**: Quick-reference shortcuts card directly beside the image preview.
- **One-Click Model Retraining**: Click **`🚀 Train AI Model`** to trigger background retraining without freezing the GUI.

---

### 3. Model Training (`train.py`)
Train the neural network using GPU acceleration (CUDA):
```bash
python train.py
```
- Trains for 12 epochs on `Dataset/` using Adam optimizer and Cross-Entropy Loss.
- Saves the best checkpoint to `rice_field_classifier.pth`.

---

### 4. CLI Batch Inference (`inference.py`)
Classify images and view results or export CSV:
```bash
# 1. Classify all images in Input/ (including subdirectories):
python inference.py

# 2. Classify a custom directory:
python inference.py --dir Input/my_fields

# 3. Export CSV table with GPS coordinates:
python inference.py --dir Input --csv results.csv

# 4. Classify a single image:
python inference.py --image Input/image.jpg
```

---

### 5. GPS Coordinate Utility (`embed_coordinates.py`)
Embed realistic Cambodia farm coordinates (Battambang, Takeo, Prey Veng, Siem Reap, Pursat, Kampong Cham, Banteay Meanchey, Kandal) into image EXIF metadata:
```bash
# Embed into an entire directory:
python embed_coordinates.py Input

# Embed into a single image:
python embed_coordinates.py Input/image.jpg
```

---

## 🧪 Testing & Validation

Run the automated test suites:
```bash
# Run all tests via test discovery:
python -m unittest discover -s tests

# Or run individual test suites:
python -m unittest tests/test_pipeline.py
python -m unittest tests/test_parcel_db.py
```
- All tests execute headlessly and verify model forward pass, dataset augmentation, continuous learning export, lazy DB creation, primary key upserts, and picture retention rules.
