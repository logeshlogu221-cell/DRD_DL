# RetinaAI — Diabetic Retinopathy Detection Web App

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your model file
Copy your trained model file into this folder and rename it:
```
drcnnrb_cbam_best__2_.pth  →  model.pth
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

## Project Structure
```
dr_webapp/
├── app.py              ← Flask backend + model inference
├── model.pth           ← Trained DRCNNRB + CBAM weights
├── requirements.txt
├── templates/
│   └── index.html      ← Full frontend UI
└── uploads/            ← Temporary upload folder (auto-created)
```

## Features
- Drag & drop or click to upload retinal fundus image
- DRCNNRB + CBAM inference with TTA×5
- Probability bar chart for all 5 classes
- Clinical severity advice per prediction
- Original vs preprocessed image comparison
- 98.29% accuracy model
