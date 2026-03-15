# NeuroVision AI (Brain Tumor Segmentation)

A full-stack prototype for **brain tumor segmentation** using a U-Net model trained on the **BraTS 2020** dataset, paired with:

- **FastAPI backend** for MRI upload + processing + report generation
- **React dashboard** for doctors to manage patients and review segmented scans
- **AI report generation** using OpenAI (via LangChain)

---

## Repository Structure

- `backend/` – FastAPI server and inference pipeline
- `src/` – Model training, preprocessing, and inference utilities
- `doctor-dashboard/` – React + Vite UI for doctors
- `model/` – Trained weights (e.g., `best_model.weights.h5`)
- `BraTS2020_TrainingData/` – BraTS training dataset (expected layout)
- `BraTS2020_ValidationData/` – BraTS validation dataset (expected layout)
- `services/` – AI report generation + chat assistant services
- `xai_seg/` – Explainability / Grad-CAM / SHAP tooling (optional)

---

## Prerequisites

### System
- Windows (this repo is configured with Windows-style paths by default)
- Python 3.11+ (recommended)
- Node.js 18+ (for the React dashboard)

### Python Dependencies

There is no single top-level `requirements.txt`, but the code relies on these main packages:

- `fastapi`, `uvicorn`
- `tensorflow` / `keras`
- `nibabel`, `opencv-python`, `matplotlib`
- `supabase` (Supabase client)
- `langchain_openai` (report generation / agent chat)

You can install the dependencies manually, e.g.:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn tensorflow nibabel opencv-python matplotlib supabase langchain_openai pydantic fpdf python-dotenv
```

> Some dependencies may have GPU variants (e.g., `tensorflow-gpu`). Adjust based on your environment.

---

## Dataset (BraTS 2020)

This project expects the BraTS dataset in the following structure:

```text
BraTS2020_TrainingData/
  MICCAI_BraTS2020_TrainingData/
    BraTS20_Training_001/
      BraTS20_Training_001_flair.nii
      BraTS20_Training_001_t1ce.nii
      BraTS20_Training_001_seg.nii
    ...

BraTS2020_ValidationData/
  MICCAI_BraTS2020_ValidationData/
    BraTS20_Validation_001/
      BraTS20_Validation_001_flair.nii
      BraTS20_Validation_001_t1ce.nii
    ...
```

The training pipeline in `src/preprocessing.py` uses this structure and expects the path to match the hardcoded constant:

- `src/preprocessing.py`: `TRAIN_DATASET_PATH = "D:/Hackdata/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"`

If your dataset is elsewhere, update that constant (or refactor to use environment variables).

---

## Training the Model

Train a U-Net model using the BraTS training data:

```powershell
cd d:\Hackdata
python -m src.train --epochs 35 --lr 0.001 --dataset-path "D:/Hackdata/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" --save-path "D:/Hackdata/model/best_model.weights.h5"
```

The script will:

- Split data into train/val/test
- Create generators for 2-channel slices (FLAIR + T1CE)
- Train U-Net and save best weights to `model/best_model.weights.h5`

---

## Inference / API Server (FastAPI)

### 1) Configure environment variables

Create a `.env` file in `backend/` (or set environment variables) with:

```env
OPENAI_API_KEY=your_openai_api_key
MODEL_PATH=D:/Hackdata/model/best_model.weights.h5
DATASET_PATH=D:/Hackdata/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData
```

> Keep `OPENAI_API_KEY` secret.

### 2) Run the backend server

```powershell
cd d:\Hackdata\backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3) Key API endpoints

- `POST /doctor/signup` – create a doctor account
- `POST /doctor/login` – login and receive session info
- `POST /patients` – create a patient record
- `POST /upload-mri` – upload `.nii` MRI scan (runs segmentation pipeline)
- `GET /scan/{scan_id}/analysis` – get inference results
- `GET /scan/{scan_id}/report/download` – download PDF report
- `POST /ai/ask` – ask the AI assistant clinical questions

---

## Frontend (Doctor Dashboard)

The React UI lives in `doctor-dashboard/`.

```powershell
cd d:\Hackdata\doctor-dashboard
npm install
npm run dev
```

The dashboard is configured to talk to the backend at `http://localhost:8000`.

---

## Optional: Explainability (xAI)

The `xai_seg/` folder contains tools for visualizing Grad-CAM, SHAP, and other explainability techniques.

Install its dependencies:

```powershell
cd d:\Hackdata\xai_seg
pip install -r requirements.txt
```

Run the explainability scripts from that folder (see the code for usage patterns).

---

## Notes / Next Improvements

- Paths are currently hard-coded for a Windows `D:\Hackdata` base; consider converting them to environment variables or config files.
- Supabase credentials are currently checked in (`backend/database.py`). Replace with secure config management for production.
- The model is trained for 100 slices and uses a fixed slice range; adjust `VOLUME_START_AT` and `VOLUME_SLICES` if needed.

---

## Quick Sanity Check

Run a quick inference using the built-in script:

```powershell
cd d:\Hackdata
python inference.py
```

This will load `model/best_model.weights.h5`, run segmentation on the default validation case, and output `result.png`.

---

