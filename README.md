# Automatic Number Plate Recognition (ANPR) Dashboard

A computer-vision project that detects vehicle number plates and extracts plate text from uploaded images. It combines a YOLO-based plate detector, EasyOCR, image preprocessing, and a FastAPI web dashboard.

## Features

- Upload an image for number-plate detection.
- Read detected plate text using EasyOCR and image-processing variants.
- View annotated output images and processed plate crops.
- Store detection history locally in SQLite.
- Review manually validated detections and evaluation metrics.

## Technology stack

- Python
- FastAPI and Uvicorn
- YOLO / Ultralytics
- EasyOCR
- OpenCV and NumPy
- SQLite
- HTML, CSS, and JavaScript

## Project structure

```text
ANPR-Number-Plate-Recognition/
├── app.py
├── evaluation.py
├── evaluate.py
├── ANPR_best.pt
├── requirements.txt
├── .gitignore
├── README.md
└── static/
    ├── index.html
    ├── css/
    └── js/
```

## Setup (Windows)

1. Install Python 3.10 or 3.11 and create a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Ensure `ANPR_best.pt` is in the project root, alongside `app.py`.
3. Start the server:

   ```powershell
   uvicorn app:app --host 127.0.0.1 --port 8000
   ```

4. Open http://127.0.0.1:8000 in your browser.

The first EasyOCR run may download model files. The project code currently initializes EasyOCR with `gpu=True`, so a compatible CUDA/PyTorch setup and GPU are required as written. For CPU-only use, change this to `gpu=False` in `app.py` before running.

## Evaluation

Run the evaluation script after creating and validating detections in the dashboard:

```powershell
python evaluate.py
```

## Data and privacy

This repository intentionally excludes the original image collections, named student folders, nested archives, and local detection-history database. Only upload images that you have permission to publish. Number plates can be identifying information; use consented or appropriately anonymized demo images.

## Notes

- The included `ANPR_best.pt` is a trained model artifact. Verify that you have permission to redistribute it before making this repository public.
- Results depend on image quality, lighting, camera angle, and the model's training data. Evaluate on a separate test set before making accuracy claims.
- This repository is a project demonstration, not a production-ready vehicle identification system.
