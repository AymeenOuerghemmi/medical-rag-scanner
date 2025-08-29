# Backend (FastAPI) — Medical Scanner + RAG

## Quick start

```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Endpoints
- `POST /infer` — multipart form with `file` (DICOM `.dcm` or PNG/JPG). Returns:
  - `predictions`: dict of class→probability
  - `top_label`: highest‑probability class
  - `gradcam_url`: path to Grad‑CAM overlay image
  - `rag`: top‑3 knowledge snippets (TF‑IDF cosine similarity)

- `GET /kb/search?q=...` — text search over `kb/*.md`

## Model
- ResNet18 backbone with 4 classes: `normal`, `pneumonia`, `covid19`, `pulmonary_embolism`.
- If `models/model.pt` is present, it will be loaded. Otherwise, ImageNet weights are used with a fresh final layer (not accurate for medical use).

## Notes
- This is a **research/demo** only. Do **not** use for medical diagnosis.
- DICOM windowing defaults (lung-ish): window center = -600, width = 1500. Adjust in `infer.py` if needed.
