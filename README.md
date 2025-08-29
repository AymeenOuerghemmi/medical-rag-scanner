# Full‑stack Medical Scanner + RAG (Demo)

- Backend: FastAPI (PyTorch ResNet18), DICOM support, Grad‑CAM, simple TF‑IDF RAG
- Frontend: React + Vite (upload, predictions, Grad‑CAM, RAG snippets)
- Docker: `docker-compose up --build`

## Dev quick start
1) API
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
2) UI
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and point `VITE_API_BASE` to `http://localhost:8000` if needed.

## Disclaimer
This is a research/education demo only — not for clinical use.
