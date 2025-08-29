import os
import uuid
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .infer import InferenceEngine, preprocess_image_to_tensor, save_gradcam_overlay
from .rag import KB

app = FastAPI(title="Medical Scanner RAG Demo", version="0.1.0")

# CORS
_allowed = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173")
ALLOWED_ORIGINS = [o.strip() for o in _allowed.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

app.mount("/media", StaticFiles(directory="media"), name="media")

engine = InferenceEngine()
kb = KB(base_path="kb")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/kb/search")
def kb_search(q: str = Query(..., min_length=2)):
    docs = kb.search(q, top_k=5)
    return {"query": q, "results": docs}

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    raw = await file.read()

    try:
        pil_img, tensor = preprocess_image_to_tensor(raw, filename=file.filename)
    except Exception as e:
        # DICOM corrompu/compressé sans plugin, ou image non supportée
        raise HTTPException(status_code=415, detail=f"Unsupported image or DICOM decode error: {e}")

    pred = engine.predict(tensor)
    top_label = max(pred, key=pred.get)

    query = f"CT scanner findings and guidelines for {top_label} detection"
    results = kb.search(query, top_k=3)

    out_name = f"gradcam_{uuid.uuid4().hex}.png"
    out_path = f"media/{out_name}"
    save_gradcam_overlay(engine, pil_img, tensor, out_path)

    return JSONResponse({
        "predictions": pred,
        "top_label": top_label,
        "gradcam_url": f"/media/{out_name}",
        "rag": results
    })
