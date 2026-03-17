"""
FastAPI backend for Sickle Cell Disease prediction.
Loads the trained MobileNetV2 model and serves predictions via REST API.

Usage:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  GET  /          — health check
  POST /predict   — upload an image, returns prediction + confidence
"""

import io
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from dataset_utils import build_val_transform
from model import build_mobilenet_v2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = Path("best_model.pth")
CLASS_NAMES: list[str] = ["Negative", "Positive"]
NUM_CLASSES: int = 2
IMAGE_SIZE: int = 224
DROPOUT: float = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model state (loaded once at startup)
# ---------------------------------------------------------------------------
_model: nn.Module | None = None
_transform = build_val_transform(IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Lifespan — load model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _model

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH.resolve()}. "
            "Run train.py first."
        )

    _model = build_mobilenet_v2(
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        freeze_backbone=False,
    )
    _model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    _model.to(DEVICE)
    _model.eval()
    print(f"Model loaded from {CHECKPOINT_PATH} on {DEVICE}")

    yield  # app runs here

    _model = None
    print("Model unloaded.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sickle Cell Disease Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    prediction: str          # "Positive" or "Negative"
    confidence: float        # 0.0 – 1.0
    positive_prob: float     # probability for Positive class
    negative_prob: float     # probability for Negative class


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health_check() -> dict[str, str]:
    """Health check — confirms the API and model are ready."""
    return {"status": "ok", "model": "MobileNetV2", "device": str(DEVICE)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Accept a blood cell image and return a sickle cell disease prediction.

    Args:
        file: Uploaded image file (JPEG / PNG).

    Returns:
        PredictionResponse with prediction label and confidence scores.

    Raises:
        HTTPException 400: If the file is not a valid image.
        HTTPException 503: If the model is not loaded.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Read and validate image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {exc}",
        ) from exc

    # Preprocess
    tensor = _transform(image).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

    # Inference
    with torch.no_grad():
        logits: torch.Tensor = _model(tensor)
        probs: torch.Tensor = torch.softmax(logits, dim=1)[0]

    negative_prob: float = probs[0].item()
    positive_prob: float = probs[1].item()
    predicted_idx: int = int(torch.argmax(probs).item())
    prediction: str = CLASS_NAMES[predicted_idx]
    confidence: float = probs[predicted_idx].item()

    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        positive_prob=round(positive_prob, 4),
        negative_prob=round(negative_prob, 4),
    )
