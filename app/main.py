"""X-Ray Vision — FastAPI application.

Upload a bone X-ray, get the probability that it contains an abnormality
plus a SmoothGrad sensitivity map highlighting the suspicious region.
"""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .inference import XRayModel

STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_TYPES = {"image/png", "image/jpeg"}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = XRayModel()
    yield


app = FastAPI(
    title="X-Ray Vision",
    description="Detection and visualization of bone abnormalities in "
    "X-ray images using convolutional neural networks.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Please upload a PNG or JPEG image.")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 16 MB.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(422, "The uploaded file is not a valid image.")

    result = app.state.model.analyze(image)

    return {
        "abnormal_probability": result.abnormal_probability,
        "normal_probability": result.normal_probability,
        "input_image": _data_url(result.input_png),
        "heatmap": _data_url(result.heatmap_png),
    }


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
