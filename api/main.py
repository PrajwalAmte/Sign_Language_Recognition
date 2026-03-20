import io
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image

from api.model import SignPredictor
from api.schemas import PixelRequest, PredictionResponse, HealthResponse
from monitoring.logger import PredictionLogger

app = FastAPI(title="Sign Language Recognition API", version="1.0.0")
predictor: SignPredictor | None = None
logger = PredictionLogger()


@app.on_event("startup")
def startup():
    global predictor
    predictor = SignPredictor()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=predictor is not None)


@app.post("/predict/pixels", response_model=PredictionResponse)
def predict_pixels(req: PixelRequest):
    letter, confidence, idx = predictor.predict(req.pixels)
    logger.log(letter, confidence, idx)
    return PredictionResponse(letter=letter, confidence=confidence, class_index=idx)


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    pixels = list(np.array(img, dtype=np.float32).flatten())
    letter, confidence, idx = predictor.predict(pixels)
    logger.log(letter, confidence, idx)
    return PredictionResponse(letter=letter, confidence=confidence, class_index=idx)
