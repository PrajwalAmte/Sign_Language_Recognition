from pydantic import BaseModel, Field


class PixelRequest(BaseModel):
    pixels: list[float] = Field(
        ...,
        min_length=784,
        max_length=784,
        description="Flat list of 784 pixel values (28x28 grayscale, 0-255).",
    )


class PredictionResponse(BaseModel):
    letter: str
    confidence: float
    class_index: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
