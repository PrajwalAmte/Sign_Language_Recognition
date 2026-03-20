# Sign Language Recognition

CNN-based American Sign Language (ASL) fingerspelling recognition trained on the [Sign MNIST](https://www.kaggle.com/datamunge/sign-language-mnist) dataset. Recognises 24 static hand signs (A-Y, excluding J and Z which require motion).

## ASL Alphabet Reference

![ASL Gesture](sign_mnist/asl_sign.png)

## Project Structure

```
├── params.yaml              # Centralised hyperparameters
├── dvc.yaml                 # DVC pipeline (preprocess → train → evaluate)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── preprocess.py        # CSV → normalised .npz arrays
│   ├── train.py             # CNN training with MLflow tracking
│   └── evaluate.py          # Classification report & confusion matrix
├── api/
│   ├── main.py              # FastAPI endpoints
│   ├── model.py             # SignPredictor wrapper
│   └── schemas.py           # Pydantic request/response models
├── monitoring/
│   └── logger.py            # JSONL prediction logger
├── data/raw/                # Sign MNIST CSVs
└── .github/workflows/ci.yml # CI with accuracy gate
```

## Quick Start (Local)

```bash
python -m venv .venv --python=python3.11
source .venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline
dvc repro

# Or step by step
python -m src.preprocess
python -m src.train
python -m src.evaluate
```

## Docker

```bash
# Train + evaluate + serve API
docker compose up api

# Train only
docker compose up train

# MLflow UI
docker compose up mlflow
# Open http://localhost:5000
```

## API Usage

```bash
# Start the API server
uvicorn api.main:app --reload

# Health check
curl http://localhost:8000/health

# Predict from pixel array
curl -X POST http://localhost:8000/predict/pixels \
  -H "Content-Type: application/json" \
  -d '{"pixels": [0, 0, ..., 255]}'

# Predict from image upload
curl -X POST http://localhost:8000/predict/image \
  -F "file=@hand_sign.png"
```

## Model

| Layer | Output Shape | Params |
|-------|-------------|--------|
| Conv2D (75, 3×3) + BN + MaxPool | 14×14×75 | 750 |
| Conv2D (50, 3×3) + BN + MaxPool | 7×7×50 | 33,800 |
| Conv2D (25, 3×3) + BN + MaxPool | 4×4×25 | 11,275 |
| Dense (512) + Dropout (0.3) | 512 | 204,800 |
| Dense (24, softmax) | 24 | 12,312 |

Total trainable parameters: ~264,000

## Tech Stack

- **TensorFlow / Keras 3** — model training & inference
- **MLflow** — experiment tracking
- **FastAPI** — REST API for inference
- **DVC** — reproducible ML pipeline
- **Docker** — containerised training & serving
- **GitHub Actions** — CI with 90% accuracy gate

2. **Loading the Pre-trained CNN Model**: Load the pre-trained CNN model (`smnist.h5`) for gesture recognition.

3. **Initializing the Camera Feed**: Initialize the camera feed to capture video frames.

4. **Capturing and Processing Video Frames**: Continuously capture and process video frames from the camera.

5. **Gesture Prediction**: Press the **Space** key to predict the ASL gesture. The predicted gesture will be displayed on the screen.

6. **Closing the Window**: Press the **ESC** key to close the video feed window.

7. **Converting Predictions into Speech using gTTS**: Utilize the gTTS library to convert predictions into speech.

8. **Playing the Generated Speech**: Play the generated speech using an appropriate audio player.


To predict and convert ASL gestures, run the following command:


```
python prediction.py
```

## Requirements

- Python (version 3.x)
- Required libraries (install using pip):

```bash
pip install numpy pandas matplotlib seaborn keras scikit-learn opencv-python mediapipe gTTS
```

## Acknowledgements

This project makes use of the Mediapipe library for hand gesture recognition and the Google Text-to-Speech (gTTS) library for speech synthesis. The ASL gesture dataset used for this project was obtained from [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) and serves as the primary dataset for training and testing the ASL gesture recognition model.

## License

This project is licensed under the MIT License. Refer to the `LICENSE` file for more information.

Feel free to contribute and adapt this project to enhance accessibility and communication for individuals using American Sign Language.
