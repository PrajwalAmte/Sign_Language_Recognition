"""
Live ASL sign prediction via webcam.

Usage:
  1. Start the API in a separate terminal:
       uvicorn api.main:app

  2. Run this script:
       python predict_live.py

Controls:
  SPACE  — predict the hand sign shown in the green box
  ESC    — quit
"""

import os
import io
import cv2
import requests
import numpy as np
from gtts import gTTS

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
ROI_SIZE = 450       # size of the capture box in pixels


def speak(text: str):
    tts = gTTS(text=text, lang="en")
    tmp = "/tmp/asl_pred.mp3"
    tts.save(tmp)
    os.system(f"afplay {tmp}")   # macOS; use 'mpg123' on Linux


def predict_crop(crop_bgr: np.ndarray) -> dict | None:
    _, buf = cv2.imencode(".png", crop_bgr)
    try:
        resp = requests.post(
            f"{API_URL}/predict/image",
            files={"file": ("frame.png", buf.tobytes(), "image/png")},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.ConnectionError:
        print("Cannot reach API — is 'uvicorn api.main:app' running?")
    return None


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    last_letter = ""
    last_conf = 0.0
    print("Camera open. Position hand in the green box, then press SPACE. ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Fixed ROI in the centre-right of the frame
        cx, cy = int(w * 0.70), h // 2
        x1, y1 = cx - ROI_SIZE // 2, cy - ROI_SIZE // 2
        x2, y2 = cx + ROI_SIZE // 2, cy + ROI_SIZE // 2
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Draw the capture box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Place hand here", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Overlay last prediction
        if last_letter:
            cv2.putText(frame, f"{last_letter}  ({last_conf:.1%})",
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                        (0, 200, 255), 3, cv2.LINE_AA)

        cv2.putText(frame, "SPACE=predict  ESC=quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1)

        cv2.imshow("ASL Sign Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:       # ESC
            break
        elif key == 32:     # SPACE
            crop = frame[y1:y2, x1:x2]
            pred = predict_crop(crop)
            if pred:
                last_letter = pred["letter"]
                last_conf = pred["confidence"]
                print(f"Predicted: {last_letter}  confidence: {last_conf:.1%}")
                speak(f"The sign is {last_letter}")
            else:
                print("Prediction failed.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

