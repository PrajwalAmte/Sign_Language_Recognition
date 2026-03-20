import json
import datetime
from pathlib import Path


class PredictionLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "predictions.jsonl"

    def log(self, letter: str, confidence: float, class_index: int):
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "letter": letter,
            "confidence": round(confidence, 4),
            "class_index": class_index,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
