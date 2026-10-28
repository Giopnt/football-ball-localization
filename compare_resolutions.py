import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for folder in ("ultralytics", "matplotlib"):
    (ROOT / ".cache" / folder).mkdir(parents=True, exist_ok=True)

os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".cache" / "ultralytics")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

import torch
from ultralytics import YOLO, settings

settings.update({"sync": False, "hub": False})

WEIGHTS = ROOT / "runs" / "ball_yolo11n_1280_v1" / "weights" / "best.pt"
DATA = ROOT / "data" / "processed" / "football_ball_v1" / "data.yaml"

for path in (WEIGHTS, DATA):
    if not path.is_file():
        raise FileNotFoundError(path)

device = "mps" if torch.backends.mps.is_available() else "cpu"

# ყოველ გაშვებას თავისი შედეგების საქაღალდე ექნება.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = ROOT / "runs" / f"resolution_comparison_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

model = YOLO(str(WEIGHTS))

# ორივე შეფასებისთვის საერთო პარამეტრები.
validation_settings = {
    "data": str(DATA),
    "split": "val",
    "batch": 2,
    "device": device,
    "workers": 0,
    "half": False,
    "conf": 0.001,
    "iou": 0.7,
    "plots": True,
    "verbose": False,
}

report = {
    "weights": str(WEIGHTS),
    "settings": validation_settings,
    "results": [],
}

for image_size in (1280, 1600):
    print(f"\nEvaluating image size: {image_size}")

    metrics = model.val(
        **validation_settings,
        imgsz=image_size,
        project=str(output_dir),
        name=f"imgsz_{image_size}",
        exist_ok=False,
    )

    row = {
        "image_size": image_size,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }

    report["results"].append(row)
    print(json.dumps(row, indent=2))

    # პირველი შეფასებაც შენახული იქნება, სანამ მეორე დასრულდება.
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

print(f"\nComparison saved to: {output_dir / 'comparison.json'}")