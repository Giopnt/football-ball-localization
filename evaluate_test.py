import json
from datetime import datetime

import torch

# არსებული პარამეტრები და ლოკალური YOLO კონფიგურაცია.
from ball_pipeline import (
    ROOT,
    WEIGHTS,
    IMAGE_SIZE,
    DECISION_CONFIDENCE,
)
from ultralytics import YOLO


DATA = ROOT / "data" / "processed" / "football_ball_v1" / "data.yaml"

for path in (WEIGHTS, DATA):
    if not path.is_file():
        raise FileNotFoundError(path)

device = "mps" if torch.backends.mps.is_available() else "cpu"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = ROOT / "runs" / f"test_evaluation_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=False)

model = YOLO(str(WEIGHTS))

if model.names != {0: "ball"}:
    raise ValueError(f"Unexpected model classes: {model.names}")

evaluation_settings = {
    "data": str(DATA),
    "split": "test",
    "imgsz": IMAGE_SIZE,
    "batch": 2,
    "device": device,
    "workers": 0,
    "half": False,

    # დაბალი ზღვარი საჭიროა Precision–Recall მრუდისა და AP-ისთვის.
    "conf": 0.001,

    # NMS: დუბლირებული აღმოჩენების გაფილტვრის ზღვარი.
    "iou": 0.7,

    "plots": True,
    "verbose": False,
}

print(f"Device: {device}")
print(f"Evaluating test split at image size: {IMAGE_SIZE}")

metrics = model.val(
    **evaluation_settings,
    project=str(output_dir),
    name="metrics",
    exist_ok=False,
)

report = {
    "weights": str(WEIGHTS),
    "settings": evaluation_settings,
    "demo_decision_threshold": DECISION_CONFIDENCE,
    "metrics": {
        "precision_at_best_f1": float(metrics.box.mp),
        "recall_at_best_f1": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    },
}

report_path = output_dir / "test_report.json"
report_path.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(report, indent=2, ensure_ascii=False))
print(f"\nReport saved to: {report_path}")
print(f"Plots and examples: {output_dir / 'metrics'}")