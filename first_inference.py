"""პირველი inference: COCO-ზე pretrained YOLO11n და სამი ფეხბურთის ფოტო.

გაშვება: .venv/bin/python first_inference.py
ჯერ არ ვსწავლობთ მოდელს და არ ვადგენთ გუნდის ნახევარს.
"""

import os
from pathlib import Path

# მისამართები ამ ფაილის საქაღალდეს ეყრდნობა, ამიტომ სხვა საქაღალდიდანაც მუშაობს.
ROOT = Path(__file__).resolve().parent
(ROOT / ".cache" / "ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

import torch
from ultralytics import YOLO, settings

settings.update({"sync": False, "hub": False})

WEIGHTS = ROOT / "models" / "yolo11n.pt"
PHOTOS = ROOT / "data" / "samples"
IMAGE_SIZE = 640
CONFIDENCE = 0.25

if not WEIGHTS.is_file():
    raise FileNotFoundError(f"Model weights are missing: {WEIGHTS}")
if not list(PHOTOS.glob("*.jpg")):
    raise FileNotFoundError(f"No JPG photos found in: {PHOTOS}")

model = YOLO(str(WEIGHTS))
ball_class_id = next(i for i, name in model.names.items() if name == "sports ball")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}; class: {ball_class_id} (sports ball)")

results = model.predict(
    source=str(PHOTOS),
    classes=[ball_class_id],
    imgsz=IMAGE_SIZE,
    conf=CONFIDENCE,
    device=device,
    save=True,
    project=str(ROOT / "runs"),
    name=f"first_inference_{device}_{IMAGE_SIZE}",
    exist_ok=True,
    verbose=False,
)

for result in results:
    print(f"\n{Path(result.path).name}: {len(result.boxes)} ball candidate(s)")
    for box in result.boxes:
        coordinates = [round(value, 1) for value in box.xyxy[0].tolist()]
        confidence = box.conf.item()
        print(f"  xyxy={coordinates}; confidence={confidence:.3f}")
    if len(result.boxes) == 0:
        print(f"  No sports ball prediction reached confidence {CONFIDENCE}.")

print(f"\nAnnotated images: {results[0].save_dir}")
