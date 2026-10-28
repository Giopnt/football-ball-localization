import json

from half_localization import locate_half
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ბიბლიოთეკების დამხმარე ფაილები პროექტის შიგნით შევინახოთ.
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
PHOTOS = [
    ROOT / "data" / "samples" / f"sample_{number:02d}.jpg"
    for number in range(1, 4)
]

IMAGE_SIZE = 1280
CONFIDENCE = 0.05

# წინასწარ შევამოწმოთ, რომ მოდელი და სამივე ფოტო არსებობს.
for path in [WEIGHTS, *PHOTOS]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

model = YOLO(str(WEIGHTS))

# ჩვენს გაწვრთნილ მოდელს ერთი კლასი აქვს: 0 — ball.
if model.names != {0: "ball"}:
    raise ValueError(f"Unexpected model classes: {model.names}")

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Device: {device}")
print(f"Classes: {model.names}")

results = model.predict(
    source=[str(photo) for photo in PHOTOS],
    classes=[0],
    imgsz=IMAGE_SIZE,
    conf=CONFIDENCE,
    device=device,
    save=True,
    project=str(ROOT / "runs"),
    name=f"trained_samples_{IMAGE_SIZE}_conf{CONFIDENCE:.2f}",
    exist_ok=False,
    verbose=False,
)

for result in results:
    count = len(result.boxes)
    print(f"\n{Path(result.path).name}: {count} ball candidate(s)")

    for number, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf.item()

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        print(f"  Candidate {number}:")
        print(f"    confidence = {confidence:.3f}")
        print(f"    bbox = ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        print(f"    center = ({center_x:.1f}, {center_y:.1f})")

    if count == 0:
        print("  No ball detected at this confidence threshold.")
    elif count > 1:
        print("  Multiple candidates — inspect the image before choosing.")

print(f"\nAnnotated images: {results[0].save_dir}")

# მეორე ფოტოსთვის შენახული შუახაზის კალიბრაცია.
calibration_path = ROOT / "data" / "calibrations" / "sample_02.json"
calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

# კალიბრაცია მხოლოდ შესაბამის ფოტოზე გამოვიყენოთ.
target_image = (ROOT / calibration["image"]).resolve()

target_result = next(
    result
    for result in results
    if Path(result.path).resolve() == target_image
)

# კოორდინატები ფოტოს ზომასთანაა დაკავშირებული.
height, width = target_result.orig_shape

if calibration["image_size"] != [width, height]:
    raise ValueError("Calibration and image sizes do not match.")

print("\nMidfield check: sample_02.jpg")

if len(target_result.boxes) == 1:
    box = target_result.boxes[0]
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    # ცენტრს მოდელის ჩარჩოდან ვიღებთ.
    ball_center = ((x1 + x2) / 2, (y1 + y2) / 2)

    candidate_half = locate_half(
        ball_center=ball_center,
        line_start=calibration["line_start"],
        line_end=calibration["line_end"],
        team_a_point=calibration["team_a_point"],
        margin_px=calibration["margin_px"],
    )

    print(f"Detected center: {ball_center}")
    print(f"Confidence: {box.conf.item():.3f}")
    print(f"Candidate half (diagnostic): {candidate_half}")

else:
    print("Half: unknown — expected exactly one ball candidate.")