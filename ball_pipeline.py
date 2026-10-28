from visualize_prediction import draw_prediction
import json
import os
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "ball_yolo11n_1280_v1" / "weights" / "best.pt"

IMAGE_SIZE = 1280
CANDIDATE_CONFIDENCE = 0.05
DECISION_CONFIDENCE = 0.332

for folder in ("ultralytics", "matplotlib"):
    (ROOT / ".cache" / folder).mkdir(parents=True, exist_ok=True)

os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".cache" / "ultralytics")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

import cv2
import torch
from ultralytics import YOLO, settings

from half_localization import locate_half

settings.update({"sync": False, "hub": False})


def decide_half(candidates, calibration):
    answer = {
        "status": "not_detected",
        "half": "unknown",
        "ball": None,
        "team_name": None,
    }

    if not candidates:
        return answer

    accepted = [
        candidate for candidate in candidates
        if candidate["confidence"] >= DECISION_CONFIDENCE
    ]

    if not accepted:
        answer["status"] = "low_confidence"
        return answer

    if len(accepted) > 1:
        answer["status"] = "multiple_candidates"
        return answer

    ball = accepted[0]
    half = locate_half(
        ball_center=ball["center"],
        line_start=calibration["line_start"],
        line_end=calibration["line_end"],
        team_a_point=calibration["team_a_point"],
        margin_px=calibration["margin_px"],
    )

    answer["ball"] = ball
    answer["half"] = half
    answer["status"] = "near_midline" if half == "near_midline" else "ok"

    team_key = {"team_a_half": "A", "team_b_half": "B"}.get(half)
    if team_key:
        answer["team_name"] = calibration.get("team_names", {}).get(team_key)

    return answer


def predict_half(model, image_path, calibration_path, device):
    image_path = (ROOT / image_path).resolve()
    calibration_path = (ROOT / calibration_path).resolve()

    for path in (image_path, calibration_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if model.names != {0: "ball"}:
        raise ValueError(f"Unexpected model classes: {model.names}")

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot decode image: {image_path}")

    height, width = image.shape[:2]

    if (ROOT / calibration["image"]).resolve() != image_path:
        raise ValueError("This calibration belongs to a different image.")

    if calibration["image_size"] != [width, height]:
        raise ValueError("Calibration and image sizes do not match.")

    for key in ("line_start", "line_end", "team_a_point"):
        point = calibration[key]
        if len(point) != 2 or not all(isfinite(v) for v in point):
            raise ValueError(f"Invalid calibration point: {key}")
        x, y = point
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Calibration point outside image: {key}")

    margin = calibration["margin_px"]
    if not isfinite(margin) or margin < 0:
        raise ValueError("margin_px must be finite and non-negative.")

    # მოდელის გაშვებამდე ხაზისა და A-ს მხარის სისწორეს ვამოწმებთ.
    locate_half(
        None, calibration["line_start"], calibration["line_end"],
        calibration["team_a_point"], margin,
    )

    result = model.predict(
        source=image,
        classes=[0],
        imgsz=IMAGE_SIZE,
        conf=CANDIDATE_CONFIDENCE,
        iou=0.7,
        device=device,
        half=False,
        save=False,
        verbose=False,
    )[0]

    candidates = []
    for number, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        candidates.append({
            "id": number,
            "confidence": float(box.conf.item()),
            "bbox": [x1, y1, x2, y2],
            "center": [(x1 + x2) / 2, (y1 + y2) / 2],
        })

    answer = decide_half(candidates, calibration)
    answer.update({
        "image": (
            image_path.relative_to(ROOT).as_posix()
            if image_path.is_relative_to(ROOT) else image_path.name
        ),
        "image_size": IMAGE_SIZE,
        "candidate_threshold": CANDIDATE_CONFIDENCE,
        "decision_threshold": DECISION_CONFIDENCE,
        "candidates": candidates,
    })
    return answer


if __name__ == "__main__":
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(WEIGHTS))

    image_path = ROOT / "data" / "samples" / "sample_03.jpg"
    calibration_path = ROOT / "data" / "calibrations" / "sample_03.json"

    answer = predict_half(
        model=model,
        image_path=image_path,
        calibration_path=calibration_path,
        device=device,
    )

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot decode image: {image_path}")

    calibration = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )

    annotated = draw_prediction(image, calibration, answer)

    output_dir = ROOT / "runs" / "half_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_image = output_dir / f"{image_path.stem}_result.jpg"
    output_json = output_dir / f"{image_path.stem}_result.json"

    if not cv2.imwrite(str(output_image), annotated):
        raise OSError(f"Could not save image: {output_image}")

    output_json.write_text(
        json.dumps(answer, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(answer, indent=2, ensure_ascii=False))
    print(f"\nAnnotated image: {output_image}")
    print(f"JSON result: {output_json}")
