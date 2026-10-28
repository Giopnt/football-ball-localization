import json
from collections import Counter
from datetime import datetime
from math import isfinite
from pathlib import Path

import cv2
import torch

from ball_pipeline import (
    ROOT, WEIGHTS, IMAGE_SIZE,
    CANDIDATE_CONFIDENCE, DECISION_CONFIDENCE,
)
from ultralytics import YOLO


SPLIT_FILE = ROOT / "data" / "processed" / "football_ball_v1" / "test.txt"
MATCH_IOU = 0.50
MAX_EXAMPLES = 3

STATUSES = (
    "single_correct",
    "single_incorrect",
    "multiple_candidates",
    "low_confidence",
    "not_detected",
)


def box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])

    intersection = max(0, right - left) * max(0, bottom - top)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def read_ground_truth(image_path, width, height):
    label_path = (
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    )
    lines = label_path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip()]

    # ეს სკრიპტი ჩვენი ერთბურთიანი მონაცემებისთვისაა.
    if len(lines) != 1:
        raise ValueError(f"Expected exactly one labeled ball: {label_path}")

    values = [float(value) for value in lines[0].split()]

    if len(values) != 5 or not all(isfinite(v) for v in values):
        raise ValueError(f"Invalid label: {label_path}")

    class_id, xc, yc, bw, bh = values

    if class_id != 0 or not (
        0 <= xc <= 1 and 0 <= yc <= 1
        and 0 < bw <= 1 and 0 < bh <= 1
    ):
        raise ValueError(f"Invalid ball label: {label_path}")

    # YOLO-ს ნორმალიზებული კოორდინატები გადაგვყავს პიქსელებში.
    return [
        (xc - bw / 2) * width,
        (yc - bh / 2) * height,
        (xc + bw / 2) * width,
        (yc + bh / 2) * height,
    ]


def score_candidates(candidates, ground_truth):
    accepted = sorted(
        [
            c for c in candidates
            if c["confidence"] >= DECISION_CONFIDENCE
        ],
        key=lambda c: c["confidence"],
        reverse=True,
    )

    # ერთ მონიშნულ ბურთს მაქსიმუმ ერთი TP შეესაბამება.
    match = next(
        (
            c for c in accepted
            if box_iou(c["bbox"], ground_truth) >= MATCH_IOU
        ),
        None,
    )

    tp = int(match is not None)
    fp = len(accepted) - tp
    fn = 1 - tp

    if not candidates:
        status = "not_detected"
    elif not accepted:
        status = "low_confidence"
    elif len(accepted) > 1:
        status = "multiple_candidates"
    else:
        status = "single_correct" if tp else "single_incorrect"

    return {
        "status": status,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_id": match["id"] if match else None,
        "accepted_count": len(accepted),
    }


def save_example(image, ground_truth, candidates, score, destination):
    canvas = image.copy()

    def draw_box(box, color, label, below=False):
        x1, y1, x2, y2 = [round(v) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        x = min(max(5, x1), max(5, canvas.shape[1] - 180))
        y = min(
            canvas.shape[0] - 8,
            max(25, y2 + 20 if below else y1 - 10),
        )

        for text_color, thickness in [((0, 0, 0), 3), (color, 1)]:
            cv2.putText(
                canvas, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                text_color, thickness, cv2.LINE_AA,
            )

    for candidate in candidates:
        if candidate["confidence"] < DECISION_CONFIDENCE:
            color, tag = (0, 165, 255), "LOW"
        elif candidate["id"] == score["matched_id"]:
            color, tag = (0, 220, 0), "TP"
        else:
            color, tag = (0, 0, 255), "FP"

        draw_box(
            candidate["bbox"],
            color,
            f"{tag} {candidate['confidence']:.3f}",
            below=True,
        )

    draw_box(ground_truth, (255, 255, 0), "GT")

    title = (
        f"{score['status']} | TP={score['tp']} "
        f"FP={score['fp']} FN={score['fn']}"
    )

    cv2.rectangle(
        canvas, (0, 0), (canvas.shape[1] - 1, 45), (25, 25, 25), -1
    )
    cv2.putText(
        canvas, title, (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2, cv2.LINE_AA,
    )

    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Could not save: {destination}")


def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)

    image_paths = [
        Path(line.strip())
        for line in SPLIT_FILE.read_text().splitlines()
        if line.strip()
    ]

    if not image_paths or len(set(image_paths)) != len(image_paths):
        raise ValueError("The test image list is empty or contains duplicates.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "runs" / f"error_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(WEIGHTS))

    if model.names != {0: "ball"}:
        raise ValueError(f"Unexpected model classes: {model.names}")

    counts = Counter({status: 0 for status in STATUSES})
    saved = Counter()
    rows = []

    print(f"Device: {device} | Images: {len(image_paths)}")

    for number, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Cannot decode: {image_path}")

        height, width = image.shape[:2]
        ground_truth = read_ground_truth(image_path, width, height)

        # იგივე inference პარამეტრები, რაც ჩვენს აპში.
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

        candidates = [
            {
                "id": index,
                "confidence": float(box.conf.item()),
                "bbox": box.xyxy[0].tolist(),
            }
            for index, box in enumerate(result.boxes, start=1)
        ]

        score = score_candidates(candidates, ground_truth)
        status = score["status"]
        counts[status] += 1
        example_path = None

        # თითოეული კატეგორიიდან პირველი სამი მაგალითი შევინახოთ.
        if saved[status] < MAX_EXAMPLES:
            folder = output_dir / "examples" / status
            folder.mkdir(parents=True, exist_ok=True)

            example_path = folder / f"{number:03d}_{image_path.name}"
            save_example(
                image, ground_truth, candidates, score, example_path
            )
            saved[status] += 1

        rows.append({
            "image": str(image_path),
            "ground_truth": ground_truth,
            "candidates": candidates,
            **score,
            "example": str(example_path) if example_path else None,
        })

        if number % 10 == 0 or number == len(image_paths):
            print(f"Processed {number}/{len(image_paths)}")

    tp, fp, fn = [
        sum(row[key] for row in rows)
        for key in ("tp", "fp", "fn")
    ]

    total = len(rows)
    single = counts["single_correct"] + counts["single_incorrect"]

    assert tp + fn == total
    assert sum(counts.values()) == total

    summary = {
        "split": "test",
        "images": total,
        "weights": str(WEIGHTS),
        "device": device,
        "image_size": IMAGE_SIZE,
        "candidate_threshold": CANDIDATE_CONFIDENCE,
        "decision_threshold": DECISION_CONFIDENCE,
        "match_iou": MATCH_IOU,
        "nms_iou": 0.7,
        "detection": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / total,
            "f1": 2 * tp / (2 * tp + fp + fn),
        },
        "selection_before_half_check": {
            "counts": dict(counts),
            "single_candidate_rate": single / total,
            "correct_single_candidate_rate":
                counts["single_correct"] / total,
            "precision_among_single_candidates":
                counts["single_correct"] / single if single else None,
        },
    }

    for filename, data in [
        ("summary.json", summary),
        ("per_image.json", rows),
    ]:
        (output_dir / filename).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2))
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()