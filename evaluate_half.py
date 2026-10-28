import json
from collections import Counter
from datetime import datetime

import cv2
import torch

from ball_pipeline import (
    ROOT, WEIGHTS, IMAGE_SIZE,
    CANDIDATE_CONFIDENCE, DECISION_CONFIDENCE,
    predict_half,
)
from ultralytics import YOLO
from analyze_errors import box_iou, read_ground_truth
from half_localization import locate_half
from visualize_prediction import draw_prediction


DATA = ROOT / "data" / "half_evaluation"
MATCH_IOU = 0.50
HALVES = ("team_a_half", "team_b_half")


def assess(answer, expected_half, ground_truth):
    answered = (
        answer["status"] == "ok"
        and answer["half"] in HALVES
    )

    if not answered:
        outcome = "unknown"
    else:
        outcome = (
            "correct_half"
            if answer["half"] == expected_half
            else "wrong_half"
        )

    selected_iou = (
        box_iou(answer["ball"]["bbox"], ground_truth)
        if answer["ball"] is not None
        else None
    )

    ball_correct = (
        selected_iou is not None
        and selected_iou >= MATCH_IOU
    )

    return {
        "outcome": outcome,
        "selected_ball_iou": selected_iou,
        "selected_ball_matched": ball_correct,
        "strict_success":
            outcome == "correct_half" and ball_correct,
    }


def summarize(rows):
    total = len(rows)
    counts = Counter(row["outcome"] for row in rows)

    correct = counts["correct_half"]
    wrong = counts["wrong_half"]
    answered = correct + wrong
    strict = sum(row["strict_success"] for row in rows)

    return {
        "images": total,
        "correct_half": correct,
        "wrong_half": wrong,
        "unknown": counts["unknown"],
        "answer_coverage":
            answered / total if total else None,
        "half_accuracy_when_answered":
            correct / answered if answered else None,
        "strict_successes": strict,
        "strict_success_rate":
            strict / total if total else None,
        "correct_half_without_ball_match": correct - strict,
        "geometry_matches_with_gt_center":
            sum(row["geometry_matches"] for row in rows),
        "status_counts": dict(
            Counter(row["prediction"]["status"] for row in rows)
        ),
    }


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    labels = json.loads((DATA / "labels.json").read_text())
    cases = manifest["cases"]

    missing = [
        case["id"] for case in cases
        if case["id"] not in labels
    ]
    if missing:
        raise ValueError(f"Finish labeling first: {missing}")

    if any(
        labels[c["id"]]["status"] not in ("labeled", "skipped")
        for c in cases
    ):
        raise ValueError("Unexpected annotation status.")

    eligible = [
        c for c in cases
        if labels[c["id"]]["status"] == "labeled"
    ]
    skipped = [
        {
            "id": c["id"],
            "reason": labels[c["id"]].get("reason"),
        }
        for c in cases
        if labels[c["id"]]["status"] == "skipped"
    ]

    if not eligible:
        raise ValueError("No labeled images to evaluate.")

    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "runs" / f"half_evaluation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(WEIGHTS))
    rows = []

    print(f"Device: {device} | Evaluating: {len(eligible)} images")

    for case in eligible:
        annotation = labels[case["id"]]
        expected = annotation["expected_half"]

        calibration_path = (
            DATA / "calibrations" / f"{case['id']}.json"
        )
        calibration = json.loads(calibration_path.read_text())

        if expected not in HALVES or annotation["image"] != case["image"]:
            raise ValueError(f"Invalid annotation: {case['id']}")

        if calibration["image"] != case["image"]:
            raise ValueError(
                f"Calibration image mismatch: {case['id']}"
            )

        image_path = ROOT / case["image"]
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Cannot read: {image_path}")

        # პროგნოზის ფუნქციას სწორ პასუხს არ ვაწვდით.
        answer = predict_half(
            model, image_path, calibration_path, device
        )

        # სწორ მონიშვნებს პროგნოზის შესაფასებლად ვიყენებთ.
        height, width = image.shape[:2]
        ground_truth = read_ground_truth(image_path, width, height)

        gt_center = [
            (ground_truth[0] + ground_truth[2]) / 2,
            (ground_truth[1] + ground_truth[3]) / 2,
        ]

        geometry_half = locate_half(
            gt_center,
            calibration["line_start"],
            calibration["line_end"],
            calibration["team_a_point"],
            calibration["margin_px"],
        )

        row = {
            "id": case["id"],
            "source_split": case["source_split"],
            "source_clip": case["source_clip"],
            "expected_half": expected,
            "ground_truth_box": ground_truth,
            "geometry_half_with_gt_center": geometry_half,
            "geometry_matches": geometry_half == expected,
            "prediction": answer,
            **assess(answer, expected, ground_truth),
        }

        # პროგნოზის სურათს სწორ მონიშვნასაც ვამატებთ.
        annotated = draw_prediction(image, calibration, answer)

        x1, y1, x2, y2 = [round(v) for v in ground_truth]
        cv2.rectangle(
            annotated, (x1, y1), (x2, y2), (255, 255, 0), 2
        )
        cv2.putText(
            annotated, "GT",
            (max(0, x1 - 5), max(20, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 0), 2,
        )

        cv2.rectangle(
            annotated, (0, 55), (width - 1, 93),
            (25, 25, 25), -1,
        )

        caption = (
            f"Expected: {expected} | {row['outcome']} | "
            f"Ball+half match: {row['strict_success']}"
        )
        cv2.putText(
            annotated, caption, (12, 81),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1,
        )

        output_image = output_dir / f"{case['id']}.jpg"

        if not cv2.imwrite(str(output_image), annotated):
            raise OSError(f"Could not save: {output_image}")

        row["annotated_image"] = str(output_image)
        rows.append(row)

        print(
            f"{case['id']}: expected={expected}, "
            f"predicted={answer['half']}, "
            f"status={answer['status']}, "
            f"strict_success={row['strict_success']}"
        )

    report = {
        "purpose": manifest["purpose"],
        "selected_images": len(cases),
        "skipped": skipped,
        "weights": str(WEIGHTS),
        "settings": {
            "device": device,
            "image_size": IMAGE_SIZE,
            "candidate_threshold": CANDIDATE_CONFIDENCE,
            "decision_threshold": DECISION_CONFIDENCE,
            "match_iou": MATCH_IOU,
        },
        "overall": summarize(rows),
        "by_source_split": {
            split: summarize([
                r for r in rows if r["source_split"] == split
            ])
            for split in sorted({
                r["source_split"] for r in rows
            })
        },
    }

    for filename, data in [
        ("summary.json", report),
        ("per_image.json", rows),
    ]:
        (output_dir / filename).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps(report, indent=2))
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()