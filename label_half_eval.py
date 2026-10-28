import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from half_localization import locate_half


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "football_ball_v1"
OUT = ROOT / "data" / "half_evaluation"

SEED = 42
PER_CLIP = 3

SKIP_REASONS = {
    ord("1"): "midline_not_visible",
    ord("2"): "ball_position_uncertain",
    ord("3"): "near_midline_uncertain",
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_manifest():
    path = OUT / "manifest.json"

    # განმეორებით გაშვებისას იმავე ფოტოებს ვიყენებთ.
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    groups = defaultdict(list)

    for split in ("val", "test"):
        for line in (DATA / f"{split}.txt").read_text().splitlines():
            if line.strip():
                image_path = Path(line.strip()).resolve()
                clip = image_path.name.split("_")[0]
                groups[(split, clip)].append(image_path)

    rng = random.Random(SEED)
    cases = []

    for (split, clip), paths in sorted(groups.items()):
        chosen = rng.sample(sorted(paths), PER_CLIP)

        for image_path in sorted(chosen):
            cases.append({
                "id": f"case_{len(cases) + 1:02d}",
                "image": image_path.relative_to(ROOT).as_posix(),
                "source_split": split,
                "source_clip": clip,
            })

    manifest = {
        "purpose": "Diagnostic half-localization audit; mixed val/test subset",
        "seed": SEED,
        "per_clip": PER_CLIP,
        "cases": cases,
    }

    save_json(path, manifest)
    return manifest


def annotate(case):
    image_path = ROOT / case["image"]
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read: {image_path}")

    height, width = image.shape[:2]

    label_path = (
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    )
    values = label_path.read_text().split()

    if len(values) != 5 or float(values[0]) != 0:
        raise ValueError(f"Expected one ball label: {label_path}")

    # ბურთის ცენტრი მონაცემთა ნაკრების არსებული მონიშვნიდან.
    gt_center = [
        float(values[1]) * width,
        float(values[2]) * height,
    ]

    scale = min(1152 / width, 700 / height, 1.0)
    preview = cv2.resize(
        image, (round(width * scale), round(height * scale))
    )

    scale_x = width / preview.shape[1]
    scale_y = height / preview.shape[0]

    gt_preview = (
        round(gt_center[0] / scale_x),
        round(gt_center[1] / scale_y),
    )

    points = []
    expected_half = None
    point_names = ["LINE 1", "LINE 2", "A SIDE"]
    window = "Label midfield evaluation"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append([
                round(x * scale_x, 2),
                round(y * scale_y, 2),
            ])
            print(point_names[len(points) - 1], points[-1])

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_click)

    print(f"\n{case['id']} | {case['source_split']} | {image_path.name}")
    print("Click LINE 1, LINE 2, A SIDE. Then A/B selects the true half.")
    print("Enter: save | R: reset | Esc: pause")
    print("Skip: 1=line hidden, 2=ball uncertain, 3=near line uncertain")

    record = {
        "image": case["image"],
        "ground_truth_ball_center": gt_center,
    }

    try:
        while True:
            canvas = preview.copy()

            display_points = [
                (round(x / scale_x), round(y / scale_y))
                for x, y in points
            ]

            if len(display_points) >= 2:
                cv2.line(
                    canvas, display_points[0], display_points[1],
                    (0, 255, 255), 2,
                )

            for name, point in zip(point_names, display_points):
                cv2.circle(canvas, point, 5, (0, 255, 255), 2)
                cv2.putText(
                    canvas, name,
                    (point[0] + 8, max(25, point[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1,
                )

            cv2.drawMarker(
                canvas, gt_preview, (255, 255, 0),
                cv2.MARKER_CROSS, 18, 1,
            )

            cv2.rectangle(
                canvas, (0, 0), (canvas.shape[1] - 1, 42),
                (25, 25, 25), -1,
            )

            caption = (
                f"{case['id']} | GT: cyan cross | "
                f"Your label: {expected_half or '-'}"
            )
            cv2.putText(
                canvas, caption, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == 27 or cv2.getWindowProperty(
                window, cv2.WND_PROP_VISIBLE
            ) < 1:
                return None

            if key in SKIP_REASONS:
                return None, {
                    **record,
                    "status": "skipped",
                    "expected_half": None,
                    "reason": SKIP_REASONS[key],
                }

            if key in (ord("r"), ord("R")):
                points.clear()
                expected_half = None

            if len(points) == 3 and key in (
                ord("a"), ord("A"), ord("b"), ord("B")
            ):
                expected_half = (
                    "team_a_half"
                    if key in (ord("a"), ord("A"))
                    else "team_b_half"
                )

            if key in (10, 13):
                if len(points) != 3 or expected_half is None:
                    print("Choose three points, then press A or B.")
                    continue

                try:
                    # მხოლოდ ხაზისა და A-ს წერტილის ვალიდაცია.
                    # ბურთს არ გადავცემთ და სწორ პასუხს არ ვითვლით.
                    locate_half(
                        None, points[0], points[1], points[2],
                        margin_px=5,
                    )
                except ValueError as error:
                    print(error, "Press R to reset.")
                    continue

                calibration = {
                    "image": case["image"],
                    "image_size": [width, height],
                    "line_start": points[0],
                    "line_end": points[1],
                    "team_a_point": points[2],
                    "margin_px": 5,
                    "team_names": {"A": None, "B": None},
                }

                return calibration, {
                    **record,
                    "status": "labeled",
                    "expected_half": expected_half,
                }

    finally:
        cv2.destroyAllWindows()


def main():
    manifest = get_manifest()
    labels_path = OUT / "labels.json"

    labels = (
        json.loads(labels_path.read_text())
        if labels_path.exists()
        else {}
    )

    for case in manifest["cases"]:
        if case["id"] in labels:
            continue

        outcome = annotate(case)

        if outcome is None:
            break

        calibration, record = outcome

        if calibration is not None:
            save_json(
                OUT / "calibrations" / f"{case['id']}.json",
                calibration,
            )

        labels[case["id"]] = record
        save_json(labels_path, labels)

        print(
            "Saved:", case["id"],
            record["status"], record.get("expected_half"),
        )

    counts = Counter(
        r["expected_half"]
        if r["status"] == "labeled"
        else f"skipped:{r['reason']}"
        for r in labels.values()
    )

    print(json.dumps({
        "selected": len(manifest["cases"]),
        "completed": len(labels),
        "remaining": len(manifest["cases"]) - len(labels),
        "counts": dict(counts),
    }, indent=2))

    print(f"Saved in: {OUT}")


if __name__ == "__main__":
    main()