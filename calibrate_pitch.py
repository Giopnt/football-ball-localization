import argparse
import json
from pathlib import Path

import cv2

from half_localization import locate_half


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Mark the midfield line and the A side."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Image path relative to the project folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing calibration.",
    )
    args = parser.parse_args()

    image_path = (ROOT / args.image).resolve()

    # JSON-ში პროექტთან მიმართებით ფარდობით გზას ვინახავთ.
    try:
        relative_image_path = image_path.relative_to(ROOT)
    except ValueError:
        parser.error("Place the image inside the project folder.")

    output_path = (
        ROOT / "data" / "calibrations" / f"{image_path.stem}.json"
    )

    if output_path.exists() and not args.overwrite:
        parser.error(
            f"Calibration already exists: {output_path.name}. "
            "Use --overwrite to replace it."
        )

    image = cv2.imread(str(image_path))
    if image is None:
        parser.error(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]

    # მხოლოდ საჩვენებელ ვერსიას ვაპატარავებთ.
    scale = min(1152 / width, 700 / height, 1.0)
    preview = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
    )

    # დაწკაპუნებები ორიგინალი ფოტოს კოორდინატებში გადავიყვანოთ.
    scale_x = width / preview.shape[1]
    scale_y = height / preview.shape[0]

    points = []
    labels = ["LINE 1", "LINE 2", "A SIDE"]
    window = "Midfield calibration"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            point = (
                round(x * scale_x, 2),
                round(y * scale_y, 2),
            )
            points.append(point)
            print(labels[len(points) - 1], point)

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_click)

    print("Click: LINE 1, LINE 2, A SIDE.")
    print("Enter: save | R: restart | Esc: cancel")

    try:
        while True:
            canvas = preview.copy()
            display_points = [
                (round(x / scale_x), round(y / scale_y))
                for x, y in points
            ]

            if len(display_points) >= 2:
                cv2.line(
                    canvas,
                    display_points[0],
                    display_points[1],
                    (0, 255, 255),
                    2,
                )

            for label, point in zip(labels, display_points):
                cv2.circle(canvas, point, 5, (0, 0, 255), 2)
                cv2.putText(
                    canvas,
                    label,
                    (point[0] + 8, max(20, point[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == 27 or cv2.getWindowProperty(
                window, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

            if key in (ord("r"), ord("R")):
                points.clear()

            if key in (10, 13) and len(points) == 3:
                line_start, line_end, team_a_point = points

                try:
                    # ბურთის გარეშე ვამოწმებთ ხაზსა და A წერტილს.
                    locate_half(
                        ball_center=None,
                        line_start=line_start,
                        line_end=line_end,
                        team_a_point=team_a_point,
                        margin_px=5,
                    )
                except ValueError as error:
                    print(error, "Press R to restart.")
                    continue

                calibration = {
                    "image": relative_image_path.as_posix(),
                    "image_size": [width, height],
                    "line_start": line_start,
                    "line_end": line_end,
                    "team_a_point": team_a_point,
                    "margin_px": 5,
                    "team_names": {"A": None, "B": None},
                }

                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(
                        calibration,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                print("Saved:", output_path)
                break

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()