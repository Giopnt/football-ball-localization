import cv2


def draw_prediction(image, calibration, answer):
    canvas = image.copy()

    line_start = tuple(round(v) for v in calibration["line_start"])
    line_end = tuple(round(v) for v in calibration["line_end"])
    a_point = tuple(round(v) for v in calibration["team_a_point"])

    # OpenCV-ში ფერების მიმდევრობაა BGR.
    yellow = (0, 255, 255)
    green = (0, 220, 0)
    orange = (0, 165, 255)

    cv2.line(canvas, line_start, line_end, yellow, 3, cv2.LINE_AA)
    cv2.circle(canvas, a_point, 7, yellow, 2)

    cv2.putText(
        canvas,
        "A side",
        (a_point[0] + 12, max(25, a_point[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        yellow,
        2,
    )

    selected_id = answer["ball"]["id"] if answer["ball"] else None

    for candidate in answer["candidates"]:
        x1, y1, x2, y2 = [round(v) for v in candidate["bbox"]]
        center = tuple(round(v) for v in candidate["center"])

        selected = candidate["id"] == selected_id
        color = green if selected else orange

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        if selected:
            cv2.circle(canvas, center, 4, color, -1)

        label = f"#{candidate['id']} conf={candidate['confidence']:.3f}"
        position = (max(0, x1 - 20), max(25, y1 - 12))

        # შავი კონტური ტექსტს ბალახის ფონზეც წაკითხვადს ხდის.
        for text_color, thickness in [((0, 0, 0), 3), (color, 1)]:
            cv2.putText(
                canvas, label, position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, text_color, thickness, cv2.LINE_AA,
            )

    messages = {
        "not_detected": "Half unknown: no ball detected",
        "low_confidence": "Half unknown: low confidence",
        "multiple_candidates": "Half unknown: multiple candidates",
        "near_midline": "Ball near the midfield line",
    }

    if answer["status"] == "ok":
        side = "A" if answer["half"] == "team_a_half" else "B"
        confidence = answer["ball"]["confidence"]
        message = f"Ball: team {side} half | confidence={confidence:.3f}"
    else:
        message = messages.get(answer["status"], "Half unknown")

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 55), (25, 25, 25), -1)
    cv2.putText(
        canvas, message, (15, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )

    return canvas