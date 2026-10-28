from math import hypot


def signed_distance(point, line_start, line_end):
    """წერტილიდან ხაზამდე მანძილი, მხარის აღმნიშვნელი ნიშნით."""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1
    length = hypot(dx, dy)

    if length == 0:
        raise ValueError("Choose two different points for the midfield line.")

    return (dx * (y - y1) - dy * (x - x1)) / length


def locate_half(
    ball_center,
    line_start,
    line_end,
    team_a_point,
    margin_px=5,
):
    """დაადგინე ბურთის ნახევარი ფოტოს კოორდინატების მიხედვით."""
    if margin_px < 0:
        raise ValueError("margin_px must be non-negative.")

    reference_distance = signed_distance(
        team_a_point, line_start, line_end
    )

    if abs(reference_distance) <= margin_px:
        raise ValueError("Choose team_a_point farther from the midfield line.")

    if ball_center is None:
        return "unknown"

    ball_distance = signed_distance(
        ball_center, line_start, line_end
    )

    if abs(ball_distance) <= margin_px:
        return "near_midline"

    if ball_distance * reference_distance > 0:
        return "team_a_half"

    return "team_b_half"


if __name__ == "__main__":
    # ხელოვნური კოორდინატები ლოგიკის შესამოწმებლად.
    line_start = (500, 100)
    line_end = (700, 900)
    team_a_point = (300, 500)

    examples = [
        (400, 500),
        (800, 500),
        (600, 500),
        None,
    ]

    for ball_center in examples:
        half = locate_half(
            ball_center,
            line_start,
            line_end,
            team_a_point,
        )
        print(ball_center, "->", half)