import json

import cv2
import streamlit as st
import torch

# ეს import ჯერ ჩვენს YOLO პარამეტრებს ამზადებს.
from ball_pipeline import (
    ROOT,
    WEIGHTS,
    IMAGE_SIZE,
    CANDIDATE_CONFIDENCE,
    DECISION_CONFIDENCE,
    predict_half,
)
from ultralytics import YOLO
from visualize_prediction import draw_prediction


st.set_page_config(page_title="ბურთის მდებარეობა", layout="wide")


@st.cache_resource(scope="session", max_entries=1, show_spinner=False)
def load_model(weights_path, weights_version):
    # weights_version ფაილის ცვლილებისას ქეშს განაახლებს.
    return YOLO(weights_path)


st.title("რომელ ნახევარზეა ბურთი?")
st.caption(
    "აირჩიე ფოტო, რომლის შუახაზიც უკვე მონიშნულია. "
    "A და B კალიბრაციისას განსაზღვრული მხარეებია."
)

calibration_files = sorted(
    (ROOT / "data" / "calibrations").glob("*.json")
)

if not calibration_files:
    st.error("კალიბრაციის ფაილები ვერ მოიძებნა.")
    st.stop()

calibration_path = st.selectbox(
    "აირჩიე დაკალიბრებული ფოტო",
    calibration_files,
    format_func=lambda path: path.stem,
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
st.caption(f"გამოთვლის მოწყობილობა: {device.upper()}")

try:
    calibration_text = calibration_path.read_text(encoding="utf-8")
    calibration = json.loads(calibration_text)
    image_path = (ROOT / calibration["image"]).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"ფოტო ვერ გაიხსნა: {image_path.name}")

    weights_version = WEIGHTS.stat().st_mtime_ns

    input_id = (
        str(calibration_path),
        calibration_text,
        str(image_path),
        image_path.stat().st_mtime_ns,
        weights_version,
        IMAGE_SIZE,
        CANDIDATE_CONFIDENCE,
        DECISION_CONFIDENCE,
        device,
    )

except (OSError, ValueError, KeyError, TypeError) as error:
    st.error(f"ფაილების წაკითხვის შეცდომა: {error}")
    st.stop()

# სხვა ფოტოს ან შეცვლილი კალიბრაციისთვის ძველი პასუხი აღარ ვაჩვენოთ.
if st.session_state.get("input_id") != input_id:
    st.session_state["input_id"] = input_id
    st.session_state.pop("result", None)

if st.button("ბურთის მოძებნა", type="primary"):
    st.session_state.pop("result", None)

    try:
        with st.spinner("მიმდინარეობს ფოტოს ანალიზი..."):
            model = load_model(str(WEIGHTS), weights_version)

            answer = predict_half(
                model=model,
                image_path=image_path,
                calibration_path=calibration_path,
                device=device,
            )

            annotated = draw_prediction(image, calibration, answer)

            encoded_ok, encoded = cv2.imencode(".jpg", annotated)
            if not encoded_ok:
                raise RuntimeError(
                    "შედეგის სურათის მომზადება ვერ მოხერხდა."
                )

            st.session_state["result"] = {
                "answer": answer,
                "annotated": annotated,
                "jpg": encoded.tobytes(),
            }

    except Exception as error:
        st.error("ანალიზი ვერ დასრულდა.")
        st.exception(error)

result = st.session_state.get("result")

if result is None:
    st.image(
        image,
        channels="BGR",
        width="stretch",
        caption=image_path.name,
    )

else:
    answer = result["answer"]

    if answer["status"] == "ok":
        side = "A" if answer["half"] == "team_a_half" else "B"
        st.success(f"ბურთი {side} ნახევარზეა.")

        if answer["team_name"]:
            st.write(f"ამ ნახევრის გუნდი: {answer['team_name']}")

        st.write(
            "არჩეული კანდიდატის confidence: "
            f"{answer['ball']['confidence']:.3f}"
        )

    else:
        messages = {
            "not_detected":
                "ბურთი ვერ მოიძებნა — ნახევარი უცნობია.",
            "low_confidence":
                "კანდიდატის confidence დაბალია — ნახევარი უცნობია.",
            "multiple_candidates":
                "რამდენიმე სარწმუნო კანდიდატია — ნახევარი უცნობია.",
            "near_midline":
                "ბურთი შუახაზთან ახლოსაა — მხარეს ვერ ვასახელებთ.",
        }

        st.warning(
            messages.get(answer["status"], "ნახევარი უცნობია.")
        )

    st.image(
        result["annotated"],
        channels="BGR",
        width="stretch",
    )

    st.caption(
        "ყვითელი: შუახაზი და A მხარე · "
        "მწვანე: არჩეული ბურთი · "
        "ნარინჯისფერი: სხვა კანდიდატები"
    )

    with st.expander("დეტალური შედეგი — JSON"):
        st.json(answer)

    st.download_button(
        "JSON-ის ჩამოტვირთვა",
        data=json.dumps(
            answer, ensure_ascii=False, indent=2
        ).encode("utf-8"),
        file_name=f"{image_path.stem}_result.json",
        mime="application/json",
        on_click="ignore",
    )

    st.download_button(
        "მონიშნული ფოტოს ჩამოტვირთვა",
        data=result["jpg"],
        file_name=f"{image_path.stem}_result.jpg",
        mime="image/jpeg",
        on_click="ignore",
    )