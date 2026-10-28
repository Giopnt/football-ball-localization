import gc
import json
from datetime import datetime
from importlib.metadata import version

import torch

from ball_pipeline import ROOT, WEIGHTS, IMAGE_SIZE
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix


DATA = ROOT / "data" / "processed" / "football_ball_v1" / "data.yaml"
BASELINE = ROOT / "models" / "yolo11n.pt"


def find_ball_class(names):
    matches = [
        class_id
        for class_id, name in names.items()
        if name in ("ball", "sports ball")
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one ball class, found: {matches}"
        )

    return matches[0]


class BallOnlyValidator(DetectionValidator):
    def init_metrics(self, model):
        super().init_metrics(model)

        self.source_ball_class = find_ball_class(model.names)
        self.source_class_count = len(model.names)

        # შეფასებისას ორივე მოდელის ბურთს კლასი 0 ექნება.
        self.names = {0: "ball"}
        self.nc = 1
        self.metrics.names = self.names
        self.class_map = [1]

        self.confusion_matrix = ConfusionMatrix(
            names=self.names,
            save_matches=self.args.plots and self.args.visualize,
        )

    def postprocess(self, predictions):
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

        if (
            self.end2end
            or predictions.ndim != 3
            or predictions.shape[1] != 4 + self.source_class_count
        ):
            raise ValueError(
                "Unexpected prediction format for YOLO11n."
            )

        # პირველი 4 არხი ჩარჩოს კოორდინატებია.
        # შემდეგი არხებიდან მხოლოდ ბურთის confidence-ს ვიღებთ.
        channels = [0, 1, 2, 3, 4 + self.source_ball_class]
        ball_predictions = predictions[:, channels, :]

        # ჩვეულებრივი NMS და შემდგომი შეფასება Ultralytics-ს რჩება.
        return super().postprocess(ball_predictions)


def main():
    installed_version = version("ultralytics")

    if installed_version != "8.3.220":
        raise RuntimeError(
            "This validator targets Ultralytics 8.3.220; "
            f"found {installed_version}"
        )

    for path in (DATA, BASELINE, WEIGHTS):
        if not path.is_file():
            raise FileNotFoundError(path)

    val_images = [
        line.strip()
        for line in (DATA.parent / "val.txt").read_text().splitlines()
        if line.strip()
    ]

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "runs" / f"baseline_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    # ორივე მოდელისთვის საერთო შეფასების პარამეტრები.
    common = {
        "data": str(DATA),
        "split": "val",
        "imgsz": IMAGE_SIZE,
        "batch": 2,
        "device": device,
        "workers": 0,
        "half": False,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "classes": None,
        "single_cls": False,
        "agnostic_nms": False,
        "rect": True,
        "augment": False,
        "cache": False,
        "plots": True,
        "save_json": False,
        "save_txt": False,
        "verbose": False,
    }

    report = {
        "ultralytics_version": installed_version,
        "validation_images": len(val_images),
        "settings": common,
        "evaluation_class": {0: "ball"},
        "results": {},
    }

    def save_report():
        (output_dir / "comparison.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(
        f"Device: {device} | "
        f"Validation images: {len(val_images)}"
    )

    for name, weights in (
        ("pretrained", BASELINE),
        ("fine_tuned", WEIGHTS),
    ):
        model = YOLO(str(weights))
        class_id = find_ball_class(model.names)

        print(
            f"\nEvaluating {name}: "
            f"{model.names[class_id]} ({class_id}) -> ball (0)"
        )

        metrics = model.val(
            validator=BallOnlyValidator,
            **common,
            project=str(output_dir),
            name=name,
            exist_ok=False,
        )

        row = {
            "weights": str(weights),
            "source_ball_class_id": class_id,
            "precision_at_best_f1": float(metrics.box.mp),
            "recall_at_best_f1": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        }

        report["results"][name] = row
        save_report()

        print(json.dumps(row, indent=2))

        # შემდეგი მოდელისთვის მეხსიერებას ვათავისუფლებთ.
        del model, metrics
        gc.collect()

        if device == "mps":
            torch.mps.empty_cache()

    baseline = report["results"]["pretrained"]
    trained = report["results"]["fine_tuned"]

    report["improvement_percentage_points"] = {
        "mAP50":
            100 * (trained["mAP50"] - baseline["mAP50"]),
        "mAP50_95":
            100 * (trained["mAP50_95"] - baseline["mAP50_95"]),
    }

    save_report()

    print("\nFinal comparison:")
    print(json.dumps(report, indent=2))
    print(f"Report saved to: {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()