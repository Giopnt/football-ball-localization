import json
import os
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent

# ბიბლიოთეკების დამხმარე ფაილები პროექტში შეინახება.
for folder in ("ultralytics", "matplotlib"):
    (ROOT / ".cache" / folder).mkdir(parents=True, exist_ok=True)

os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".cache" / "ultralytics")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

import torch
from ultralytics import YOLO, settings

settings.update({"sync": False, "hub": False})

WEIGHTS = ROOT / "models" / "yolo11n.pt"
DATA = ROOT / "data" / "processed" / "football_ball_v1" / "smoke.yaml"

memory_samples = []


def record_memory(_):
    memory_samples.append(torch.mps.driver_allocated_memory())


if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Run from your VS Code terminal.")

    for path in (WEIGHTS, DATA):
        if not path.is_file():
            raise FileNotFoundError(path)

    model = YOLO(str(WEIGHTS))

    # მეხსიერებას სასწავლო და შეფასების ბაჩების ბოლოს ვკითხულობთ.
    model.add_callback("on_train_batch_end", record_memory)
    model.add_callback("on_val_batch_end", record_memory)

    torch.mps.synchronize()
    started = perf_counter()

    model.train(
        data=str(DATA),
        epochs=1,
        imgsz=1280,
        batch=2,
        device="mps",
        workers=0,
        cache=False,
        amp=False,
        mosaic=0.0,
        scale=0.0,
        plots=False,
        val=True,
        seed=42,
        project=str(ROOT / "runs"),
        name="smoke_mps_1280",
        exist_ok=False,
    )

    torch.mps.synchronize()
    elapsed = perf_counter() - started

    report = {
        "device": "mps",
        "torch_version": torch.__version__,
        "total_seconds": round(elapsed, 2),
        "max_sampled_mps_gib": round(
            max(memory_samples, default=0) / 1024**3, 3
        ),
    }

    save_dir = Path(model.trainer.save_dir)
    (save_dir / "smoke_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSmoke run finished")
    print(json.dumps(report, indent=2))
    print("Results directory:", save_dir)
    print("Best weights:", model.trainer.best)