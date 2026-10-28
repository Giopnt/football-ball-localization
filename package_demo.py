import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "dist" / "football-ball-demo-v4"

MODEL = Path("runs/ball_yolo11n_1280_v1/weights/best.pt")

FILES = [
    Path("app.py"),
    Path("ball_pipeline.py"),
    Path("half_localization.py"),
    Path("visualize_prediction.py"),
    Path("calibrate_pitch.py"),
    Path("check_environment.py"),
    Path("check_torch.py"),
    Path("requirements.txt"),
    Path("requirements-macos.lock.txt"),
    Path(".python-version"),
    Path("README.md"),
    Path("model_sources.json"),
    MODEL,
]

FOLDERS = [
    Path("data/samples"),
    Path("data/calibrations"),
    Path("reports"),
    Path("docs"),
]


def relative_metadata(value):
    """Keep checkpoint metadata paths relative to the project."""
    if isinstance(value, dict):
        return {key: relative_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relative_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(relative_metadata(item) for item in value)
    if isinstance(value, str):
        if value == str(ROOT):
            return "."
        prefix = str(ROOT) + "/"
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def export_checkpoint(source, target):
    """Export this project's trusted checkpoint without its local path metadata."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model = checkpoint["model"]
    expected = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    checkpoint = relative_metadata(checkpoint)
    if hasattr(model, "args"):
        model.args = relative_metadata(model.args)
    torch.save(checkpoint, target)

    restored = torch.load(target, map_location="cpu", weights_only=False)
    actual = restored["model"].state_dict()
    if expected.keys() != actual.keys() or any(
        expected[name].dtype != actual[name].dtype
        or expected[name].shape != actual[name].shape
        or not torch.equal(expected[name], actual[name])
        for name in expected
    ):
        raise RuntimeError("Export changed model tensor values.")
    if model.names != restored["model"].names:
        raise RuntimeError("Export changed model classes.")
    return len(expected)


def main():
    # შეფუთვამდე ყველა საჭირო წყაროს არსებობას ვამოწმებთ.
    for relative_path in FILES:
        if not (ROOT / relative_path).is_file():
            raise FileNotFoundError(ROOT / relative_path)

    for relative_path in FOLDERS:
        if not (ROOT / relative_path).is_dir():
            raise FileNotFoundError(ROOT / relative_path)

    if DEST.exists() or DEST.with_suffix(".zip").exists():
        raise FileExistsError(
            "Demo package already exists. "
            "Choose a new folder name in DEST before rebuilding."
        )

    DEST.mkdir(parents=True)

    # ფაილებს იგივე შიდა მისამართებზე ვაკოპირებთ.
    for relative_path in FILES:
        target = DEST / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative_path != MODEL:
            shutil.copy2(ROOT / relative_path, target)

    for relative_path in FOLDERS:
        shutil.copytree(
            ROOT / relative_path,
            DEST / relative_path,
            ignore=shutil.ignore_patterns(".DS_Store", "__pycache__"),
        )

    for path in DEST.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".yaml", ".yml", ".md", ".txt"}:
            contents = path.read_text(encoding="utf-8")
            cleaned = contents.replace(str(ROOT) + "/", "")
            if cleaned != contents:
                path.write_text(cleaned, encoding="utf-8")

    tensor_count = export_checkpoint(ROOT / MODEL, DEST / MODEL)
    source_hash = hashlib.sha256(
        (ROOT / MODEL).read_bytes()
    ).hexdigest()

    copied_hash = hashlib.sha256(
        (DEST / MODEL).read_bytes()
    ).hexdigest()

    model_info = {
        "file": MODEL.as_posix(),
        "kind": "fine_tuned",
        "classes": {"0": "ball"},
        "size_bytes": (DEST / MODEL).stat().st_size,
        "sha256": copied_hash,
        "source_sha256": source_hash,
        "metadata_paths": "project_relative",
        "verified_model_tensors": tensor_count,
        "tensor_values_match_source": True,
    }

    (DEST / "model_manifest.json").write_text(
        json.dumps(model_info, indent=2),
        encoding="utf-8",
    )

    archive = shutil.make_archive(
        str(DEST),
        "zip",
        root_dir=str(DEST.parent),
        base_dir=DEST.name,
    )

    print(f"Model tensors: matched ({tensor_count})")
    print(f"Export model SHA-256: {copied_hash}")
    print("Demo folder:", DEST)
    print("ZIP archive:", archive)


if __name__ == "__main__":
    main()
