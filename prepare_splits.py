import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "football_ball_v1"
OUTPUT = ROOT / "data" / "processed" / "football_ball_v1"

# ეს კოდები ფოტოების სახელებში მოცემული საწყისი კლიპების ID-ებია.
GROUPS = {
    "train": {"08fd33", "2e57b9", "40cd38", "42ba34", "54745b", "573e61"},
    "val": {"0a2d9b", "4b770a", "798b45"},
    "test": {"0bfacc"},
}

images = sorted(
    path.resolve()
    for old_split in ("train", "valid", "test")
    for path in (RAW / old_split / "images").glob("*.jpg")
)

if len(images) != 903:
    raise ValueError(f"Expected 903 images, found {len(images)}")

known_groups = set().union(*GROUPS.values())
found_groups = {path.name.split("_")[0] for path in images}

if found_groups != known_groups:
    raise ValueError("Dataset clip IDs differ from the expected IDs.")

group_list = [
    group
    for groups in GROUPS.values()
    for group in groups
]
if len(group_list) != len(set(group_list)):
    raise ValueError("A clip was assigned to more than one split.")

splits = {
    split: [
        path for path in images
        if path.name.split("_")[0] in groups
    ]
    for split, groups in GROUPS.items()
}

OUTPUT.mkdir(parents=True, exist_ok=True)


def write_image_list(filename, paths):
    content = "\n".join(str(path) for path in paths) + "\n"
    (OUTPUT / filename).write_text(content, encoding="utf-8")


for split, paths in splits.items():
    write_image_list(f"{split}.txt", paths)
    print(f"{split}: {len(paths)} images")

# პირველი მოკლე სასწავლო გაშვებისთვის პატარა ნაკრები.
for split, count in (("train", 64), ("val", 16)):
    selected = random.Random(42).sample(splits[split], count)
    write_image_list(f"smoke_{split}.txt", selected)
    print(f"smoke_{split}: {len(selected)} images")

config = {
    "path": str(OUTPUT),
    "train": "train.txt",
    "val": "val.txt",
    "test": "test.txt",
    "names": {0: "ball"},
}

smoke_config = {
    "path": str(OUTPUT),
    "train": "smoke_train.txt",
    "val": "smoke_val.txt",
    "names": {0: "ball"},
}

for filename, data in (
    ("data.yaml", config),
    ("smoke.yaml", smoke_config),
):
    (OUTPUT / filename).write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )

manifest = {
    "strategy": "Keep each source clip in exactly one split",
    "groups": {name: sorted(ids) for name, ids in GROUPS.items()},
    "counts": {name: len(paths) for name, paths in splits.items()},
    "smoke_seed": 42,
}

(OUTPUT / "split_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("Saved:", OUTPUT)