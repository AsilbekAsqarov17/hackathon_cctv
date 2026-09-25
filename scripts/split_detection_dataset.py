"""Split development frames into temporal train/validation folders."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/traffic_coco")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    root = Path(args.root)
    images = sorted((root / "images" / "development").glob("*.jpg"))
    labels = root / "labels" / "development"
    split_at = max(1, int(len(images) * (1.0 - args.val_fraction)))
    groups = {"train": images[:split_at], "val": images[split_at:]}
    for group, files in groups.items():
        image_out = root / "images" / group
        label_out = root / "labels" / group
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for image in files:
            shutil.copy2(image, image_out / image.name)
            label = labels / f"{image.stem}.txt"
            if label.exists():
                shutil.copy2(label, label_out / label.name)
            else:
                (label_out / f"{image.stem}.txt").write_text("", encoding="utf-8")
    yaml = """path: .\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: bicycle\n  2: car\n  3: motorcycle\n  4: bus\n  5: truck\n  6: traffic light\n  7: stop sign\n"""
    (root / "dataset.yaml").write_text(yaml, encoding="utf-8")
    print(f"train={len(groups['train'])} val={len(groups['val'])} yaml={root / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
