"""Validate the generated development YOLO label folders."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/traffic_coco")
    args = parser.parse_args()
    root = Path(args.root)
    counts = Counter()
    images = 0
    errors = []
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        for image in sorted(image_dir.glob("*.jpg")):
            images += 1
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                errors.append(f"missing label: {image.name}")
                continue
            for line_no, line in enumerate(label.read_text().splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label}:{line_no}: expected 5 fields")
                    continue
                try:
                    class_id = int(parts[0]); values = [float(x) for x in parts[1:]]
                except ValueError:
                    errors.append(f"{label}:{line_no}: invalid number")
                    continue
                counts[class_id] += 1
                if class_id < 0 or any(value < 0 or value > 1 for value in values):
                    errors.append(f"{label}:{line_no}: out-of-range normalized value")
    print(f"images={images} boxes={sum(counts.values())} class_counts={dict(counts)}")
    if errors:
        print("\n".join(errors[:50]))
        return 1
    print("dataset labels valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
