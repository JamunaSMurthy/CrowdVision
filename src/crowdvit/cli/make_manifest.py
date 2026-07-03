"""Build a two-column CSV manifest (video_path,label_index) plus a class-name
JSON map from a folder of ``<class_name>/<video_file>`` videos.

  python scripts/make_manifest.py --root /data/kinetics400/train \
      --out-csv data/manifests/kinetics400_train.csv \
      --out-classmap data/manifests/kinetics400_classes.json

For val/test splits, reuse the training class map so label indices line up:

  python scripts/make_manifest.py --root /data/kinetics400/val \
      --out-csv data/manifests/kinetics400_val.csv \
      --out-classmap data/manifests/kinetics400_classes.json \
      --existing-classmap data/manifests/kinetics400_classes.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Folder of <class_name>/<video> subfolders")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-classmap", required=True)
    parser.add_argument(
        "--existing-classmap",
        default=None,
        help="Reuse an existing class map instead of building a new one from --root",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    if args.existing_classmap:
        with open(args.existing_classmap) as f:
            class_to_idx = json.load(f)
    else:
        class_to_idx = {p.name: i for i, p in enumerate(class_dirs)}

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    num_rows = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for class_dir in class_dirs:
            if class_dir.name not in class_to_idx:
                continue
            label = class_to_idx[class_dir.name]
            for video_path in sorted(class_dir.iterdir()):
                if video_path.suffix.lower() in VIDEO_EXTENSIONS:
                    writer.writerow([str(video_path.resolve()), label])
                    num_rows += 1

    if not args.existing_classmap:
        Path(args.out_classmap).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_classmap, "w") as f:
            json.dump(class_to_idx, f, indent=2)

    print(f"Wrote {num_rows} rows to {out_csv}")


if __name__ == "__main__":
    main()
