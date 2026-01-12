from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from PIL import Image


def normalize_label(text: str) -> str:
    normalized = " ".join(text.strip().split())
    return normalized.upper()


def _load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_dataset(
    annotations_dir: Path,
    datasets_dir: Path,
    charset: str,
    seed: int = 42,
) -> Dict:
    crops_dir = datasets_dir / "crops"
    train_dir = datasets_dir / "train"
    val_dir = datasets_dir / "val"
    crops_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    allowed = set(charset)
    all_records: List[Dict] = []
    rejected: List[Dict] = []

    for annotation_file in annotations_dir.glob("*.jsonl"):
        records = _load_jsonl(annotation_file)
        for record in records:
            normalized = normalize_label(record.get("text", ""))
            if not normalized or any(char not in allowed for char in normalized):
                rejected.append({**record, "normalized": normalized})
                continue
            record["normalized"] = normalized
            all_records.append(record)

    random.seed(seed)
    random.shuffle(all_records)

    split_index = int(len(all_records) * 0.9)
    train_records = all_records[:split_index]
    val_records = all_records[split_index:]

    def _save_split(records: List[Dict], split_dir: Path) -> List[Dict]:
        output_records: List[Dict] = []
        for idx, record in enumerate(records):
            image_path = Path(record["image"]).resolve()
            bbox = record["bbox"]
            if not image_path.exists():
                continue
            with Image.open(image_path) as image:
                cropped = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                crop_name = f"{image_path.stem}_{idx}.png"
                crop_path = crops_dir / crop_name
                cropped.save(crop_path)
            output = {
                "image": str(crop_path),
                "text": record["normalized"],
            }
            output_records.append(output)

        jsonl_path = split_dir / "annotations.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for item in output_records:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        return output_records

    train_output = _save_split(train_records, train_dir)
    val_output = _save_split(val_records, val_dir)

    return {
        "total": len(all_records),
        "train": len(train_output),
        "val": len(val_output),
        "rejected": rejected,
        "charset": charset,
    }
