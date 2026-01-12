from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
import torch
from doctr.models import recognition


def _load_model(checkpoint_path: Optional[Path], vocab: Optional[str] = None):
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        vocab = checkpoint.get("charset", vocab)
        model = recognition.fast_base(pretrained=False, vocab=vocab)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        return model
    return recognition.fast_base(pretrained=True, vocab=vocab)


def _fake_decode() -> str:
    return "PREDICTION_PLACEHOLDER"


def run_inference(image_path: Path, bbox: Optional[List[int]], outputs_dir: Path) -> Dict:
    if not image_path.exists():
        return {"error": "Image not found"}

    with Image.open(image_path) as image:
        if bbox:
            crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        else:
            crop = image
        crop = crop.convert("RGB")

    checkpoint_dir = outputs_dir / "checkpoints"
    best_path = checkpoint_dir / "best.pt"

    base_model = _load_model(None)
    tuned_model = _load_model(best_path)

    base_model.eval()
    tuned_model.eval()

    return {
        "base": _fake_decode(),
        "tuned": _fake_decode(),
    }
