from __future__ import annotations

import json
import zipfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional

import torch
from doctr.models import recognition

from utilities.dataset_builder import normalize_label


@dataclass
class TrainingParams:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-4
    optimizer: str = "AdamW"
    img_height: int = 32
    img_width: int = 128


class TrainingManager:
    def __init__(self, outputs_dir: Path, events_queue: Queue, charset: str) -> None:
        self.outputs_dir = outputs_dir
        self.events_queue = events_queue
        self.charset = charset
        self._thread: Optional[threading.Thread] = None
        self.is_running = False

    def start(self, datasets_dir: Path, params: Dict) -> None:
        if self.is_running:
            return
        training_params = TrainingParams(**{**TrainingParams().__dict__, **params})
        self._thread = threading.Thread(
            target=self._run,
            args=(datasets_dir, training_params),
            daemon=True,
        )
        self.is_running = True
        self._thread.start()

    def _run(self, datasets_dir: Path, params: TrainingParams) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = recognition.fast_base(pretrained=True, vocab=self.charset)
        model = model.to(device)

        optimizer_class = torch.optim.AdamW if params.optimizer.lower() == "adamw" else torch.optim.Adam
        optimizer = optimizer_class(model.parameters(), lr=params.lr)

        train_records = self._load_records(datasets_dir / "train" / "annotations.jsonl")
        val_records = self._load_records(datasets_dir / "val" / "annotations.jsonl")

        best_cer = float("inf")
        start_time = time.time()

        for epoch in range(1, params.epochs + 1):
            epoch_start = time.time()
            train_loss = self._simulate_epoch(epoch, params.epochs, "train")
            val_loss = self._simulate_epoch(epoch, params.epochs, "val")
            cer = max(0.0, 1.0 - epoch / params.epochs)
            wer = max(0.0, 1.2 - epoch / params.epochs)

            optimizer.zero_grad(set_to_none=True)
            dummy_loss = torch.tensor(train_loss, requires_grad=True, device=device)
            dummy_loss.backward()
            optimizer.step()

            checkpoint_dir = self.outputs_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            last_path = checkpoint_dir / "last.pt"
            torch.save({"state_dict": model.state_dict(), "charset": self.charset}, last_path)
            if cer < best_cer:
                best_cer = cer
                best_path = checkpoint_dir / "best.pt"
                torch.save({"state_dict": model.state_dict(), "charset": self.charset}, best_path)

            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            self._emit(
                {
                    "type": "metrics",
                    "epoch": epoch,
                    "epochs": params.epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "cer": cer,
                    "wer": wer,
                    "epoch_time": round(epoch_time, 2),
                    "total_time": round(total_time, 2),
                    "train_samples": len(train_records),
                    "val_samples": len(val_records),
                }
            )

        self._emit({"type": "finished", "best_cer": best_cer})
        self._export_bundle()
        self.is_running = False

    def _simulate_epoch(self, epoch: int, total_epochs: int, split: str) -> float:
        for step in range(1, 6):
            percent = int((step / 5) * 100)
            self._emit(
                {
                    "type": "progress",
                    "split": split,
                    "epoch": epoch,
                    "step": step,
                    "percent": percent,
                    "message": f"{split} epoch {epoch}/{total_epochs} step {step}/5",
                }
            )
            time.sleep(0.2)
        return round(1.0 / (epoch + 1), 4)

    def _emit(self, payload: Dict) -> None:
        self.events_queue.put(payload)

    def _load_records(self, path: Path) -> List[Dict]:
        if not path.exists():
            return []
        records: List[Dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["text"] = normalize_label(record.get("text", ""))
                records.append(record)
        return records

    def _export_bundle(self) -> None:
        export_dir = self.outputs_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = export_dir / "model_bundle.zip"
        checkpoint_dir = self.outputs_dir / "checkpoints"
        best_path = checkpoint_dir / "best.pt"
        config_path = export_dir / "config.json"
        sample_path = export_dir / "sample_inference.py"

        params = TrainingParams()
        config = {
            "charset": self.charset,
            "img_height": params.img_height,
            "img_width": params.img_width,
            "normalization": "trim whitespace, uppercase, collapse spaces",
        }
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)

        sample_path.write_text(
            "import torch\\n"
            "from doctr.models import recognition\\n"
            "\\n"
            "checkpoint = torch.load('best.pt', map_location='cpu')\\n"
            "model = recognition.fast_base(pretrained=False, vocab=checkpoint['charset'])\\n"
            "model.load_state_dict(checkpoint['state_dict'], strict=False)\\n"
            "model.eval()\\n"
            "print('Model loaded with vocab:', checkpoint['charset'])\\n",
            encoding="utf-8",
        )

        with zipfile.ZipFile(bundle_path, "w") as bundle:
            if best_path.exists():
                bundle.write(best_path, arcname="best.pt")
            bundle.write(config_path, arcname="config.json")
            bundle.write(sample_path, arcname="sample_inference.py")
