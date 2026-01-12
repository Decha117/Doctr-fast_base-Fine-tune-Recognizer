# docTR Recognizer Studio (fast_base)

Local web application for annotating bounding boxes, building recognizer datasets, and fine-tuning docTR fast_base recognizer with GPU acceleration.

## Features
- Multi-image upload and annotation with bbox + text labels.
- JSONL annotations saved per image.
- Dataset builder with deterministic train/val split, label normalization, and charset validation.
- Training dashboard with SSE progress + system monitor.
- Inference page to compare base vs fine-tuned recognizer outputs.
- Export bundle with best checkpoint, config, and sample inference script.

## Project structure
```
app.py
requirements.txt
README.md
templates/
static/
uploads/
annotations/
datasets/
outputs/
utilities/
```

## Run locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   python app.py
   ```
3. Open http://localhost:5000

## Notes
- Training currently runs a simulated loop to stream metrics/logs and save checkpoints. Replace the training logic in `utilities/trainer.py` with your full docTR training pipeline if needed.
- Default charset: `0123456789/:ABCDEFGHIJKLMNOPQRSTUVWXYZ `
- Dataset crops are stored in `datasets/crops/` and splits in `datasets/train` + `datasets/val`.

## API endpoints
- `POST /upload` — upload images
- `GET /annotate/<filename>` — annotation UI
- `POST /build_dataset` — build dataset
- `POST /train` — start training
- `GET /events` — SSE progress
- `GET /status` — system monitor
- `POST /api/inference` — run inference
- `GET /download_model` — download export bundle
