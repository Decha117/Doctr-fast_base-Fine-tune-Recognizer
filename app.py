from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from utilities.dataset_builder import build_dataset
from utilities.inference import run_inference
from utilities.monitor import get_system_snapshot
from utilities.trainer import TrainingManager

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUTS_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
ANNOTATIONS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

DEFAULT_CHARSET = "0123456789/:ABCDEFGHIJKLMNOPQRSTUVWXYZ "

training_events: "Queue[dict]" = Queue()
training_manager = TrainingManager(
    outputs_dir=OUTPUTS_DIR,
    events_queue=training_events,
    charset=DEFAULT_CHARSET,
)


def _annotation_path(filename: str) -> Path:
    safe_name = secure_filename(filename)
    return ANNOTATIONS_DIR / f"{safe_name}.jsonl"


def _load_annotations(filename: str) -> List[Dict[str, Any]]:
    path = _annotation_path(filename)
    if not path.exists():
        return []
    annotations: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            annotations.append(json.loads(line))
    return annotations


def _save_annotations(filename: str, annotations: List[Dict[str, Any]]) -> None:
    path = _annotation_path(filename)
    with path.open("w", encoding="utf-8") as handle:
        for item in annotations:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


@app.route("/")
def index() -> str:
    uploads = sorted([p.name for p in UPLOAD_DIR.iterdir() if p.is_file()])
    return render_template("upload.html", uploads=uploads)


@app.route("/upload", methods=["POST"])
def upload() -> Response:
    files = request.files.getlist("files")
    for file in files:
        if not file.filename:
            continue
        filename = secure_filename(file.filename)
        file.save(UPLOAD_DIR / filename)
    return redirect(url_for("index"))


@app.route("/annotate/<filename>")
def annotate(filename: str) -> str:
    safe_name = secure_filename(filename)
    return render_template(
        "annotate.html",
        filename=safe_name,
        image_url=url_for("uploaded_file", filename=safe_name),
    )


@app.route("/api/annotations/<filename>", methods=["GET", "POST"])
def annotations(filename: str) -> Response:
    safe_name = secure_filename(filename)
    if request.method == "GET":
        return jsonify(_load_annotations(safe_name))

    payload = request.get_json(force=True)
    annotation = {
        "image": f"uploads/{safe_name}",
        "bbox": payload.get("bbox"),
        "text": payload.get("text", ""),
    }
    annotations_list = _load_annotations(safe_name)
    annotations_list.append(annotation)
    _save_annotations(safe_name, annotations_list)
    return jsonify({"status": "ok", "count": len(annotations_list)})


@app.route("/api/annotations/<filename>/<int:index>", methods=["PUT", "DELETE"])
def update_annotation(filename: str, index: int) -> Response:
    safe_name = secure_filename(filename)
    annotations_list = _load_annotations(safe_name)
    if index < 0 or index >= len(annotations_list):
        return jsonify({"error": "Index out of range"}), 404

    if request.method == "DELETE":
        annotations_list.pop(index)
    else:
        payload = request.get_json(force=True)
        annotations_list[index]["bbox"] = payload.get("bbox", annotations_list[index]["bbox"])
        annotations_list[index]["text"] = payload.get("text", annotations_list[index]["text"])
    _save_annotations(safe_name, annotations_list)
    return jsonify({"status": "ok", "count": len(annotations_list)})


@app.route("/build_dataset", methods=["POST"])
def build_dataset_endpoint() -> Response:
    charset = request.form.get("charset", DEFAULT_CHARSET)
    summary = build_dataset(
        annotations_dir=ANNOTATIONS_DIR,
        datasets_dir=DATASETS_DIR,
        charset=charset,
        seed=42,
    )
    return jsonify(summary)


@app.route("/train", methods=["POST"])
def start_training() -> Response:
    if training_manager.is_running:
        return jsonify({"status": "already_running"}), 409

    payload = request.get_json(force=True)
    training_manager.start(
        datasets_dir=DATASETS_DIR,
        params=payload,
    )
    return jsonify({"status": "started"})


@app.route("/events")
def events() -> Response:
    def stream() -> Any:
        while True:
            data = training_events.get()
            yield f"data: {json.dumps(data)}\n\n"

    return Response(stream(), mimetype="text/event-stream")


@app.route("/status")
def status() -> Response:
    return jsonify(get_system_snapshot())


@app.route("/inference")
def inference_page() -> str:
    uploads = sorted([p.name for p in UPLOAD_DIR.iterdir() if p.is_file()])
    return render_template("inference.html", uploads=uploads)


@app.route("/api/inference", methods=["POST"])
def inference_api() -> Response:
    payload = request.get_json(force=True)
    filename = secure_filename(payload.get("filename", ""))
    bbox = payload.get("bbox")
    if not filename:
        return jsonify({"error": "Missing filename"}), 400

    image_path = UPLOAD_DIR / filename
    result = run_inference(
        image_path=image_path,
        bbox=bbox,
        outputs_dir=OUTPUTS_DIR,
    )
    return jsonify(result)


@app.route("/uploads/<filename>")
def uploaded_file(filename: str) -> Response:
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/download_model")
def download_model() -> Response:
    zip_path = OUTPUTS_DIR / "export" / "model_bundle.zip"
    if not zip_path.exists():
        return jsonify({"error": "Bundle not found. Train first."}), 404
    return send_from_directory(zip_path.parent, zip_path.name, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
