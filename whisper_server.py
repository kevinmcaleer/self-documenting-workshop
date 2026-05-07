"""
Local Whisper transcription server — runs on your Mac.

Receives audio chunks from the Uno Q over HTTP and returns transcribed text.
Uses faster-whisper (CTranslate2) for fast CPU/Metal inference.

Install:
    pip install faster-whisper flask

Run:
    python whisper_server.py

The server listens on port 8178. The Uno Q posts 16-bit PCM WAV chunks to
/transcribe and gets back {"text": "..."}.
"""

from __future__ import annotations

import io
import logging
import tempfile

from flask import Flask, request, jsonify
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("whisper_server")

app = Flask(__name__)

log.info("loading whisper model (base.en) — first run downloads ~150 MB")
model = WhisperModel("base.en", device="cpu", compute_type="int8")
log.info("model ready")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_bytes = request.get_data()
    if not audio_bytes:
        return jsonify({"text": "", "error": "no audio data"}), 400

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        f.write(audio_bytes)
        f.flush()
        segments, _ = model.transcribe(f.name, beam_size=1, language="en")
        text = " ".join(seg.text.strip() for seg in segments)

    log.info("transcribed: %s", text[:80])
    return jsonify({"text": text})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8178)
