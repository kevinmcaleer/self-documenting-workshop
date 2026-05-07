"""
Workshop Scribe — passive observer agent for the Arduino Uno Q.

Runs on the MPU (Linux side). Watches the bench via USB camera, listens to
ambient narration via USB mic, logs events. At session end, classifies the
session and produces the appropriate deliverable.

Cost-control principles baked in:
  1. Cheap local "is anything happening" gate before any API call
  2. Sonnet for routine analysis, Opus only for the end-of-session decision
  3. Frame deduplication — don't analyse near-identical frames
  4. ASR transcript is the cheap signal; vision is the expensive one
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import wave
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import cv2
import requests
import sounddevice as sd
from anthropic import Anthropic
from arduino.app_utils import Bridge
from arduino.app_bricks.web_ui import WebUI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SESSION_DIR = Path.home() / "workshop_scribe" / "sessions"
FRAME_INTERVAL_S = 30
MIN_PIXEL_DELTA = 0.04
TIMELAPSE_INTERVAL_S = 10
IDLE_TIMEOUT_S = 15 * 60
WEB_FRAME_INTERVAL_S = 0.5

ANALYSIS_MODEL = "claude-sonnet-4-6"
DECISION_MODEL = "claude-opus-4-7"

WHISPER_URL = os.environ.get("WHISPER_URL", "http://192.168.1.100:8178")
AUDIO_CHUNK_S = 10
AUDIO_SAMPLE_RATE = 16000

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("scribe")

client = Anthropic()

# ---------------------------------------------------------------------------
# Bricks
# ---------------------------------------------------------------------------

bridge = Bridge()
webui = WebUI(assets_dir_path="/app/assets")

# Camera via OpenCV — no Camera brick exists; we grab frames directly.
cap = cv2.VideoCapture(0)


def capture_frame() -> cv2.Mat | None:
    ok, frame = cap.read()
    return frame if ok else None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class Event:
    timestamp: float
    kind: str
    detail: str
    frame_path: str | None = None


@dataclass
class Session:
    started_at: float = field(default_factory=time.time)
    events: list[Event] = field(default_factory=list)
    timelapse_frames: list[str] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)

    @property
    def session_dir(self) -> Path:
        d = SESSION_DIR / datetime.fromtimestamp(self.started_at).strftime("%Y%m%d-%H%M%S")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def log_event(self, kind: str, detail: str, frame_path: str | None = None) -> None:
        ev = Event(time.time(), kind, detail, frame_path)
        self.events.append(ev)
        log.info("event[%s] %s", kind, detail)
        bridge.call("pulse_led", {"kind": kind})


# ---------------------------------------------------------------------------
# App state — tracks whether a session is active
# ---------------------------------------------------------------------------

session: Session | None = None
detector: FrameChangeDetector | None = None
session_tasks: list[asyncio.Task] = []


# ---------------------------------------------------------------------------
# The cheap local gate — runs every loop, no API cost
# ---------------------------------------------------------------------------

class FrameChangeDetector:
    def __init__(self) -> None:
        self._prev: cv2.Mat | None = None

    def is_meaningfully_different(self, frame: cv2.Mat) -> bool:
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self._prev is None:
            self._prev = gray
            return True
        delta = cv2.absdiff(gray, self._prev)
        changed_ratio = (delta > 25).mean()
        if changed_ratio > MIN_PIXEL_DELTA:
            self._prev = gray
            return True
        return False


# ---------------------------------------------------------------------------
# Audio recording + remote Whisper transcription
# ---------------------------------------------------------------------------

def record_chunk() -> bytes:
    """Record AUDIO_CHUNK_S seconds of 16-bit mono PCM, return as WAV bytes."""
    audio = sd.rec(
        int(AUDIO_CHUNK_S * AUDIO_SAMPLE_RATE),
        samplerate=AUDIO_SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def transcribe_chunk(wav_bytes: bytes) -> str:
    """POST a WAV chunk to the Whisper server running on the Mac."""
    try:
        resp = requests.post(
            f"{WHISPER_URL}/transcribe",
            data=wav_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("text", "").strip()
    except requests.RequestException as e:
        log.warning("whisper server error: %s", e)
        return ""


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are observing a maker's workshop bench. Look at this frame and respond with a JSON object — nothing else — with these fields:

{
  "activity": one of [soldering, wiring, screwing, testing, debugging, idle, other],
  "components_visible": list of identifiable electronic components or tools (max 5),
  "notable": short string describing anything worth noting for a build log, or null,
  "is_key_moment": true if this looks like a milestone (first power-on, completed assembly, visible mistake) — used sparingly
}

Be terse. If nothing is happening, say activity is "idle" and notable is null."""


async def analyse_frame(sess: Session, frame_path: Path) -> None:
    img_b64 = base64.standard_b64encode(frame_path.read_bytes()).decode()

    resp = client.messages.create(
        model=ANALYSIS_MODEL,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": img_b64,
                }},
                {"type": "text", "text": ANALYSIS_PROMPT},
            ],
        }],
    )

    try:
        data = json.loads(resp.content[0].text)
    except (json.JSONDecodeError, IndexError):
        log.warning("could not parse analysis response")
        return

    if data.get("activity") and data["activity"] != "idle":
        sess.log_event("activity", data["activity"], str(frame_path))
    for comp in data.get("components_visible") or []:
        sess.log_event("component", comp, str(frame_path))
    if data.get("notable"):
        sess.log_event("notable", data["notable"], str(frame_path))
    if data.get("is_key_moment"):
        sess.log_event("key_moment", data.get("notable") or "milestone", str(frame_path))


# ---------------------------------------------------------------------------
# Capture loops
# ---------------------------------------------------------------------------

async def analysis_loop(sess: Session, det: FrameChangeDetector) -> None:
    while True:
        await asyncio.sleep(FRAME_INTERVAL_S)
        frame = capture_frame()
        if frame is None or not det.is_meaningfully_different(frame):
            continue
        path = sess.session_dir / f"analysis_{int(time.time())}.jpg"
        cv2.imwrite(str(path), frame)
        await analyse_frame(sess, path)


async def timelapse_loop(sess: Session) -> None:
    while True:
        await asyncio.sleep(TIMELAPSE_INTERVAL_S)
        frame = capture_frame()
        if frame is None:
            continue
        path = sess.session_dir / f"tl_{int(time.time())}.jpg"
        cv2.imwrite(str(path), frame)
        sess.timelapse_frames.append(str(path))


async def transcript_loop(sess: Session) -> None:
    """Record audio in chunks, send to Mac's Whisper server, log results."""
    loop = asyncio.get_event_loop()
    while True:
        text = await loop.run_in_executor(None, lambda: transcribe_chunk(record_chunk()))
        if text:
            sess.transcript.append(text)
            sess.log_event("narration", text)
            webui.send_message("transcript", {
                "text": text,
                "time": datetime.now().strftime("%H:%M:%S"),
            })


async def mcu_event_loop(sess: Session) -> None:
    async for ev in bridge.subscribe("mcu_events"):
        if ev["kind"] == "marker_pressed":
            sess.log_event("marker", "user marked this moment as important")
        elif ev["kind"] == "iron_on":
            sess.log_event("activity", "soldering iron powered on")
        elif ev["kind"] == "session_end":
            await end_session()


async def web_frame_loop() -> None:
    while True:
        await asyncio.sleep(WEB_FRAME_INTERVAL_S)
        frame = capture_frame()
        if frame is None:
            continue
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        webui.send_message("frame", {
            "jpeg": base64.b64encode(jpeg.tobytes()).decode(),
        })


async def status_broadcast_loop() -> None:
    while True:
        await asyncio.sleep(1)
        if session is not None:
            webui.send_message("status", {
                "state": "recording",
                "duration": int(time.time() - session.started_at),
                "events": len(session.events),
            })
        else:
            webui.send_message("status", {"state": "idle", "duration": 0, "events": 0})


# ---------------------------------------------------------------------------
# End-of-session classification + output generation
# ---------------------------------------------------------------------------

CLASSIFICATION_PROMPT = """You watched a maker's workshop session. Below is the full event log and transcript of what was said.

Decide which deliverable best fits this session. Choose ONE:
  - "tutorial": linear progression, clear narration, suitable as a teaching post
  - "build_log": iterative work with debugging, suitable as a development diary
  - "video_script": strong narrative arc with a finished artefact, suitable for a YouTube video

Then produce that deliverable. For tutorial: clean blog post with steps. For build_log: chronological notes with decisions and lessons. For video_script: md2script format with scene headings and dialogue.

Respond with JSON: {"choice": "<one of the above>", "reasoning": "<one sentence>", "output": "<the full deliverable in markdown>"}"""


async def classify_and_generate(sess: Session) -> dict:
    digest = {
        "duration_minutes": int((time.time() - sess.started_at) / 60),
        "events": [asdict(e) for e in sess.events],
        "transcript": " ".join(sess.transcript),
        "key_moment_frames": [
            e.frame_path for e in sess.events if e.kind == "key_moment"
        ],
    }

    resp = client.messages.create(
        model=DECISION_MODEL,
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": CLASSIFICATION_PROMPT + "\n\n" + json.dumps(digest, indent=2),
        }],
    )
    return json.loads(resp.content[0].text)


def write_deliverable(sess: Session, result: dict) -> Path:
    out = sess.session_dir / f"{result['choice']}.md"
    out.write_text(result["output"])
    log.info("wrote %s — choice was '%s' because: %s",
             out, result["choice"], result["reasoning"])
    return out


# ---------------------------------------------------------------------------
# Session lifecycle (called from web dashboard and MCU button)
# ---------------------------------------------------------------------------

async def start_session():
    global session, detector, session_tasks
    if session is not None:
        return

    session = Session()
    detector = FrameChangeDetector()
    log.info("session started — session_dir=%s", session.session_dir)
    bridge.call("set_status", {"state": "watching"})

    session_tasks = [
        asyncio.create_task(analysis_loop(session, detector)),
        asyncio.create_task(timelapse_loop(session)),
        asyncio.create_task(transcript_loop(session)),
        asyncio.create_task(mcu_event_loop(session)),
    ]


async def end_session():
    global session, session_tasks
    if session is None:
        return

    log.info("session ending — generating deliverable")
    for t in session_tasks:
        t.cancel()
    session_tasks = []

    bridge.call("set_status", {"state": "thinking"})
    webui.send_message("status", {"state": "thinking", "duration": 0, "events": 0})

    result = await classify_and_generate(session)
    path = write_deliverable(session, result)

    bridge.call("set_status", {"state": "done"})
    webui.send_message("status", {"state": "done", "duration": 0, "events": 0})
    log.info("deliverable: %s", path)

    session = None


# ---------------------------------------------------------------------------
# Web API handlers (called by the WebUI brick)
# ---------------------------------------------------------------------------

async def api_start_session():
    await start_session()
    return {"status": "started"}


async def api_stop_session():
    await end_session()
    return {"status": "stopping"}


async def api_session_status():
    if session is None:
        return {"active": False, "duration": 0, "events": 0, "transcript_lines": 0}
    return {
        "active": True,
        "duration": int(time.time() - session.started_at),
        "events": len(session.events),
        "transcript_lines": len(session.transcript),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    webui.expose_api("POST", "/api/session/start", api_start_session)
    webui.expose_api("POST", "/api/session/stop", api_stop_session)
    webui.expose_api("GET", "/api/session/status", api_session_status)
    webui.start()
    log.info("dashboard at %s", webui.local_url)

    background_tasks = [
        asyncio.create_task(web_frame_loop()),
        asyncio.create_task(status_broadcast_loop()),
    ]

    try:
        await asyncio.gather(*background_tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        if session is not None:
            await end_session()
        webui.stop()


if __name__ == "__main__":
    asyncio.run(main())
