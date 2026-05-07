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
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import cv2  # for the local "did anything change" gate
from anthropic import Anthropic

# --- App Lab Bricks (imported by the App Lab runtime; stubbed for clarity) ---
# In a real App Lab project these come from the Bricks system.
from bricks import camera, asr_cloud, bridge   # noqa: F401

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SESSION_DIR = Path.home() / "workshop_scribe" / "sessions"
FRAME_INTERVAL_S = 30          # how often to *consider* a frame
MIN_PIXEL_DELTA = 0.04         # skip frame if <4% changed since last analysed
TIMELAPSE_INTERVAL_S = 10      # frames captured for timelapse (no analysis)
IDLE_TIMEOUT_S = 15 * 60       # session ends after this much silence + stillness

ANALYSIS_MODEL = "claude-sonnet-4-6"
DECISION_MODEL = "claude-opus-4-7"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("scribe")

client = Anthropic()  # reads ANTHROPIC_API_KEY from env


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class Event:
    timestamp: float
    kind: str           # "activity" | "component" | "narration" | "marker"
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
        # Pulse the LED matrix on the MCU side so it's visible on camera
        bridge.call("pulse_led", {"kind": kind})


# ---------------------------------------------------------------------------
# The cheap local gate — runs every loop, no API cost
# ---------------------------------------------------------------------------

class FrameChangeDetector:
    """Avoid sending near-identical frames to the API."""

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


async def analyse_frame(session: Session, frame_path: Path) -> None:
    """Send one frame to Sonnet for structured analysis."""
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
        session.log_event("activity", data["activity"], str(frame_path))
    for comp in data.get("components_visible") or []:
        session.log_event("component", comp, str(frame_path))
    if data.get("notable"):
        session.log_event("notable", data["notable"], str(frame_path))
    if data.get("is_key_moment"):
        session.log_event("key_moment", data.get("notable") or "milestone", str(frame_path))


# ---------------------------------------------------------------------------
# Capture loops — one slow loop for analysis, one fast loop for timelapse
# ---------------------------------------------------------------------------

async def analysis_loop(session: Session, detector: FrameChangeDetector) -> None:
    while True:
        await asyncio.sleep(FRAME_INTERVAL_S)
        frame = camera.capture()  # via App Lab Camera Brick
        if not detector.is_meaningfully_different(frame):
            continue
        path = session.session_dir / f"analysis_{int(time.time())}.jpg"
        cv2.imwrite(str(path), frame)
        await analyse_frame(session, path)


async def timelapse_loop(session: Session) -> None:
    while True:
        await asyncio.sleep(TIMELAPSE_INTERVAL_S)
        frame = camera.capture()
        path = session.session_dir / f"tl_{int(time.time())}.jpg"
        cv2.imwrite(str(path), frame)
        session.timelapse_frames.append(str(path))


async def transcript_loop(session: Session) -> None:
    """Stream from the ASR Cloud Brick. Ambient narration is cheap and rich signal."""
    async for chunk in asr_cloud.stream():
        if chunk.text.strip():
            session.transcript.append(chunk.text)
            session.log_event("narration", chunk.text)


async def mcu_event_loop(session: Session) -> None:
    """Listen for MCU events: button-press markers, soldering iron on/off."""
    async for ev in bridge.subscribe("mcu_events"):
        if ev["kind"] == "marker_pressed":
            session.log_event("marker", "user marked this moment as important")
        elif ev["kind"] == "iron_on":
            session.log_event("activity", "soldering iron powered on")


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


async def classify_and_generate(session: Session) -> dict:
    """End-of-session: Opus reads everything and decides what to produce."""
    digest = {
        "duration_minutes": int((time.time() - session.started_at) / 60),
        "events": [asdict(e) for e in session.events],
        "transcript": " ".join(session.transcript),
        "key_moment_frames": [
            e.frame_path for e in session.events if e.kind == "key_moment"
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


def write_deliverable(session: Session, result: dict) -> Path:
    out = session.session_dir / f"{result['choice']}.md"
    out.write_text(result["output"])
    log.info("wrote %s — choice was '%s' because: %s",
             out, result["choice"], result["reasoning"])
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    session = Session()
    detector = FrameChangeDetector()
    log.info("session started — session_dir=%s", session.session_dir)
    bridge.call("set_status", {"state": "watching"})

    tasks = [
        asyncio.create_task(analysis_loop(session, detector)),
        asyncio.create_task(timelapse_loop(session)),
        asyncio.create_task(transcript_loop(session)),
        asyncio.create_task(mcu_event_loop(session)),
    ]

    try:
        # In real use: detect end-of-session via idle timeout, button hold, or
        # voice command "scribe, wrap up". Stubbed as run-until-cancelled.
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("session ending — generating deliverable")
        for t in tasks:
            t.cancel()
        bridge.call("set_status", {"state": "thinking"})
        result = await classify_and_generate(session)
        path = write_deliverable(session, result)
        bridge.call("set_status", {"state": "done"})
        log.info("deliverable: %s", path)


if __name__ == "__main__":
    asyncio.run(main())

