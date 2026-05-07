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

Architecture note: this file is a single @brick class plus a WebUI brick.
App.run() drives the lifecycle — it auto-discovers @brick.loop methods and
runs each in its own daemon thread. We do not run our own asyncio loop;
asyncio is what was breaking before (the WebUI brick's uvicorn server
expects to be driven by App, not co-located in a foreign event loop).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import threading
import time
import uuid
import wave
import zipfile
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import sounddevice as sd
from anthropic import Anthropic
from arduino.app_utils import App, Bridge, Logger, brick
from arduino.app_bricks.web_ui import WebUI
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# JSON-body schemas for settings endpoints. FastAPI infers query params for
# bare scalars; using Pydantic models guarantees the value comes from the
# request body, which matches what the dashboard's fetch() sends.
# ---------------------------------------------------------------------------

class RotationRequest(BaseModel):
    rotation: int


class ThresholdRequest(BaseModel):
    threshold: float


class TranscriptionEnabledRequest(BaseModel):
    enabled: bool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Sessions live under the project dir (bind-mounted to /app inside the
# container) so they survive container destruction, are visible from the
# host for SCP / direct editing, and can be downloaded straight from the
# dashboard. The fallback to ~/workshop_scribe is for running outside the
# container during development.
APP_DIR = Path("/app") if Path("/app").exists() else Path.home() / "workshop_scribe"
SESSION_DIR = APP_DIR / "sessions"
CONFIG_PATH = APP_DIR / ".scribe-config.json"  # rotation, rms threshold, etc.
FRAME_INTERVAL_S = 30
MIN_PIXEL_DELTA = 0.04
TIMELAPSE_INTERVAL_S = 10
WEB_FRAME_INTERVAL_S = 0.5
STATUS_BROADCAST_INTERVAL_S = 1.0
CAMERA_RETRY_INTERVAL_S = 30.0

ANALYSIS_MODEL = "claude-sonnet-4-6"
DECISION_MODEL = "claude-opus-4-7"

WHISPER_URL = os.environ.get("WHISPER_URL", "http://192.168.1.100:8178")
AUDIO_CHUNK_S = 10
AUDIO_SAMPLE_RATE = 16000
AUDIO_BLOCK_MS = 100  # callback cadence — gives a ~10 Hz mic level meter
# Default RMS threshold below which we treat a recorded chunk as silence
# and skip the Whisper call entirely. Whisper hallucinates beautifully on
# silence ("Thanks for watching, see you in the next one, bye bye...")
# because the model was trained on a lot of YouTube. int16 audio ranges
# to ±32k; quiet rooms sit ~50–200, real speech sits ~1000+. The live
# value is per-bench tunable via the dashboard slider.
AUDIO_RMS_THRESHOLD_DEFAULT = 250

log = Logger("scribe")


# ---------------------------------------------------------------------------
# Secrets — App Lab CLI 0.9.0 silently ignores a top-level `environment:`
# block in app.yaml, and the canonical "Brick Configuration" path only
# applies to bricks that declare `secret: true` variables (which we don't
# use, since we go to the Anthropic SDK directly for Sonnet 4.6 vision and
# Opus 4.7). So we load secrets ourselves from a project-local file.
#
# The project directory is bind-mounted to /app inside the container, so a
# .secrets.env at the project root is the only path that's reachable both
# from the host (where Kev edits it) and from inside the container (where
# this code runs). Gitignored.
# ---------------------------------------------------------------------------

def _load_secrets_env() -> None:
    """Read KEY=value lines from .secrets.env at the project root and
    inject them into os.environ (without overriding anything already set
    by the compose generator). Bash-style comments and blank lines OK."""
    env_path = Path("/app/.secrets.env")
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_secrets_env()

if not os.environ.get("ANTHROPIC_API_KEY"):
    log.warning(
        "ANTHROPIC_API_KEY not set — vision analysis and end-of-session "
        "classification will fail. Create .secrets.env at the project root "
        "with ANTHROPIC_API_KEY=sk-... (see README)."
    )

client = Anthropic()


# ---------------------------------------------------------------------------
# Camera discovery — /dev/video0 and /dev/video1 on the Uno Q are the
# Qualcomm Venus M2M codec, not capture devices. Walk sysfs and pick the
# first node bound to uvcvideo that actually returns a frame.
# ---------------------------------------------------------------------------

def find_camera() -> cv2.VideoCapture | None:
    candidates: list[str] = []
    for sysfs in sorted(Path("/sys/class/video4linux").glob("video*")):
        try:
            driver = (sysfs / "device" / "driver").resolve().name
        except OSError:
            continue
        if driver == "uvcvideo":
            candidates.append(f"/dev/{sysfs.name}")

    if not candidates:
        log.error(
            "no USB UVC camera found — check that the webcam is plugged into "
            "the Uno Q's USB host port (lsusb should list it). /dev/video0 "
            "and /dev/video1 are the SoC's Venus codec, not a webcam."
        )
        return None

    for path in candidates:
        c = cv2.VideoCapture(path)
        if not c.isOpened():
            c.release()
            continue
        ok, _ = c.read()
        if ok:
            log.info(f"camera: opened {path}")
            return c
        c.release()

    log.error(f"found UVC nodes {candidates} but none returned frames")
    return None


# ---------------------------------------------------------------------------
# Session data
# ---------------------------------------------------------------------------

@dataclass
class Event:
    timestamp: float
    kind: str
    detail: str
    frame_path: str | None = None
    # Optional handle so the dashboard can delete an individual line later.
    # Set for narration events; left None for activity/component/etc.
    id: str | None = None


@dataclass
class Narration:
    """A transcribed line. Stored on the session so the user can delete
    individual lines from the dashboard if Whisper hallucinated, before
    they pollute the end-of-session classification digest."""
    id: str
    text: str
    timestamp: float

    @property
    def time_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")


@dataclass
class Session:
    started_at: float = field(default_factory=time.time)
    events: list[Event] = field(default_factory=list)
    timelapse_frames: list[str] = field(default_factory=list)
    transcript: list[Narration] = field(default_factory=list)

    @property
    def session_dir(self) -> Path:
        d = SESSION_DIR / datetime.fromtimestamp(self.started_at).strftime("%Y%m%d-%H%M%S")
        d.mkdir(parents=True, exist_ok=True)
        return d


class FrameChangeDetector:
    """Cheap local gate that drops near-identical frames before they hit the API."""

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
# Prompts
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are observing a maker's workshop bench. Look at this frame and respond with a JSON object only.

Return raw JSON. Do NOT wrap your response in markdown code fences (no ```json, no ```). Do not include any prose before or after the JSON.

Schema:
{
  "activity": one of [soldering, wiring, screwing, testing, debugging, idle, other],
  "components_visible": list of identifiable electronic components or tools (max 5),
  "notable": short string describing anything worth noting for a build log, or null,
  "is_key_moment": true if this looks like a milestone (first power-on, completed assembly, visible mistake) — used sparingly
}

Be terse. If nothing is happening, say activity is "idle" and notable is null."""


def _strip_json_fence(text: str) -> str:
    """Belt-and-braces: even with the prompt asking for raw JSON, Sonnet
    sometimes wraps responses in ```json ... ``` fences. Strip those (and
    any leading/trailing whitespace) before json.loads()."""
    s = text.strip()
    if s.startswith("```"):
        # Drop the opening fence (with or without a language tag).
        s = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", s)
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()

CLASSIFICATION_PROMPT = """You watched a maker's workshop session. Below is the full event log and transcript of what was said.

Decide which deliverable best fits this session. Choose ONE:
  - "tutorial": linear progression, clear narration, suitable as a teaching post
  - "build_log": iterative work with debugging, suitable as a development diary
  - "video_script": strong narrative arc with a finished artefact, suitable for a YouTube video

Then produce that deliverable. For tutorial: clean blog post with steps. For build_log: chronological notes with decisions and lessons. For video_script: md2script format with scene headings and dialogue.

Respond with JSON: {"choice": "<one of the above>", "reasoning": "<one sentence>", "output": "<the full deliverable in markdown>"}"""


# ---------------------------------------------------------------------------
# Audio capture + remote Whisper transcription
# ---------------------------------------------------------------------------

# record_chunk lives on the Scribe brick — it needs access to the WebUI
# instance so the audio callback can emit live RMS for the dashboard
# meter. See Scribe._record_chunk below.


# Whisper hallucinations on near-silence and noise. Two filter layers:
#
# 1) No alphanumeric characters at all (".", "...", "…") — pure
#    punctuation, definitely noise.
# 2) Known YouTube-trained sign-off boilerplate. Whisper saw enormous
#    amounts of "thanks for watching, see you in the next one, bye bye"
#    in its training data and reaches for it on quiet audio.
# 3) Word-level repetition. "Bye Bye Bye Bye" or "The The The The" —
#    when one or two unique words make up >70% of the chunk, drop it.
_HAS_WORD_CHAR = re.compile(r"[A-Za-z0-9]")
_WORD_RE = re.compile(r"[A-Za-z']+")
_HALLUCINATION_PHRASES = (
    "thanks for watching",
    "thank you for watching",
    "see you in the next one",
    "see you next time",
    "see you in the next video",
    "subscribe to the channel",
    "subscribe to my channel",
    "don't forget to subscribe",
    "like and subscribe",
    "ring the bell",
    "hit that subscribe button",
)


def is_real_speech(text: str) -> bool:
    if not _HAS_WORD_CHAR.search(text):
        return False

    lower = text.lower()
    for phrase in _HALLUCINATION_PHRASES:
        if phrase in lower:
            return False

    words = _WORD_RE.findall(lower)
    if len(words) >= 3:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:
            return False

    return True


def transcribe_chunk(wav_bytes: bytes) -> str:
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
        log.warning(f"whisper server error: {e}")
        return ""


# ---------------------------------------------------------------------------
# The Scribe brick
# ---------------------------------------------------------------------------

@brick
class Scribe:
    """Workshop Scribe agent, packaged as an App Lab brick.

    App.run() auto-discovers each @brick.loop method and runs it on its own
    daemon thread, calling it repeatedly until shutdown. Each loop method
    sleeps at the top to control its own rate and returns once per iteration.

    Per-session loops (analysis, timelapse, transcript) check self.session
    and short-circuit when no session is active — that way the threads are
    always alive and we don't have to start/stop them per session.
    """

    def __init__(self, webui: WebUI) -> None:
        self.webui = webui
        self.session: Session | None = None
        # While True, end_session() has been called and _finalize_session
        # is still grinding through the Opus call. The session object isn't
        # cleared until that thread finishes, so the status broadcast loop
        # has to consult this flag to avoid flipping the pill back to
        # "recording" every second during the 20-30s classification window.
        self._wrapping_up = False
        self.detector = FrameChangeDetector()
        self.cap: cv2.VideoCapture | None = find_camera()
        self._last_camera_warning = time.time()
        self._cap_lock = threading.Lock()

        # Per-bench tunable settings, loaded from .scribe-config.json so
        # they survive restarts. Defaults if the file is absent.
        cfg = self._load_config()
        self.rotation = int(cfg.get("rotation", 0)) % 360
        if self.rotation not in (0, 90, 180, 270):
            self.rotation = 0
        self.rms_threshold = float(cfg.get("rms_threshold", AUDIO_RMS_THRESHOLD_DEFAULT))
        self.transcription_enabled = bool(cfg.get("transcription_enabled", True))

        # Console ring buffer — replay the last N entries to any client that
        # connects mid-session so they see context, not a blank screen.
        self._console_history: deque[dict] = deque(maxlen=200)
        webui.on_connect(self._on_client_connect)

        # Wire up the dashboard's REST API. expose_api just adds FastAPI
        # routes — safe to call before the server starts.
        webui.expose_api("POST", "/api/session/start", self.api_start_session)
        webui.expose_api("POST", "/api/session/stop", self.api_stop_session)
        webui.expose_api("GET", "/api/session/status", self.api_session_status)
        webui.expose_api("POST", "/api/debug/snapshot", self.api_debug_snapshot)
        webui.expose_api("GET", "/api/sessions", self.api_list_sessions)
        webui.expose_api("GET", "/api/sessions/{name}/download", self.api_download_session)
        webui.expose_api("DELETE", "/api/sessions/{name}", self.api_delete_session)
        webui.expose_api("GET", "/api/settings", self.api_get_settings)
        webui.expose_api("POST", "/api/camera/rotation", self.api_set_rotation)
        webui.expose_api("POST", "/api/audio/threshold", self.api_set_threshold)
        webui.expose_api("POST", "/api/audio/transcription", self.api_set_transcription)
        webui.expose_api("DELETE", "/api/transcript/{narration_id}", self.api_delete_narration)

    # Persistent settings ---------------------------------------------------

    def _load_config(self) -> dict:
        try:
            return json.loads(CONFIG_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_config(self) -> None:
        cfg = {
            "rotation": self.rotation,
            "rms_threshold": self.rms_threshold,
            "transcription_enabled": self.transcription_enabled,
        }
        try:
            CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        except OSError as e:
            log.warning(f"could not persist config to {CONFIG_PATH}: {e}")

    # Debug console ---------------------------------------------------------

    def _console(self, level: str, source: str, text: str) -> None:
        """Emit a structured line to the dashboard console *and* the python
        log. Level is one of debug/info/warn/error and styles the line on
        the client. Lines are kept in a ring buffer so a refresh replays
        recent context."""
        entry = {
            "level": level,
            "source": source,
            "text": text,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        self._console_history.append(entry)
        getattr(log, level if level in ("info", "warning", "error") else "info")(
            f"[{source}] {text}"
        )
        self.webui.send_message("console", entry)

    def _on_client_connect(self, sid: str) -> None:
        """Replay context for any newly-connected dashboard client so the
        UI doesn't appear blank on a refresh mid-session: console history
        plus the active session's transcript so far."""
        for entry in list(self._console_history):
            self.webui.send_message("console", entry, room=sid)
        sess = self.session
        if sess is not None:
            for n in sess.transcript:
                self.webui.send_message("transcript", {
                    "id": n.id, "text": n.text, "time": n.time_str,
                }, room=sid)

    # App lifecycle hooks ----------------------------------------------------

    def start(self) -> None:
        """Called once by App before any loop methods spin up.

        Register MCU→MPU RPC handlers here. The Bridge socket may not be
        connected yet; provide() will block briefly waiting for the router
        and raise if it can't reach it. We log and continue — the dashboard
        is still useful without the MCU.
        """
        mcu_handlers_registered = 0
        for name, handler in [
            ("marker_pressed", self.on_marker_pressed),
            ("iron_on", self.on_iron_on),
            ("iron_off", self.on_iron_off),
            ("session_end", self.on_session_end),
        ]:
            try:
                Bridge.provide(name, handler)
                mcu_handlers_registered += 1
            except Exception as e:
                log.warning(f"Bridge.provide({name}) failed (MCU may be offline): {e}")

        self._console(
            "info", "boot",
            f"scribe ready — mcu handlers registered: {mcu_handlers_registered}/4, "
            f"camera: {'ok' if self.cap is not None else 'missing'}, "
            f"api key: {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'MISSING'}"
        )

    # Camera ----------------------------------------------------------------

    # OpenCV's rotate codes for the four orthogonal angles we support.
    # Hung-on-ceiling, mirrored in a vise, etc. all just need a flip.
    _ROTATE_CODES = {
        90:  cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    def capture_frame(self) -> cv2.Mat | None:
        """Thread-safe frame grab with rotation applied. Both the web
        preview loop and the analysis loop call this concurrently — the
        rotation is applied here so Sonnet always sees the same orientation
        as the dashboard preview."""
        with self._cap_lock:
            if self.cap is None:
                now = time.time()
                if now - self._last_camera_warning > CAMERA_RETRY_INTERVAL_S:
                    self._last_camera_warning = now
                    log.warning("no camera available — retrying detection")
                    self.cap = find_camera()
                if self.cap is None:
                    return None
            ok, frame = self.cap.read()
            if not ok:
                log.warning("camera read failed — will retry detection")
                self.cap.release()
                self.cap = None
                return None
        code = self._ROTATE_CODES.get(self.rotation)
        if code is not None:
            frame = cv2.rotate(frame, code)
        return frame

    # Continuous loops (always running) -------------------------------------

    @brick.loop
    def web_frame_loop(self) -> None:
        time.sleep(WEB_FRAME_INTERVAL_S)
        frame = self.capture_frame()
        if frame is None:
            return
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not ok:
            return
        self.webui.send_message("frame", {
            "jpeg": base64.b64encode(jpeg.tobytes()).decode(),
        })

    @brick.loop
    def status_broadcast_loop(self) -> None:
        time.sleep(STATUS_BROADCAST_INTERVAL_S)
        if self._wrapping_up:
            # Don't reveal duration/events while Opus is generating —
            # the dashboard pill should read "thinking" cleanly.
            self.webui.send_message("status", {
                "state": "thinking", "duration": 0, "events": 0,
            })
            return
        sess = self.session
        if sess is not None:
            self.webui.send_message("status", {
                "state": "recording",
                "duration": int(time.time() - sess.started_at),
                "events": len(sess.events),
            })
        else:
            self.webui.send_message("status", {
                "state": "idle", "duration": 0, "events": 0,
            })

    # Session-scoped loops (no-op when no session) --------------------------

    @brick.loop
    def analysis_loop(self) -> None:
        time.sleep(FRAME_INTERVAL_S)
        sess = self.session
        if sess is None:
            return
        frame = self.capture_frame()
        if frame is None:
            return
        if not self.detector.is_meaningfully_different(frame):
            self._console("debug", "frame-gate", "no significant change — skipping vision call")
            return
        path = sess.session_dir / f"analysis_{int(time.time())}.jpg"
        cv2.imwrite(str(path), frame)
        try:
            self._analyse_frame(sess, path)
        except Exception as e:
            self._console("error", "vision", f"analyse_frame failed: {e}")

    @brick.loop
    def timelapse_loop(self) -> None:
        time.sleep(TIMELAPSE_INTERVAL_S)
        sess = self.session
        if sess is None:
            return
        frame = self.capture_frame()
        if frame is None:
            return
        path = sess.session_dir / f"tl_{int(time.time())}.jpg"
        cv2.imwrite(str(path), frame)
        sess.timelapse_frames.append(str(path))

    def _record_chunk(self) -> tuple[bytes, float]:
        """Record AUDIO_CHUNK_S of mono 16-bit PCM, returning (wav_bytes,
        overall_rms). Uses an InputStream + callback so we can emit a
        live RMS reading at ~10 Hz to the dashboard for the VU meter,
        rather than only learning the energy after the full chunk.
        """
        chunk_samples = int(AUDIO_CHUNK_S * AUDIO_SAMPLE_RATE)
        block_samples = max(1, int(AUDIO_SAMPLE_RATE * AUDIO_BLOCK_MS / 1000))
        buf = np.zeros(chunk_samples, dtype=np.int16)
        written = 0

        def callback(indata, frames, _time_info, _status):
            nonlocal written
            mono = indata[:, 0] if indata.ndim > 1 else indata
            n = min(frames, chunk_samples - written)
            if n > 0:
                buf[written:written + n] = mono[:n]
                written += n
            # Live VU meter — short-window RMS broadcast straight to UI.
            block_rms = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))
            self.webui.send_message("audio_level", {
                "rms": block_rms,
                "threshold": self.rms_threshold,
            })

        with sd.InputStream(
            callback=callback,
            samplerate=AUDIO_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=block_samples,
        ):
            deadline = time.monotonic() + AUDIO_CHUNK_S + 1.0
            while written < chunk_samples and time.monotonic() < deadline:
                time.sleep(0.05)

        # Final reading: RMS over the whole chunk, used for the gate.
        used = buf[:written] if written > 0 else buf
        overall_rms = float(np.sqrt(np.mean(used.astype(np.float32) ** 2)))

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(AUDIO_SAMPLE_RATE)
            wf.writeframes(buf.tobytes())
        return wav_buf.getvalue(), overall_rms

    @brick.loop
    def transcript_loop(self) -> None:
        """Always-on transcription preview.

        Whisper runs locally on the Mac so there's no cost to keep this
        loop running outside a session — and it's genuinely useful as a
        confidence check that the mic is picking up dialogue correctly
        before you commit to starting a session. Lines emit to the
        dashboard regardless; they only get persisted into the session's
        event log when a session is actually active.

        Two-layer noise rejection:
          1. RMS energy gate before we even hit Whisper (silence skipped).
          2. Hallucination filter on the returned text (denylist + repetition).
        """
        try:
            wav, rms = self._record_chunk()
        except Exception as e:
            log.warning(f"_record_chunk failed (mic missing?): {e}")
            time.sleep(2)
            return
        # Pause check happens *after* the recording so the mic level meter
        # keeps animating while paused — that way the user can see when
        # the room is quiet enough to resume.
        if not self.transcription_enabled:
            return
        if rms < self.rms_threshold:
            self._console(
                "debug", "asr",
                f"silence (rms={rms:.0f} < threshold={self.rms_threshold:.0f}) — skipping whisper"
            )
            return
        text = transcribe_chunk(wav)
        if not is_real_speech(text):
            self._console(
                "debug", "asr",
                f"dropped non-speech (rms={rms:.0f}): {text[:80]!r}"
            )
            return

        narration = Narration(
            id=uuid.uuid4().hex[:8],
            text=text,
            timestamp=time.time(),
        )
        self.webui.send_message("transcript", {
            "id": narration.id,
            "text": narration.text,
            "time": narration.time_str,
        })

        sess = self.session
        if sess is not None:
            sess.transcript.append(narration)
            self._log_event(sess, "narration", text, id=narration.id)

    # Per-frame analysis ----------------------------------------------------

    def _analyse_frame(self, sess: Session | None, frame_path: Path) -> dict | None:
        """Send a frame to Sonnet and (when a session is active) log the
        results as events. Returns the parsed JSON response (or None on
        parse failure) so the manual /api/debug/snapshot path can show it
        even outside a session."""
        img_bytes = frame_path.read_bytes()
        img_b64 = base64.standard_b64encode(img_bytes).decode()

        self._console(
            "info", "vision",
            f"→ {ANALYSIS_MODEL}: {len(img_bytes) // 1024}kB JPEG"
        )
        t0 = time.perf_counter()
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
        dt_ms = int((time.perf_counter() - t0) * 1000)
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "input_tokens", "?") if usage else "?"
        out_tok = getattr(usage, "output_tokens", "?") if usage else "?"

        raw_text = resp.content[0].text if resp.content else ""
        try:
            data = json.loads(_strip_json_fence(raw_text))
        except (json.JSONDecodeError, IndexError):
            self._console(
                "warning", "vision",
                f"← {dt_ms}ms — couldn't parse JSON: {raw_text[:120]!r}"
            )
            return None

        # One terse summary line for the console — easy to scan during a build.
        components = data.get("components_visible") or []
        comps_str = ", ".join(components[:5]) if components else "—"
        notable = data.get("notable") or ""
        summary = (
            f"← {dt_ms}ms ({in_tok}→{out_tok} tok) "
            f"activity={data.get('activity', '?')}  "
            f"components=[{comps_str}]"
        )
        if notable:
            summary += f"  notable: {notable!r}"
        if data.get("is_key_moment"):
            summary += "  ★KEY MOMENT"
        self._console("info", "vision", summary)

        if sess is None:
            return data  # snapshot-only mode: don't log session events

        if data.get("activity") and data["activity"] != "idle":
            self._log_event(sess, "activity", data["activity"], str(frame_path))
        for comp in components:
            self._log_event(sess, "component", comp, str(frame_path))
        if data.get("notable"):
            self._log_event(sess, "notable", data["notable"], str(frame_path))
        if data.get("is_key_moment"):
            self._log_event(sess, "key_moment", data.get("notable") or "milestone", str(frame_path))
        return data

    def _log_event(self, sess: Session, kind: str, detail: str,
                   frame_path: str | None = None, id: str | None = None) -> None:
        sess.events.append(Event(time.time(), kind, detail, frame_path, id))
        self._console("info", f"event:{kind}", detail)
        # Best-effort MCU pulse — don't let a missing sketch.ino kill the loop.
        try:
            Bridge.notify("pulse_led", kind)
        except Exception:
            pass

    # Session lifecycle -----------------------------------------------------

    def start_session(self) -> None:
        if self.session is not None:
            return
        self.session = Session()
        self.detector = FrameChangeDetector()
        self._console("info", "session", f"started — {self.session.session_dir}")
        try:
            Bridge.notify("set_status", "watching")
        except Exception:
            pass

    def end_session(self) -> None:
        sess = self.session
        if sess is None or self._wrapping_up:
            return
        self._wrapping_up = True
        self._console(
            "info", "session",
            f"ending — {len(sess.events)} events, "
            f"{len(sess.transcript)} transcript lines, "
            f"{len(sess.timelapse_frames)} timelapse frames"
        )
        try:
            Bridge.notify("set_status", "thinking")
        except Exception:
            pass
        self.webui.send_message("status", {"state": "thinking", "duration": 0, "events": 0})

        # Run the Opus call off-thread so the HTTP/RPC caller returns
        # immediately. This is the dramatic-pause moment in the eventual
        # YouTube cut — we want the dashboard pill to show "thinking"
        # while the model is actually working.
        threading.Thread(
            target=self._finalize_session, args=(sess,),
            name="Scribe._finalize_session", daemon=True,
        ).start()

    def _finalize_session(self, sess: Session) -> None:
        # Write the transcript first so a crash during classification still
        # leaves a usable artefact in the session folder.
        try:
            tpath = self._write_transcript(sess)
            if tpath is not None:
                self._console("info", "transcript", f"wrote {tpath}")
        except Exception as e:
            self._console("warning", "transcript", f"could not write transcript.txt: {e}")

        try:
            self._console("info", "classify", f"→ {DECISION_MODEL}: digesting session")
            t0 = time.perf_counter()
            result = self._classify_and_generate(sess)
            dt_ms = int((time.perf_counter() - t0) * 1000)
            self._console(
                "info", "classify",
                f"← {dt_ms}ms  choice={result['choice']!r}  "
                f"reason: {result.get('reasoning', '?')}"
            )
            path = self._write_deliverable(sess, result)
            self._console("info", "deliverable", f"wrote {path}")
        except Exception as e:
            self._console("error", "classify", f"failed: {e}")
        finally:
            try:
                Bridge.notify("set_status", "done")
            except Exception:
                pass
            self.webui.send_message("status", {"state": "done", "duration": 0, "events": 0})
            self.webui.send_message("sessions_changed", {})
            self.session = None
            self._wrapping_up = False

    def _classify_and_generate(self, sess: Session) -> dict:
        digest = {
            "duration_minutes": int((time.time() - sess.started_at) / 60),
            "events": [asdict(e) for e in sess.events],
            "transcript": " ".join(n.text for n in sess.transcript),
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

    def _write_transcript(self, sess: Session) -> Path | None:
        """Save a human-readable transcript.txt next to the deliverable.

        Pulls timestamps from sess.events (kind=='narration') so each line
        is prefixed with [HH:MM:SS] — useful for cross-referencing the
        timelapse frames or for editing the eventual YouTube cut.
        Returns None if there's nothing to write.
        """
        narrations = [e for e in sess.events if e.kind == "narration"]
        if not narrations:
            return None
        lines = [
            f"# Transcript — {datetime.fromtimestamp(sess.started_at):%Y-%m-%d %H:%M:%S}",
            f"# {len(narrations)} narration line(s) over "
            f"{int((time.time() - sess.started_at) / 60)} minutes",
            "",
        ]
        for ev in narrations:
            ts = datetime.fromtimestamp(ev.timestamp).strftime("%H:%M:%S")
            lines.append(f"[{ts}]  {ev.detail}")
        out = sess.session_dir / "transcript.txt"
        out.write_text("\n".join(lines) + "\n")
        return out

    def _write_deliverable(self, sess: Session, result: dict) -> Path:
        out = sess.session_dir / f"{result['choice']}.md"
        out.write_text(result["output"])
        log.info(
            f"wrote {out} — choice was '{result['choice']}' because: {result['reasoning']}"
        )
        return out

    # MCU event handlers (invoked from the Bridge read thread) --------------

    def on_marker_pressed(self, *_args) -> None:
        self._console("info", "mcu", "marker pressed")
        sess = self.session
        if sess is not None:
            self._log_event(sess, "marker", "user marked this moment as important")

    def on_iron_on(self, *_args) -> None:
        self._console("info", "mcu", "soldering iron on (current clamp threshold crossed)")
        sess = self.session
        if sess is not None:
            self._log_event(sess, "activity", "soldering iron powered on")

    def on_iron_off(self, *_args) -> None:
        self._console("info", "mcu", "soldering iron off")
        sess = self.session
        if sess is not None:
            self._log_event(sess, "activity", "soldering iron powered off")

    def on_session_end(self, *_args) -> None:
        self._console("info", "mcu", "long-press → ending session")
        self.end_session()

    # HTTP API handlers (called by FastAPI from the dashboard) -------------

    def api_start_session(self) -> dict:
        self.start_session()
        return {"status": "started"}

    def api_stop_session(self) -> dict:
        self.end_session()
        return {"status": "stopping"}

    def api_session_status(self) -> dict:
        sess = self.session
        if sess is None:
            return {"active": False, "duration": 0, "events": 0, "transcript_lines": 0}
        return {
            "active": True,
            "duration": int(time.time() - sess.started_at),
            "events": len(sess.events),
            "transcript_lines": len(sess.transcript),
        }

    # Per-bench settings (camera rotation, audio threshold) -----------------

    def api_get_settings(self) -> dict:
        """Initial state for the dashboard's controls. Called once on
        page load so sliders and rotation buttons reflect the actual
        persisted values rather than defaulting to UI stubs."""
        return {
            "rotation": self.rotation,
            "rms_threshold": self.rms_threshold,
            "rms_threshold_default": AUDIO_RMS_THRESHOLD_DEFAULT,
            "transcription_enabled": self.transcription_enabled,
        }

    def api_set_rotation(self, req: RotationRequest) -> dict:
        if req.rotation not in (0, 90, 180, 270):
            raise HTTPException(status_code=400, detail="rotation must be 0/90/180/270")
        self.rotation = req.rotation
        self._save_config()
        self._console("info", "camera", f"rotation set to {self.rotation}°")
        return {"ok": True, "rotation": self.rotation}

    def api_set_threshold(self, req: ThresholdRequest) -> dict:
        # Clamp to a sensible range — int16 RMS theoretically maxes at
        # ~32k but anything above ~3000 would never gate real speech.
        threshold = max(0.0, min(3000.0, req.threshold))
        self.rms_threshold = threshold
        self._save_config()
        self._console("info", "audio", f"rms threshold set to {threshold:.0f}")
        return {"ok": True, "rms_threshold": self.rms_threshold}

    def api_set_transcription(self, req: TranscriptionEnabledRequest) -> dict:
        """Toggle whether transcript_loop ships chunks to Whisper.

        The mic level meter keeps animating either way (record_chunk
        runs on its own cadence) — only the network call to Whisper
        is suppressed when paused.
        """
        self.transcription_enabled = bool(req.enabled)
        self._save_config()
        self._console(
            "info", "audio",
            f"transcription {'resumed' if self.transcription_enabled else 'paused'}"
        )
        return {"ok": True, "enabled": self.transcription_enabled}

    def api_delete_narration(self, narration_id: str) -> dict:
        """Remove a narration line from the active session — both from
        sess.transcript (the list Opus will see) and sess.events (so
        transcript.txt won't include it). Always emits a
        transcript_deleted socket message so every connected client
        removes the matching DOM node, regardless of whether the line
        was preview-only or session-stored.
        """
        removed = False
        sess = self.session
        if sess is not None:
            before = len(sess.transcript)
            sess.transcript = [n for n in sess.transcript if n.id != narration_id]
            sess.events = [e for e in sess.events if e.id != narration_id]
            removed = len(sess.transcript) < before
        self.webui.send_message("transcript_deleted", {"id": narration_id})
        if removed:
            self._console("info", "transcript", f"deleted narration {narration_id}")
        return {"ok": True, "id": narration_id, "removed_from_session": removed}

    # Session library — list, download (zip), delete -----------------------

    _SESSION_NAME_RE = re.compile(r"^\d{8}-\d{6}$")  # YYYYMMDD-HHMMSS

    @classmethod
    def _is_session_name(cls, name: str) -> bool:
        """Guard against path traversal — only accept names that match the
        timestamp format we generate for session directories."""
        return bool(cls._SESSION_NAME_RE.fullmatch(name))

    def _summarise_session(self, d: Path) -> dict | None:
        try:
            started = datetime.strptime(d.name, "%Y%m%d-%H%M%S")
        except ValueError:
            return None
        files = [f for f in d.iterdir() if f.is_file()]
        deliverable = None
        for kind in ("tutorial", "build_log", "video_script"):
            if (d / f"{kind}.md").exists():
                deliverable = kind
                break
        return {
            "name": d.name,
            "started_at": started.isoformat(),
            "file_count": len(files),
            "size_bytes": sum(f.stat().st_size for f in files),
            "deliverable": deliverable,
            "is_active": (
                self.session is not None
                and self.session.session_dir.name == d.name
            ),
        }

    def api_list_sessions(self) -> list[dict]:
        """Return all completed sessions, newest first. The active
        session (if any) is included with is_active=true so the UI can
        disable destructive actions on it."""
        if not SESSION_DIR.exists():
            return []
        out: list[dict] = []
        for d in sorted(SESSION_DIR.iterdir(), reverse=True):
            if not d.is_dir() or not self._is_session_name(d.name):
                continue
            summary = self._summarise_session(d)
            if summary is not None:
                out.append(summary)
        return out

    def api_download_session(self, name: str):
        """Stream a zip of the entire session directory. Browsers handle
        the download natively when this URL is hit via a normal <a> tag."""
        if not self._is_session_name(name):
            raise HTTPException(status_code=400, detail="invalid session name")
        target = SESSION_DIR / name
        if not target.is_dir():
            raise HTTPException(status_code=404, detail="session not found")

        # Build the zip in memory. A ~90-minute session with frames is
        # ~10-30 MB — fine to hold in RAM, avoids juggling temp files.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in target.iterdir():
                if f.is_file():
                    zf.write(f, arcname=f"{name}/{f.name}")
        buf.seek(0)
        self._console("info", "session", f"download requested: {name} ({buf.getbuffer().nbytes // 1024} KB)")
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{name}.zip"'},
        )

    def api_delete_session(self, name: str) -> dict:
        if not self._is_session_name(name):
            raise HTTPException(status_code=400, detail="invalid session name")
        target = SESSION_DIR / name
        if not target.is_dir():
            raise HTTPException(status_code=404, detail="session not found")
        if self.session is not None and self.session.session_dir.name == name:
            raise HTTPException(status_code=409, detail="cannot delete the active session")
        shutil.rmtree(target)
        self._console("info", "session", f"deleted {name}")
        self.webui.send_message("sessions_changed", {})
        return {"ok": True, "name": name}

    def api_debug_snapshot(self) -> dict:
        """Capture the current camera frame and run it through the vision
        model immediately, regardless of session state. Result lands on
        the console as a normal vision event so users can verify the
        full pipeline without waiting for the 30s analysis cadence."""
        self._console("info", "snapshot", "manual snapshot requested from dashboard")
        frame = self.capture_frame()
        if frame is None:
            self._console("error", "snapshot", "no camera frame available")
            return {"ok": False, "error": "no frame"}

        # Stash next to session frames if a session is running, otherwise
        # under sessions/snapshots/ so we don't pollute a real session dir.
        sess = self.session
        target_dir = sess.session_dir if sess else SESSION_DIR / "snapshots"
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"snapshot_{int(time.time())}.jpg"
        cv2.imwrite(str(path), frame)

        try:
            data = self._analyse_frame(sess, path)
            return {"ok": True, "result": data, "path": str(path)}
        except Exception as e:
            self._console("error", "snapshot", f"vision call failed: {e}")
            return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Module entry — bricks register themselves on construction; App.run() then
# calls each brick's start() and runs every @brick.loop method on its own
# daemon thread until SIGTERM/Ctrl-C.
# ---------------------------------------------------------------------------

webui = WebUI(assets_dir_path="/app/assets")
scribe = Scribe(webui=webui)


if __name__ == "__main__":
    App.run()
