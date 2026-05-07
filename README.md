# Workshop Scribe

A passive-observer AI agent for the **Arduino Uno Q** that watches a maker's
workshop bench, listens to ambient narration, and at the end of a session
decides what kind of artefact to produce — a tutorial blog post, a build log,
a YouTube video script, or a publishable blog post with a bill of materials
and step-by-step photos.

This is a project for [kevsrobots.com](https://www.kevsrobots.com). The novelty
is the agent's *judgement*: most edge-AI demos are reactive ("door opened →
log it"), this one is interpretive — *"I watched for 90 minutes, this was a
tutorial, here's the post"*.

---

## What it actually does

While running, the agent:

- Pulls frames from a USB camera attached to the Uno Q
- Streams ambient audio from a USB microphone to a **local Whisper server**
  running on your Mac/PC (no audio leaves your network)
- Sends the occasional frame to **Claude Sonnet 4.6** for vision analysis,
  but only when a cheap local change-detector says something has actually
  changed in the scene
- At the end of the session, sends the full event log + transcript to
  **Claude Opus 4.7**, which decides what kind of artefact best fits the
  session and produces it — *plus* an always-on structured blog post with
  bill of materials, step images, and a wrap-up

A small browser dashboard (served by the Uno Q at
[http://localhost:7000](http://localhost:7000)) shows the live camera, the
running transcript, a debug console, and a list of past sessions you can
download as a zip or delete.

A typical 90-minute session costs about **40-50 pence** in API spend.

---

## Bill of Materials

### Required

| Part | Notes |
|------|-------|
| **Arduino Uno Q (4 GB)** | The dual-brain SBC that runs the agent. The 4 GB version is required for headroom — the 2 GB variant struggles with OpenCV + Anthropic SDK + the Python venv. |
| **USB-C webcam** (any UVC) | Plain USB-class-compliant cameras work — Logitech C270/C920, generic 1080p modules, etc. The Uno Q's onboard `/dev/video0` and `/dev/video1` are *not* usable cameras; they're the SoC's hardware H.264 codec. The agent walks `/sys/class/video4linux` to find a real UVC device. |
| **USB microphone** (any class-compliant) | Webcam-built-in mics work fine. No special config needed — `sounddevice` picks the system default. |
| **Powered USB-C hub** with Power Delivery | The Uno Q has one USB host port. You need a hub to attach both the camera and the microphone (and any other USB peripherals). It must support PD pass-through, otherwise the Uno Q will brown-out under camera load. |
| **A second computer** to run the Whisper server | Mac, PC, or Linux box with Python 3.10+. It just needs to be on the same network as the Uno Q. A 2017-era MacBook Pro CPU-only handles real-time `base.en` Whisper comfortably. |

### Optional (for the full hardware setup)

These extend the dashboard with physical controls and tactile feedback. The
agent works without them — they're for when you want the LED matrix to glow
"watching" / "thinking" while you're soldering and don't want to alt-tab.

| Part | Used for |
|------|----------|
| **SCT-013 non-invasive AC current clamp** | Detects when the soldering iron draws power, so the agent knows soldering is happening without burning vision tokens to confirm it. Wires to A0 via a burden resistor + DC bias network — see `hardware/sct013_circuit.md` (TBD). |
| **Momentary push-button** | Wired to D2 (`INPUT_PULLUP`). Short press = "mark this moment as important". Long-hold (2s) = end session. |

The MCU sketch (`sketch/sketch.ino`) drives the Uno Q's onboard 8×13 LED
matrix and two RGB LEDs to show the agent's current state on the bench.

---

## Architecture: the dual-brain split

```
┌─────────────────────────────────────────────────────────────┐
│  MPU side  (Linux, Python, App Lab)                         │
│                                                             │
│  python/main.py                                             │
│  ├── Camera (OpenCV)        → frame capture                 │
│  ├── sounddevice + Whisper  → live transcription            │
│  ├── FrameChangeDetector    → cheap local gate (no API)     │
│  ├── Sonnet 4.6 vision      → per-frame analysis            │
│  ├── Opus 4.7 classifier    → end-of-session deliverables   │
│  └── WebUI brick (FastAPI)  → dashboard at :7000            │
└──────────────────────────┬──────────────────────────────────┘
                           │ UART, MessagePack, 115200 baud
                           │ (Arduino-Router service on MPU)
┌──────────────────────────┴──────────────────────────────────┐
│  MCU side  (Zephyr, C++, sketch.ino)  — optional            │
│                                                             │
│  ├── 8×13 LED matrix       → status indicator               │
│  ├── RGB LEDs              → activity pulse                 │
│  ├── SCT-013 polling       → notifies MPU of iron on/off    │
│  └── Button polling        → marker / session-end           │
└─────────────────────────────────────────────────────────────┘
```

---

## Setup

### 1. Set up your Uno Q with App Lab

Follow the official [Uno Q getting-started guide](https://docs.arduino.cc/hardware/uno-q/)
to flash the OS and install App Lab. You should end up with `arduino-app-cli`
on your `PATH` and a working `arduino` user with sudo. Confirm with:

```bash
arduino-app-cli version
```

### 2. Clone this repo onto the Uno Q

SSH to the Uno Q and clone into the App Lab apps directory:

```bash
ssh arduino@<uno-q-hostname>
cd ~/ArduinoApps
git clone https://github.com/kevinmcaleer/self-documenting-workshop.git
cd self-documenting-workshop
```

### 3. Add your Anthropic API key

The agent uses the Anthropic API directly (not the App Lab `cloud_llm`
brick) because it needs vision and Opus 4.7 access. Get a key from
[console.anthropic.com](https://console.anthropic.com/settings/keys), then
create `.secrets.env` in the project root:

```bash
cat > .secrets.env <<'EOF'
ANTHROPIC_API_KEY=sk-ant-...your-key...
EOF
```

This file is gitignored. The agent reads it at startup and injects the key
into its process environment — `app.yaml`'s top-level `environment:` block
is silently ignored by App Lab CLI 0.9.0, hence this approach.

### 4. Run the Whisper server on a separate machine

Whisper transcription runs **off-board**, on whatever computer you've got
sitting on the same network as the Uno Q. Audio chunks are POSTed to it
over HTTP — nothing goes to the cloud, and the Uno Q stays free of the
multi-GB Whisper model download.

On your Mac/PC:

```bash
git clone https://github.com/kevinmcaleer/self-documenting-workshop.git
cd self-documenting-workshop
pip install faster-whisper flask
python whisper_server.py
```

First run downloads the `base.en` model (~150 MB). Subsequent runs are
instant. The server listens on **port 8178** by default.

To customise — edit `whisper_server.py`:

- **Different model size**: change `WhisperModel("base.en", ...)` to
  `tiny.en` (faster, less accurate), `small.en` (slower, more accurate),
  or `medium.en` (much slower). For workshop dialogue, `base.en` is the
  sweet spot.
- **GPU acceleration**: change `device="cpu"` to `device="cuda"` (NVIDIA)
  or `device="metal"` (Apple Silicon) if you have it. CPU `int8` is fine
  for a single-user setup.
- **Different language**: change `language="en"` in `model.transcribe(...)`
  to `it`, `de`, `fr` etc. Use a multilingual model (drop `.en`).
- **Different port**: change `app.run(host="0.0.0.0", port=8178)`.
- **Health check**: `curl http://<your-mac-ip>:8178/health` should return
  `{"status":"ok"}`.

### 5. Tell the agent where the Whisper server lives

The agent finds the Whisper server via the `WHISPER_URL` env var. The
default in `python/main.py` is `http://192.168.1.100:8178` — almost
certainly wrong for your network. Add it to `.secrets.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
WHISPER_URL=http://192.168.1.42:8178
```

(Replace `192.168.1.42` with your Mac/PC's LAN IP. Find it on a Mac with
`ifconfig | grep 'inet '`, on Linux with `hostname -I`.)

### 6. Start the app

```bash
cd ~/ArduinoApps/self-documenting-workshop
arduino-app-cli app start .
```

First start takes a couple of minutes — App Lab compiles the MCU sketch,
provisions a Python venv, and pulls dependencies. Subsequent starts are
fast (~10 seconds).

Watch the logs in another terminal:

```bash
arduino-app-cli app logs . --follow
```

Look for:

```
[boot] scribe ready — mcu handlers registered: 4/4, camera: ok, api key: set
WebUI:  The application interface is available here:
  - Local URL:   http://localhost:7000
  - Network URL: http://<uno-q-ip>:7000
App started
```

Then open the **Network URL** from your laptop's browser.

---

## Using the dashboard

- **Start Session** — begins recording. Frames get captured every 30s
  (gated by motion), audio is transcribed continuously, MCU events are
  logged.
- **Stop Session** — ends recording. Opus 4.7 classifies the session in
  ~20-30 seconds and writes the deliverables.
- **Snapshot now** — fires a one-shot Sonnet vision call against the
  current frame. Useful for testing the pipeline or sanity-checking
  what the model sees.
- **Pause/Resume** (transcription) — temporarily stops sending audio
  chunks to Whisper (e.g. for a phone call). The mic-level meter keeps
  running so you can see when the room goes quiet.
- **Rotate** (camera) — cycles through 0° / 90° / 180° / 270°. Persists
  to `.scribe-config.json`. Both the dashboard preview *and* the frames
  sent to Sonnet rotate, so vision analysis stays consistent.
- **Noise gate slider** — RMS threshold below which audio is treated as
  silence and never sent to Whisper. Drag while the room is quiet to find
  your floor; the red tick on the meter shows the current cutoff.
- **Transcript line `×`** — hover over a transcript line to delete it
  before it lands in the Opus digest. Useful for cleaning out
  hallucinations or accidental bystander chatter.

---

## What gets saved per session

Each session lands in `sessions/YYYYMMDD-HHMMSS/`:

```
sessions/20260507-141114/
├── analysis_*.jpg     ← frames sent to Sonnet (motion-triggered)
├── tl_*.jpg           ← timelapse frames every 10s
├── snapshot_*.jpg     ← any manual snapshots fired during the session
├── transcript.txt     ← timestamped narration
├── build_log.md       ← chosen deliverable (or tutorial.md / video_script.md)
└── blog.md            ← always-on blog post with BOM, steps, images
```

The session folder is bind-mounted from the host, so you can browse it
in Finder/Explorer/VS Code without going through the dashboard. The
dashboard's Sessions strip lets you download the whole folder as a zip
or delete it.

The `blog.md` references images using relative basenames
(`![](analysis_1234.jpg)`), so the markdown renders cleanly when you
extract the zip anywhere or upload the folder as-is to a blog.

---

## Cost-control design

The agent is engineered around two principles:

1. **Cheap local gates before expensive API calls.** A motion-detector in
   OpenCV drops near-identical frames before they ever hit Claude. An RMS
   energy gate drops silent audio chunks before they hit Whisper.
2. **Two-tier model usage.** Sonnet 4.6 handles routine per-frame analysis
   (cheap, fast, plenty smart). Opus 4.7 only fires once per session, at
   the end, when the agent's *judgement* matters.

Estimated cost per 90-minute session: ~50p. That includes ~40 Sonnet
vision calls and one Opus classification call (which now produces both
the chosen deliverable and the structured blog post).

---

## Project structure

```
.
├── README.md                 — this file
├── CLAUDE.md                 — design rationale and constraints
├── app.yaml                  — App Lab project config (bricks, name)
├── .secrets.env              — ANTHROPIC_API_KEY etc. (gitignored)
├── .scribe-config.json       — per-bench UI preferences (gitignored)
├── python/
│   ├── main.py               — MPU-side agent
│   └── requirements.txt      — Python deps
├── sketch/
│   └── sketch.ino            — MCU-side sketch (status LEDs, button, current sensor)
├── assets/
│   └── index.html            — dashboard UI (camera, transcript, console, sessions)
├── whisper_server.py         — runs on a separate computer, transcribes audio
└── sessions/                 — per-session outputs (gitignored)
```

---

## Tuning notes

- **Camera upside down?** Hit the `Rotate` button on the camera card — it
  cycles in 90° steps and persists. Both the preview and the frames sent
  to Sonnet rotate, so vision keeps working correctly.
- **Hallucinated transcriptions** ("Bye bye bye bye…", "Thanks for
  watching…")? Two filters catch most of these: an RMS energy gate
  (slider in the dashboard) and a denylist of YouTube-trained sign-off
  phrases. If a specific hallucination keeps slipping through, add it to
  `_HALLUCINATION_PHRASES` at the top of `python/main.py`.
- **Whisper missing quiet speech?** Watch the debug console for
  `[asr] silence (rms=850 < threshold=...)` lines. If real speech is
  showing up there, drag the noise-gate slider lower.
- **Too many vision calls?** Increase `FRAME_INTERVAL_S` in
  `python/main.py` (default 30s). Or raise `MIN_PIXEL_DELTA` to require
  bigger scene changes.

---

## Honest caveats

- The MCU sketch is drafted but not yet bench-tested as of this writing.
  The dashboard is fully usable without it — long-press to end is just
  done from the browser instead.
- The session classifier (Opus 4.7) sometimes wraps its JSON response in
  markdown code fences despite being asked not to. The agent strips them
  defensively; if you ever see `[classify] could not parse Opus JSON…` in
  the console, that's where to look.
- The Whisper server has no auth. Don't expose port 8178 to the public
  internet.
- App Lab CLI 0.9.0 silently ignores top-level `environment:` blocks in
  `app.yaml`, which is why secret loading goes through `.secrets.env`
  instead of the more obvious compose-style approach.

---

## License

TBD — pending Kev's call. If you're reading this from the public repo,
the maker-friendly assumption is permissive use with attribution; ask
before relying on that for anything commercial.
