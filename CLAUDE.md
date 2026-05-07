# Workshop Scribe

A passive-observer AI agent for the Arduino Uno Q that watches a maker's workshop bench, logs what happens, and at the end of a session decides what kind of artefact to produce — a tutorial blog post, a build log, or a YouTube video script.

This is a YouTube/maker-education project for **kevsrobots.com**. The novelty is the agent's *judgement*: most edge-AI demos are reactive ("door opened → log it"), this one is interpretive ("I watched for 90 minutes, this was a tutorial, here's the post").

## Hardware

- **Arduino Uno Q (4GB)** — dual-brain SBC
  - **MPU**: Qualcomm QRB2210 quad-core Cortex-A53, 4GB LPDDR4, 32GB eMMC, runs Debian Linux. Hosts the Python agent.
  - **MCU**: STM32U585, runs Zephyr RTOS. Hosts a small C++ sketch for hardware control and the visible status indicator.
  - Onboard 8×13 LED matrix (note: 8 rows, 13 columns — *not* 12×8 like the Uno R4 WiFi).
  - 4 onboard RGB LEDs, two MCU-controllable.
- **USB-C webcam** (any UVC) on the Uno Q's USB host port — frame source.
- **USB microphone** (any class-compliant) — narration capture.
- **SCT-013 non-invasive AC current clamp** on A0 via burden resistor + DC bias network — detects when the soldering iron is drawing power without touching mains.
- **Momentary push-button** on D2 (INPUT_PULLUP) — short press marks a moment, long hold (2s) ends the session.

## Architecture: the dual-brain split

```
┌─────────────────────────────────────────────────────────────┐
│  MPU side  (Linux, Python, App Lab)                         │
│                                                              │
│  main.py                                                     │
│  ├── Camera Brick           → frame capture                 │
│  ├── ASR Cloud Brick        → live transcription            │
│  ├── FrameChangeDetector    → cheap local gate (no API)    │
│  ├── analyse_frame()        → Sonnet 4.6 vision call       │
│  ├── classify_and_generate()→ Opus 4.7, end-of-session     │
│  └── Bridge.call() / .subscribe() to MCU                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ UART, MessagePack, 115200 baud
                           │ (Arduino-Router service on MPU)
┌──────────────────────────┴──────────────────────────────────┐
│  MCU side  (Zephyr, C++, sketch.ino)                        │
│                                                              │
│  ├── 8×13 LED matrix       → 5 status states (idle,         │
│  │                            watching, noticed, thinking,  │
│  │                            done)                          │
│  ├── RGB LEDs              → secondary colour indicator     │
│  ├── SCT-013 polling       → notifies MPU of iron on/off    │
│  └── Button polling        → marker / session-end events    │
└─────────────────────────────────────────────────────────────┘
```

**Communication is via App Lab's Bridge / RPC layer.** On the MCU side use `RPC.bind("name", fn)` to expose a function and `RPC.call("name", arg)` to call back to the MPU. On the MPU side use `from arduino.app_utils import *` then `Bridge.call("name", arg)` and `Bridge.provide("name", fn)`. The transport is UART at 115200 baud with MessagePack encoding — fine for our message rate (a few per second peak).

## Cost-control design (this matters)

The agent uses the **Anthropic API directly** with a separate API key. Claude Max subscriptions don't cover headless agent usage as of April 2026 — Anthropic explicitly closed that loophole. Don't try to work around it.

Cost-control patterns baked in:

1. **Cheap local gate before any API call.** `FrameChangeDetector` runs in OpenCV on the MPU CPU — free. Frames that haven't meaningfully changed are dropped before they reach the API.
2. **Two-tier model usage.** Sonnet 4.6 for routine per-frame analysis (cheap, fast, plenty smart). Opus 4.7 *only once* per session, at the end, for classification + deliverable generation — that's the moment its judgement matters.
3. **ASR transcript is the cheap signal; vision is the expensive one.** Voice narration via the App Lab Cloud ASR Brick is much cheaper than vision frames. Most "what is happening" can be inferred from what Kev says out loud while building.
4. **MCU sensor inputs reduce vision load further.** When the SCT-013 says the iron is drawing power, the agent already knows soldering is happening — no vision call needed to confirm.

**Estimated cost per 90-minute build session: ~40p.** That includes ~40 Sonnet vision calls and one Opus classification call. Well worth it for a publishable artefact.

## Repo layout (current state and intended)

```
/                          — repo root
├── CLAUDE.md              — this file
├── README.md              — user-facing: hardware BOM, setup, usage
├── main.py                — MPU-side agent (skeleton exists, needs cleanup)
├── sketch.ino             — MCU-side sketch (drafted)
├── app.yaml               — App Lab project config (TBD)
├── requirements.txt       — Python deps for the MPU side
├── prompts/               — externalised prompt templates
│   ├── analyse_frame.txt
│   └── classify_session.txt
├── templates/             — output deliverable templates (TBD)
│   ├── tutorial.md.j2
│   ├── build_log.md.j2
│   └── video_script.md.j2
├── hardware/              — bench wiring docs
│   └── sct013_circuit.md  — burden resistor + DC bias network
└── sessions/              — generated outputs land here (gitignored)
    └── YYYYMMDD-HHMMSS/
        ├── analysis_*.jpg
        ├── tl_*.jpg
        ├── events.jsonl
        └── {tutorial|build_log|video_script}.md
```

## Where things stand

**Done:**
- Project concept and architecture nailed down.
- `main.py` skeleton with cost-aware design (frame-change gate, two-tier model, ASR + vision + MCU events).
- `sketch.ino` for the MCU with five visual states, button, current sensor.

**Known issues in the current code:**
- `main.py` uses placeholder `from bricks import camera, asr_cloud, bridge` imports. The real App Lab API is `from arduino.app_utils import *` and `Bridge.call()` / `Bridge.provide()`. The Camera Brick and ASR Cloud Brick call shapes need verifying against current App Lab docs once the board is in hand.
- The `bridge.subscribe("mcu_events")` async-iterator pattern is speculative. The MPU→MCU direction (`Bridge.call`) is well documented; the MCU→MPU direction is less polished. Worst case fallback: MCU sets a flag, MPU polls via `Bridge.call("get_pending_event")` — uglier but bulletproof.
- The end-of-session trigger in `main.py` is stubbed as KeyboardInterrupt. Real trigger is the long-press from the MCU sending a `session_end` event.
- `sketch.ino` matrix dimensions (8 rows × 13 cols) need confirming on hardware — Arduino's own docs occasionally have these swapped.
- SCT-013 threshold (`IRON_THRESHOLD_PP = 60`) is a placeholder. Calibrate on the bench.

**Next priorities (in order):**
1. **Define the three deliverable templates** — what should `tutorial.md`, `build_log.md`, and `video_script.md` actually look like when emitted? Pinning down "good output" sharpens the prompts.
2. **Clean up `main.py`** — replace placeholder imports with real App Lab API, verify Camera Brick + ASR Brick call shapes.
3. **Write `README.md`** — hardware BOM, App Lab project setup, build order, bench wiring.
4. **Hardware bring-up** — only after the above. Build order: Blink-via-Bridge → matrix rendering → button → current sensor → full integration.

## Storytelling notes (relevant for code decisions)

This is going to become a YouTube video, so some code decisions are driven by what's visible on camera:

- The LED matrix's 5 distinct states are the audience's window into the agent's "mind." Don't reduce them — the visual variety carries the explanation.
- The "noticed" matrix flash is intentionally over-bright so it's legible in B-roll. Cinematography over subtlety.
- The end-of-session beat (long-press → THINKING sweep → DONE checkmark while Opus is generating) is a deliberate dramatic pause. Don't optimise it away to a fast spinner.
- An honest "what it got wrong" section in the eventual video matters more than perfect output. The agent's *judgement* is the hook; visible misclassifications make the project more relatable, not less.

## Useful references

- App Lab + Bridge API: see Shawn Hymel's CLI tutorial and the Random Nerd Tutorials Uno Q getting-started guide
- LED matrix library: `Arduino_LED_Matrix` — uses `matrix.renderBitmap(buf, rows, cols)` for arbitrary patterns
- RPC under the hood: MessagePack over UART at 115200 baud (myembeddedstuff.com has a logic-analyser deep-dive if you ever need to debug at protocol level)
- App Lab examples to crib from: Blink LED (Bridge basics), face detection (Camera Brick usage), ASR Cloud (added in App Lab 0.6, around April 2026)

## Working with this project

A few notes on how Kev likes to work, to save round-trips:

- He prefers paragraphs and prose over heavy bulleting in design discussions, but is fine with bullets in code-adjacent docs like this one.
- He values honest pushback over agreement — if a suggested approach has a real downside, name it.
- He's an experienced PM and IT contractor; assume engineering literacy. He's also a maker and educator; assume he's thinking about the storytelling angle alongside the engineering.
- When something needs clarifying, prefer concrete options ("A or B?") over open-ended questions.
- If a decision affects the YouTube episode, flag it — that's the project's actual deliverable.
