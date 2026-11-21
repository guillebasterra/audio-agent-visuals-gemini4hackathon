# Agent Crowd

100 Gemini-driven agents share a virtual dancefloor and compete for the spot closest to the DJ while a house/electronic set plays. There are no track labels: everything about the music is deduced from raw audio clips by Gemini models. The crowd renders as pulsing pigment blobs on a thin-layer chromatography (TLC) plate.

Built for the **Gemini 4 NYC Hackathon** (Cerebral Valley + Google DeepMind).

## Architecture Overview

- **Backend (`server/`)**: Python 3.13 + FastAPI + asyncio (`google-genai >= 3.6.1`)
  - **Decisions API (`gemini-decision-1`, `v1beta`)**: House ear (unfiltered 15 s WAV) + 5 ring-filtered calls (8 s WAV per ring) in parallel every ~3 s tick.
  - **Gemini 4 Hydrogen (`gemini-4-hydrogen`, Interactions API `v1beta`)**: Natural-language asks between blocked agents (structured JSON output, `thinking_level="low"`).
  - **Gemini 4 Neon (`gemini-4-neon`, Interactions API `v1beta`)**: Offline post-run comparative report (`thinking_level="high"`).
- **Frontend (`web/`)**: Vite + plain WebGL2 + Canvas2D overlay (no framework).
- **Analysis (`analysis/`)**: Comparative plots, statistics, highlights, and Neon report across `shove`, `silent`, and `talk` runs.

## Setup

### Backend
```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Create `.env` in the repo root:
```env
GEMINI_API_KEY=...
MODEL_DECISIONS=gemini-decision-1
MODEL_HYDROGEN=gemini-4-hydrogen
MODEL_NEON=gemini-4-neon
```

Place one or more dev MP3s in `tracks/` (gitignored, no labels).

Run the FastAPI server:
```bash
.venv/bin/uvicorn server.main:app --reload --port 8000
```

### Frontend
```bash
npm --prefix web install
npm --prefix web run dev
```

## Phase 0: Verified API Behavior & Measured Latencies

Verified with `google-genai==3.6.1` (`v1beta`) using 16 kHz mono PCM WAV slices from `tracks/getuphereletsdance.mp3` via `.venv/bin/python -m server.phase0_check`:

- **Decisions API audio support**: **Supported directly** (`{"type": "audio", "data": "<b64>", "mime_type": "audio/wav"}`). No Hydrogen perception fallback needed.
- **Audio token rate**: `30 tokens/sec` (`240` tokens for 8 s WAV, `450` tokens for 15 s WAV). Decisions output tokens = `0`.
- **Native async SDK**: Both `client.aio.decisions.create` and `client.aio.interactions.create` are natively async.
- **Hydrogen thinking level**: `thinking_level="low"` works (`"minimal"` returns HTTP 400 `THINKING_LEVEL_MINIMAL is not supported`).

| Call | Model | Input | Questions / Output | Measured Latency | Tokens |
| --- | --- | --- | --- | --- | --- |
| Decisions (single ring sample) | `gemini-decision-1` | 8 s WAV (256 KB) + text features | 5 questions (`choice`, `noul`, `score`) | **595–644 ms** | 924 in (240 audio, 684 text), 0 out |
| Decisions (full tick: 1 house ear + 5 rings × 20 agents, parallel) | `gemini-decision-1` | 1× 15 s WAV + 5× 8 s WAV | 3 + 5×43 = **218 questions** across 6 parallel calls | **1,541 ms** wall time (502–1,506 ms/call) | 886 (house) + 5×5,902 (rings), 0 out |
| Hydrogen structured + audio (`thinking_level="low"`) | `gemini-4-hydrogen` | 8 s WAV + prompt | Structured JSON (`AskTest`) | **4,024–5,159 ms** | 720 in (240 audio), 388–413 thought, 65–69 out |
| Hydrogen structured text-only (`thinking_level="low"`) | `gemini-4-hydrogen` | Text prompt | Structured JSON (`AskTest`) | **2,314 ms** | 464 in, 294 thought, 65 out |

FastAPI `/ws` WebSocket echo was verified for both JSON text frames and 8,000-byte binary Int16 PCM frames (250 ms @ 16 kHz mono) via both `TestClient` and a live `uvicorn` server.

## Visual Mapping (TLC Plate)

- **Orientation:** The renderer displays the **DJ at the bottom** (pencil line labeled `DJ`) and the **origin line with lane letters (`B C E G H L M P S Ś`) at the top**; pigment trails advect upward, away from the DJ. Physics coordinates remain `y = 0` at the DJ and `y = 1` at the back.
- **Happiness → Hue (`web/src/palette.js`):** Interpolated in OKLab across the TLC dye pigments from `docs/ref/tlc.jpg`, cool to warm:
  - `0.00` `#b9b3c9` pale grey-violet (unhappy / faded `B2`, `M1`)
  - `0.22` `#6b6eb8` blue-violet (`E1`, `M2`)
  - `0.46` `#8a3fa0` purple (`C1`, `L1`, `H1`)
  - `0.72` `#d12f7a` magenta-pink (`G2`, `S1`, `H4`)
  - `1.00` `#d9481c` red-orange (`G1`, happiest)
- **Adaptive contrast + absolute mood:** Each agent's `H` is mapped through a hybrid curve (`56%` relative to the temporally smoothed crowd `p05–p95` spread + `44%` absolute anchor, with a mild S-curve) so the crowd uses the full pigment range even when `H` clusters, while a happy drop still trends warm overall and a quiet breakdown cools down.
- **Per-agent hue jitter:** Deterministic `±8°` OKLab hue rotation per agent ID so adjacent blobs never look identical.
- **Energy → Density:** Energy `E` scales Beer–Lambert pigment density (`energyDensity = 0.18 + 1.32·E`), making tired agents pale and thin while fresh agents render rich and saturated.
- **Live Chat Feed (`C` toggle):** A narrow 280 px semi-transparent right-edge panel (`IBM Plex Mono 11 px`) shifts the square TLC plate cleanly left so it never covers any agent. Lines are priority-throttled to <= 6/tick (asks/answers → exhaustion/give-ups → `BOOTH:` house ear → `ring k hears:` → hard pushes), prefixed with each agent's current pigment color, and highlight the agent (and other party for asks/answers) on hover with a thin ring.

## Decisions made

- **Squared mass formula (`server/physics.py`):** Replaced linear collision mass with `m = (MASS_BASE + MASS_PUSH_SCALE · pushiness · E)²` (`0.2`, `2.2`) so fresh, hard-pushing diehards reliably cut through lighter or tired crowd members while exhausted agents yield easily.
- **Tangential shoulder-slip & jostle (`server/physics.py`):** Purely radial collision separation caused head-on deadlocks and catapulted lighter agents onto the DJ rail. Added a first-iteration mass-weighted tangential shoulder-slip + jostle impulse on contact so pushers weave past blockers and blocked agents visibly jostle instead of freezing.
- **Soft spring collisions & larger bodies (`server/physics.py`, `web/src/plate.js`):** Increased `AGENT_RADIUS` to `0.036` and replaced hard one-shot overlap projection with a 3-substep spring penalty (`SOFT_SPRING_RATE = 0.14`, allowing up to ~25% temporary overlap) paired with stronger shader squash so front rows visibly compress and bulge on drops.
- **Ornstein–Uhlenbeck idle wander & anti-stall (`server/physics.py`):** Added an always-on per-agent OU wander force scaled by `music_energy` and `E` (minimum ~30%, larger sway for dancers, smaller for diehards, plus a kick-synced bob on `dance`), soft wall repulsion, and a gentle nudge if an agent stays near zero speed for > 2 s so no agent ever freezes against walls or at the back.
- **Adaptive TLC color stretch & hue jitter (`web/src/palette.js`):** Because raw `H` values cluster mid-track, each agent's `H` maps through a hybrid contrast curve (`56%` relative to temporally smoothed crowd `p05–p95` + `44%` absolute anchor) plus a deterministic `±8°` OKLab hue rotation per agent ID so the floor spans the full TLC pigment range while keeping global mood meaningful.
- **Pushiness remap (`server/decide.py`):** Under the squared mass formula, raw Decisions `score / 2.0` made `silent` noticeably tamer than `shove` on seed 42 (0–120 s loud sections: `mean_v = 0.490` vs `0.556`, mean push magnitude `0.395` vs `0.792`, mean tick displacement `0.0613` vs `0.0787`). Remapping active pushers' `0..2` rubric score (`gently = 0.40, firmly = 0.70, hard = 1.00`) raised `silent` push magnitude to `0.646`, loud `mean_v` to `0.510` (peak `0.635` vs `shove` `0.617`), and full-MP3 front-row diehard share to `0.605` (and `0.652` in `talk`), making all three conditions comparable.
- **Explicit trait semantics & isolated question context (`server/decide.py`):** Because Decisions questions are evaluated independently, each agent's state and traits (`keen` = front-row desire, `space` = room-to-dance desire) are included in both `a{id}_action` and `a{id}_push`, and `ACTION_CRITERIA` clarifies what each action does so `gemini-decision-1` differentiates archetypes cleanly without any scripted rules.
- **Separate API semaphores & `TICK_SECONDS = 3.0` (`server/gemini_client.py`):** Separated `decide()` and `interact_structured()` concurrency semaphores so background Hydrogen asks (~2.8 s p50) never queue behind or stall critical-path Decisions ticks. Full-MP3 runs hit `p50 = 1,083 ms / p95 = 1,657 ms` (`silent`) and `p50 = 851 ms / p95 = 1,752 ms` (`talk`) with `0` rate-limit errors, keeping `TICK_SECONDS = 3.0`.

## Credits

- **WebGL ping-pong splat trail architecture**: inspired by Pavel Dobryakov's [WebGL Fluid Simulation](https://github.com/PavelDoGreat/WebGL-Fluid-Simulation) (MIT License).
- **OKLab color-space interpolation**: Björn Ottosson's [OKLab](https://bottosson.github.io/posts/oklab/) perceptual color transform (MIT / public domain).
- **Polynomial smooth minimum (`smin`) for rounded packed blob facets**: Inigo Quilez's [smooth minimum](https://iquilezles.org/articles/smin/).
- **Font**: IBM Plex Mono.
