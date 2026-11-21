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

## Credits

- **WebGL ping-pong splat trail architecture**: inspired by Pavel Dobryakov's [WebGL Fluid Simulation](https://github.com/PavelDoGreat/WebGL-Fluid-Simulation) (MIT License).
- **OKLab color-space interpolation**: Björn Ottosson's [OKLab](https://bottosson.github.io/posts/oklab/) perceptual color transform (MIT / public domain).
- **Polynomial smooth minimum (`smin`) for rounded packed blob facets**: Inigo Quilez's [smooth minimum](https://iquilezles.org/articles/smin/).
- **Font**: IBM Plex Mono.
