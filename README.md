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
- **Phase E2 — Energy lives at the front (`server/config.py`, `server/state.py`, `server/physics.py`):**
  - **Wider DJ hot zone:** Replaced the single point `(0.5, 0)` with segment `x ∈ [0.30, 0.70], y = 0` (`MAX_DIST_TO_DJ = 1.044`) so front rows spread naturally across the booth instead of clumping at one point.
  - **Ring-attenuated `local_music`:** Multiplied `music_energy` by per-ring gains `(1.0, 0.90, 0.75, 0.60, 0.45)` everywhere in physics, energy, reward, and motion amplitude so the front is physically loudest.
  - **Music charge & rare exhaustion:** Added front music recharge `dE/dt += g · v · local_music · w` (`ENERGY_G = 0.048`, calibrated for real-MP3 `w ≈ 1.4–1.9`), cut front-crush drain in half (`ENERGY_A = 0.010`), shifted fatigue to pushing (`ENERGY_B = 0.016`) and sustained dancing (`ENERGY_D = 0.008`), and lowered exhaustion trigger/recovery to `E ≤ 0.05` / `E ≥ 0.25`.
  - **Softened reward & traits + hype/energy contagion:** Rebalanced reward to `u = (0.4 + 0.6·E)·(k·v·w + s·(1−ρ)·local_music + m·v·local_music)/(k + s + m)` (`m = 0.60`, `U_NORM_DIVISOR = 1.15`), softened base traits (`diehard 0.80/0.35`, `between 0.55/0.55`, `dancer 0.45/0.70`), raised `H` contagion to `κ = 0.25`, and added neighbor energy contagion (`0.10/s`).
  - **Section-driven movement scaling:** Smoothed house-ear section probabilities (`α = 0.45`, ~2–3 ticks) scale movement without overriding Decisions: `build` raises forward approach & speed, `drop` packs tightest with max kick bounce (including pushers), and `breakdown` damps forward pushing and gently loosens front/mid rows backward (`loud/quiet mean_v` gap `+0.16` to `+0.24`).
  - **Visible proximity, size & blobby squash (`web/src/plate.js`, `web/src/shaders/composite.frag.glsl`, `web/src/palette.js`):** Per-agent physical/visual radii (`AGENT_RADIUS = 0.030 × archetype [1.2/1.0/0.85 ±10%] × proximity [0.80..1.25] × energy [0.85..1.00]`, also used in collisions and radius-aware crowding `ρ`), kick pulse scaled by `local_music·E`, front-to-back OKLab chroma/density vibrancy gradient, 2-mode spring-damper jelly wobble + velocity stretch + DJ-rail squish spring while keeping non-merging power-diagram facets.
  - **Plain-word prompts, collapsed chat & HUD (`server/decide.py`, `server/negotiate.py`, `server/tick.py`, `web/src/chat.js`, `web/src/ui.js`):** Replaced decimal energy in agent prompts with `fresh / ok / tired`, collapsed multiple retreating agents per tick into `"N agents retreating"`, added the top-left mode banner with change highlight and bottom-left control legend with `R` reset (which cancels in-flight ticks cleanly).
- **Phase E3 — Compact floor, probabilistic yields & clean ask bubbles (`server/config.py`, `server/decide.py`, `web/src/plate.js`, `web/src/overlay.js`):**
  - **Compact walkable floor:** Shrunk physical bounds to `x ∈ [0.15, 0.85]`, `y ∈ [0.03, 0.70]` (isotropic `0.70 × 0.70` floor, rescaling `RING_EDGES = (0.14, 0.28, 0.42, 0.56)`, `DJ_SEG = [0.36, 0.64]`, `MAX_DIST_TO_DJ = 0.731`, `AGENT_RADIUS = 0.027`, density grid, start positions, and wall-backed core collision reaction), scaled isotropically `1 / 0.70 = 1.43×` onto the full TLC plate so the crowd is denser, weaves naturally, and blobs read ~30% larger on screen.
  - **Probabilistic ask answers:** Sampled blocker `yes` with Bernoulli probability `p` (from `noul`, using a run-seeded RNG) instead of hard-thresholding `p ≥ 0.5`, producing a realistic mix of yields even when blockers are conflicted (`p ≈ 0.30–0.49`).
  - **Clean plate ask bubbles:** Capped on-plate ask bubbles to at most 3 translucent paper pills, offset vertically/horizontally so bubbles never overlap each other and stay clear of the DJ rail.
- **Phase F — Quota-safe runner & full analysis (`server/gemini_client.py`, `analysis/`):**
  - **429 backoff & pre-flight quota probe:** Added exponential backoff (`2s, 4s, 8s, 16s` + jitter) for rate-limit retries and a single lightweight Decisions probe before starting each sequential full-MP3 run, automatically rerunning any condition exceeding 5% failed calls.
  - **Calibrated house-ear regimes for post-hoc analysis:** Because `gemini-decision-1` spreads probability across 7 sections (`groove` often ~0.38 during drops while `p_drop` jumps from `0.07` to `0.30–0.50`), post-hoc drop-onset and breakdown detection uses calibrated house-ear probabilities (`p_drop ≥ 0.29` after a calm spell; `p_breakdown ≥ 0.18` with `w ≤ 1.12`) — strictly from the house ear with zero track labels.

## Phase F: Full-Track Experiment & Analysis

Reproduce the 3-condition comparison (`shove`, `silent`, `talk`) on the same track and seed:
```bash
# 1. Run all 3 conditions sequentially (quota-checked, full MP3, seed 42)
.venv/bin/python -m analysis.run_experiment --seed 42 --conditions shove silent talk

# 2. Generate summary.json + 5 TLC-styled plots (anticipation, welfare/fairness, ring agreement, pacing, who wins + OLS)
.venv/bin/python -m analysis.analyze --manifest out/experiment_manifest.json --out out

# 3. Extract demo highlight timestamps from the talk run
.venv/bin/python -m analysis.highlights runs/<talk_run_dir> --out out/highlights.json

# 4. Generate structured narrative report with Gemini 4 Neon (thinking_level="high")
.venv/bin/python -m analysis.neon_report out/summary.json runs/<talk_run_dir>/asks.jsonl \
  --highlights out/highlights.json --out out/report.md
```

### Results (`tracks/getuphereletsdance.mp3`, 340 s / 113 ticks, seed `42`)

| Metric | Shove (no minds) | Silent (Decisions) | Talk (Decisions + Hydrogen) |
| --- | ---: | ---: | ---: |
| API calls / failures (rate) | 113 / 0 (**0.00%**) | 651 / 0 (**0.00%**) | 1,263 / 0 (**0.00%**) |
| Tick latency p50 / p95 | 833 / 2,029 ms | 1,043 / 2,079 ms | 1,185 / 2,016 ms |
| Mean happiness `H` / Gini | 0.308 / **0.221** | 0.313 / **0.174** | 0.313 / **0.178** |
| Push gridlock (`dv < 0.01`) | **85.0%** | **32.7%** | **33.4%** |
| Drop front `H` vs back `H` | 0.472 vs 0.150 | 0.458 vs 0.173 | 0.460 vs 0.169 |
| Drop front `E` vs back `E` | 0.958 vs 0.578 | 0.953 vs 0.678 | 0.951 vs 0.682 |
| Post-breakdown drop lead time | +10.5 s | **+12.8 s** | **+12.0 s** |
| Top-20 front diehard share (drops) | 66.9% | 59.7% | 60.9% |
| Asks resolved / acceptance | — | — | **604** / **33.3%** (201 yields) |
| Reciprocity (`P(yes \| prior yield)`) | — | — | **42.1%** vs 32.4% base (**+9.7 pp**) |

Generated outputs in `out/`: `summary.json`, `highlights.json`, `report.md` (Gemini 4 Neon), `anticipation.png`, `welfare_fairness.png`, `ring_agreement.png`, `pacing.png`, `who_wins.png`.

## Credits

- **WebGL ping-pong splat trail architecture**: inspired by Pavel Dobryakov's [WebGL Fluid Simulation](https://github.com/PavelDoGreat/WebGL-Fluid-Simulation) (MIT License).
- **OKLab color-space interpolation**: Björn Ottosson's [OKLab](https://bottosson.github.io/posts/oklab/) perceptual color transform (MIT / public domain).
- **Polynomial smooth minimum (`smin`) for rounded packed blob facets**: Inigo Quilez's [smooth minimum](https://iquilezles.org/articles/smin/).
- **Font**: IBM Plex Mono.
