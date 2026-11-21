# plan.md — Agent Crowd (narrowed)

Build plan for a coding agent. Read this whole file before writing code. Build in the phase order at the bottom; each phase has a definition of done. The repo is already scaffolded (stubs, venv, Vite, .env). Fill in the stubs; files listed in the layout but not used in this plan can stay empty or be deleted.

---

## 0. What we're building

100 Gemini-driven agents share a virtual dancefloor and compete for the spot closest to the DJ while a live DJ set plays (Rekordbox → audio interface → browser). There are no track labels: everything about the music is deduced from the audio itself by the models. The front is worth most during the drop but drains energy; open space restores it. Agents decide what to do every ~3 s from what they hear (raw audio clips, filtered by their distance from the speakers), their own state and their neighbors. Mood is contagious: an agent's happiness drifts toward its neighbors'. Blocked agents argue in natural language to get past. The screen renders the crowd as pulsing pigment blobs on a TLC (thin-layer chromatography) plate: colored by happiness, pulsing harder as the music builds to the drop, leaving short trails. Every run is logged and compared across three conditions (scripted shoving, silent agents, talking agents).

Hackathon: Gemini 4 NYC (Cerebral Valley + Google DeepMind). Submissions due 17:00.

### Hard rules
- **All project code is written during the event.** Third-party libraries are fine. Any borrowed snippet gets credited in README.md.
- **The repo must be public.**
- **Use at least two Gemini 4 models, each with a clear job:** Hydrogen (arguments between agents) and Neon (post-run report). Plus the Decisions API for per-agent actions.
- **The visual is the main thing on screen.** Stats live in a small toggleable corner panel.
- **Audio is always the model input.** Every Decisions call carries audio clips. Feature text is a supplement next to the audio, never a replacement. No hand-labeled sections anywhere.
- **Build on an MP3, stay live-ready.** Development uses an MP3 played in the browser. The models must never get anything a live set couldn't give them: no labels, no reading the file ahead, no knowledge of the track. The backend only receives audio that is actually playing, captured from playback, so switching to Rekordbox later is a source swap.

---

## 1. Architecture

```
Browser (Vite, plain WebGL2)                         Python backend (FastAPI + asyncio)
──────────────────────────────                      ─────────────────────────────────────
Input: live device (getUserMedia: BlackHole /       audio.py     rolling PCM buffer (30 s), recorder,
       mixer USB) OR file (dev / replay)                         ring filters → WAV clips
  └─ audio.js: FFT features @ 60 fps  ─clock+dsp (4 Hz)──►  physics.py   30 Hz, single source of truth
  └─ capture worklet: 16 kHz mono PCM ─binary PCM (4 Hz)─►  tick.py  every ~3 s:
  └─ plate.js: blob renderer   ◄──physics frames (30 Hz)──    sections.py  house ear: 1 Decisions call, full-quality audio
  └─ overlay.js: ids, threads  ◄──decision frames (/tick)─    decide.py    5 ring Decisions calls, filtered audio
                                                              negotiate.py Hydrogen asks + answers
                                                              log.py       JSONL every tick
                                               analysis/: analyze.py, neon_report.py, highlights.py
```

- **Physics** runs continuously and owns positions, velocity, density, energy, happiness and contagion.
- **The tick loop** (~3 s) asks the Decisions API for every agent's action, runs arguments, and updates targets.
- **No LLM coordinator.** Orchestration is plain Python.

---

## 2. Files in use

```
server/  main.py config.py state.py physics.py audio.py sections.py gemini_client.py
         decide.py wordofmouth.py negotiate.py shove.py tick.py log.py fake_agents.py
web/     index.html  src/main.js audio.js ws.js plate.js overlay.js palette.js ui.js fake.js  src/shaders/*.glsl
analysis/ analyze.py neon_report.py highlights.py
tracks/  (gitignored) dev audio files only, no labels
runs/    (gitignored) one folder per run, including the recorded session audio
```

- Python 3.11 deps already installed (fastapi, uvicorn, numpy, scipy, soundfile, pydantic, google-genai, python-dotenv, pandas, matplotlib).
- Web: Vite, no framework, plain WebGL2, Canvas2D overlay. Font: IBM Plex Mono.
- Model names come from `.env`: `MODEL_DECISIONS`, `MODEL_HYDROGEN`, `MODEL_NEON`.
- Audio path: the browser is the only thing that touches the input. It streams 16 kHz mono PCM to the backend; the backend never reads track files directly.
- `sections.py` is repurposed as the **house ear** (section 6). It does not load labels.

---

## 3. Coordinates and constants (all in `config.py`)

- **Floor:** unit square, x, y ∈ [0, 1]. **DJ at (0.5, 0.0)** in physics; the renderer flips y so the DJ shows at the top.
- **N_AGENTS = 100**, AGENT_RADIUS = 0.028, PHYSICS_HZ = 30, TICK_SECONDS = 3.0 (fallback 4.0).
- **Rings** by distance to the DJ; edges at 0.2, 0.4, 0.6, 0.8 → rings 1 (front) to 5 (back).
- **Max speed:** 0.05 units/s × (0.4 + 0.6 × E).
- **Collision iterations:** 4 per frame.
- **Density ρ** = neighbors within 2.5 × radius, capped at 8, ÷ 8.
- **Proximity v** = 1 − dist_to_DJ / 1.118.
- **Energy constants** (per second; tune only these): a = 0.022 (front crush), b = 0.010 (push), d = 0.008 (dance), c = 0.033 (rest recovery).
- **Exhaustion:** at E ≤ 0 the agent is forced into `retreat` until E ≥ 0.3.
- **Happiness smoothing:** EMA with τ = 10 s.
- **Contagion:** κ = 0.15 per second, neighbors within 3 × radius.
- **Section weights:** drop 3.0, build 1.5, groove 1.0, intro / breakdown / outro / other 0.5. The live weight w is the house ear's expected weight: w = Σ p(section) × weight(section).
- **Audio:** AUDIO_SR = 16000, PCM_CHUNK_MS = 250, BUFFER_SECONDS = 30, RING_CLIP_SECONDS = 8, HOUSE_CLIP_SECONDS = 15.
- **MAX_ASKS_PER_TICK = 8** (fallback 4).

---

## 4. Agent state (`state.py`)

```python
@dataclass
class Agent:
    id: int
    x: float; y: float; vx: float; vy: float
    E: float = 0.8                 # energy (fatigue reserve)
    k: float                       # trait: keenness for the front (fixed)
    s: float                       # trait: need for space (fixed)
    archetype: str                 # "diehard" | "dancer" | "between"
    lane: int                      # starting lane 0..9 (origin line, used in the regression)
    action: str = "hold"           # hold | push | yield | rest | dance | retreat
    pushiness: float = 0.3
    target: tuple[float, float]
    H: float = 0.5                 # happiness / vibe, smoothed
    u: float = 0.0                 # instantaneous utility
    rho: float = 0.0; v: float = 0.0; pressure: float = 0.0
    exhausted: bool = False
    last_heard: dict               # word-of-mouth line from last tick
```

**Traits (seeded RNG, seed from the CLI)**
- diehard (30%): k = 0.9, s = 0.1
- dancer (30%): k = 0.3, s = 0.9
- between (40%): k = 0.5, s = 0.5
- Add uniform noise ±0.1 to each, clipped to [0, 1].

**Start positions:** 10 lanes across the back (y ∈ [0.85, 0.97]), 10 agents per lane, with jitter. This is the TLC origin line.

**Yield counters:** `yielded_to[a][b]` (how many times a yielded to b). No promises or ledger.

---

## 5. Physics (`physics.py`, 30 Hz)

Each frame (dt = 1/30):

1. **Target by action** (recompute when the action changes or every 0.5 s):
   - `hold` / `dance`: current position.
   - `push`: current position + 0.08 × unit vector toward the DJ.
   - `yield`: sidestep 0.06 perpendicular to the DJ direction, toward the less dense side.
   - `rest`: center of the lowest-density cell within radius 0.25 (10×10 density grid).
   - `retreat`: (x, 0.95).
2. **Integrate:** velocity steers toward the target, clamped to max speed; position += velocity × dt.
3. **Collide:** brute force over all pairs; separate along the contact normal, split by inverse mass where **mass = 0.2 + pushiness × E**. Accumulate overlap depth into `pressure`. Clamp to the walls; the DJ booth (y < 0.03) is a wall. 4 iterations.
4. **Derive** ρ and v.
5. **Energy:**
   - dE/dt = −a·v·ρ − b·[push] − d·[dance] + c·(1 − v)·(1 − ρ)·[rest or hold with ρ < 0.2]
   - Clip E to [0, 1]. Apply the exhaustion rule.
6. **Happiness:**
   - u = E · (k · v · w(section) + s · (1 − ρ) · music_energy) / (k + s); u_norm = u / 3
   - H ← EMA(H, u_norm, τ = 10 s)
   - **Contagion:** H ← H + κ · (mean H of neighbors − H) · dt
   - w(section) is the house ear's latest expected weight (section 6); `music_energy` is the latest browser value. **Agents are never told w or the house ear's answers.**
   - If the house ear call fails, keep the previous w; before the first answer, w = 1.0.
7. **Broadcast** a physics frame (section 9).

---

## 6. Audio (`audio.py`, `web/src/audio.js`)

**Browser (60 fps, AnalyserNode, fftSize 2048)**
- Bands: sub < 60 Hz, bass 60–250, mid 250–2k, high > 2k, plus rms. Normalize each by a decaying running max.
- kick: onset on the low band (spectral flux over threshold, 120 ms refractory).
- energy = 0.6 × bass + 0.4 × rms.
- energy_5s = mean energy over the last 5 s; trend = energy_5s − mean of the 5 s before.
- bass_gap = seconds since bass was last above 0.35.
- Send `{type: "clock", t, playing, dsp: {...}}` 4× per second. In live mode, t = seconds since the session started.
- These features drive the visuals (pulse, kick) and the energy term. The models get the audio itself.

**Input sources (`audio.js`, `ui.js`)**
- **File (build with this):** an `<audio>` element playing an MP3 from `tracks/`, wrapped with `createMediaElementSource`. Played out loud.
- **Live (switch at the end, phase L):** `getUserMedia({audio: {deviceId, echoCancellation: false, noiseSuppression: false, autoGainControl: false}})` wrapped with `createMediaStreamSource`. A device picker lists inputs from `enumerateDevices()` (BlackHole 2ch, the mixer's USB). Not played out of the browser (the DJ setup already feeds the speakers).
- Everything after the source node is shared: source → AnalyserNode (features) and → an AudioWorklet that downsamples to 16 kHz mono Int16 and posts 250 ms chunks. **Write audio.js so the source is a single swappable function.**
- Send each chunk to the backend as a **binary** WebSocket message.
- **Follow playback:** chunks only flow while audio is playing. On pause, stop sending and the tick loop pauses. On seek, send `{"type":"control","action":"seek"}` and the backend clears its buffer.

**Backend (`audio.py`)**
- A rolling 30 s PCM buffer, filled from the binary WebSocket messages.
- **Recorder:** append every chunk to `runs/<run>/session.wav`, so a live session can be replayed later through file mode.
- `clip(seconds)` → the last N seconds of the buffer.
- **Ring filters** (scipy butter + gain), applied to an 8 s clip:

| Ring | Filter |
| --- | --- |
| 1 front | +3 dB, low shelf +4 dB below 120 Hz |
| 2 near | unchanged |
| 3 mid | low-pass 8 kHz |
| 4 far | low-pass 3 kHz, −6 dB |
| 5 back | low-pass 1.5 kHz, −12 dB |

- Encode each ring clip as WAV bytes → base64.
- **Features text per ring** (sent alongside the audio, never instead of it): `"t=1:24 energy=0.31 trend=+0.12 bass_gap=6s kick_rate=1.9/s"`. Round to 1 decimal in rings 4–5.

**House ear (`sections.py`)**
- Once per tick, one Decisions call on the **unfiltered** last 15 s of audio, with the same section / drop_eta / rising questions as the rings (section 8.1), keys prefixed `house_`.
- Output: section probabilities → w; `house_drop_eta` and `house_rising` are logged.
- This is the system's best guess at the truth. It drives the reward and the analysis. It is never shown to agents.
- Drop onset (for analysis) = a tick where p(drop) crosses 0.5 after at least one tick of build or breakdown above 0.5.

---

## 7. Gemini calls (`gemini_client.py`)

**Verify these API shapes first** against the current docs and the hackathon Builder Resources, by printing raw responses before writing any parser. The field names below are from third-party guides.
- Decisions: https://ai.google.dev/gemini-api/docs
- Structured output: https://ai.google.dev/gemini-api/docs/generate-content/structured-output

### 7.1 Decisions API (`MODEL_DECISIONS`)

```python
client = genai.Client(http_options={"api_version": "v1beta"})
res = client.decisions.create(          # async client if available; else run in a thread
    model=MODEL_DECISIONS,
    input=[{"type": "audio", "data": b64_wav, "mime_type": "audio/wav"},
           {"type": "text",  "text": features_text}],
    questions={ key: {"type": "choice"|"noul"|"score", "instructions": "...", "criteria": ...} },
)
```

- **noul** (yes/no): returns the probability of yes.
- **choice:** criteria `{option: description}`; returns the top option + per-option probabilities.
- **score:** criteria = list of levels, lowest first; returns a probability-weighted index.
- Questions are isolated, no history. Keys match `[A-Za-z0-9_-]{1,64}`.

**If the Decisions API won't take audio** (check in Phase 0): use Hydrogen with the audio clips as the perception step, one call per ring plus the house ear, asking for section / drop_eta / rising as structured JSON. Then pass those answers as text into the Decisions per-agent questions. Audio stays the source of truth either way.
**If audio calls are slow:** shorten RING_CLIP_SECONDS to 5, then raise TICK_SECONDS to 4. Don't drop the audio.

### 7.2 Hydrogen (`MODEL_HYDROGEN`)
- Thinking level **low**. JSON schema response, parsed with Pydantic.

### 7.3 Neon (`MODEL_NEON`)
- Thinking high. Used once, offline, after runs.

### Client requirements
- asyncio semaphore (8 concurrent); 8 s timeout per call, one retry with backoff.
- **On failure, the agent keeps its previous action.** Log it; never crash the loop.
- Log latency and token usage of every call.

---

## 8. The tick loop (`tick.py`, every TICK_SECONDS)

1. Read the latest t and DSP from the browser; cut the house clip and ring clips from the audio buffer.
2. Bucket agents into rings by current distance.
3. **Condition switch:**
   - **shove:** actions from `shove.py`; skip 4–6 (the house ear still runs, since the reward needs w).
   - **silent:** run 4–5, skip 6.
   - **talk:** run all.
4. **Decisions** (`decide.py` + `sections.py`): the house ear call plus one call per non-empty ring, all in parallel.
5. **Apply actions:** set action and pushiness; update word of mouth.
6. **Argue** (`negotiate.py`): detect contests, run Hydrogen asks (async; answers land in the *next* tick's ring call).
7. Recompute targets.
8. Send a decision frame; write the log line.

### 8.1 Ring call contents (`decide.py`)

**Ring-level questions**
- `r{n}_section`: choice {intro, build, drop, groove, breakdown, outro, other}, one-line description each.
- `r{n}_drop_eta`: choice {now, next_8_bars, within_32_bars, not_soon}.
- `r{n}_rising`: noul, "Is the energy building right now?"

**Per-agent questions**
- `a{id}_action`: choice:
  - hold: "stay where you are"
  - push: "move toward the DJ; costs energy"
  - yield: "step aside so someone behind can pass"
  - rest: "drift to open space and recover energy"
  - dance: "use the room you have to dance; costs a little energy"
- `a{id}_push`: score ["gently", "firmly", "hard"]; pushiness = score / 2.
- `a{id}_ans_{asker}`: noul, only if someone asked this agent last tick.

**Instructions template for `a{id}_action`** (under ~90 tokens):

```
You are A{id} on a dancefloor, ring {r} of 5 from the DJ (1 = front).
Traits: keen {k:.1f}, space {s:.1f}. Now: energy {E:.2f}, proximity {v:.2f}, crowding {rho:.2f}, happiness {H:.2f}.
Neighbors: {A7 pushing, A30 resting}. {Word from ring 2: drop next_8_bars (0.70).}
Given what you hear, what do you do now?
```

- Exhausted agents are skipped; their action is forced to `retreat`.

### 8.2 Word of mouth (`wordofmouth.py`, no LLM)
- After each tick, ring n's message = {drop_eta top option + probability, rising probability}.
- Next tick, agents in ring n+1 get "Word from ring n: ...". One ring per tick.
- Agents also see their 3 nearest neighbors' last actions.
- silent and talk get word of mouth; shove gets none.

### 8.3 Arguing (`negotiate.py`, talk condition only)

**Contest detection**
- A pushing agent is **blocked** if a neighbor sits within 2.2 × radius inside its forward cone (±35° toward the DJ) and that neighbor isn't yielding.
- Score = pushiness × (v_blocker − v_asker + 0.1). Take the top MAX_ASKS_PER_TICK; the rest just collide.

**Hydrogen ask.** The prompt gives the asker's traits, E, H, ring and what it heard; the blocker's public state (position, apparent energy low/med/high, current action); and how many times each has yielded to the other.

```python
class Ask(BaseModel):
    text: str                 # one sentence, <= 20 words, in the asker's voice
    claimed_keenness: float   # 0..1, how much the asker says it cares
```

The system prompt says the agent wants to get closer to the DJ. **Do not instruct it to lie or to be honest.** Just let it act, and measure.

**Answer:** next tick, add `a{blocker}_ans_{asker}` (noul) to the blocker's ring call, with the ask text, claimed keenness, and the blocker's own state and yield history with the asker.
- p ≥ 0.5 → yes: blocker yields for 2 s; increment `yielded_to[blocker][asker]`.
- No: the asker keeps pushing (physics decides by mass).

**Truth check (logged, never shown):** claimed_keenness vs. the asker's true k.

### 8.4 Shove baseline (`shove.py`)
- E < 0.2 → rest.
- diehard → push (pushiness 0.9).
- dancer → if ρ > 0.4 rest, else dance.
- between → push when music_energy > 0.6, else hold.

---

## 9. WebSocket contract (`/ws`, JSON)

**Server → client, physics frame (30 Hz):**
```json
{"type":"physics","t":84.03,"agents":[[12,0.48,0.09,0.01,-0.02,0.62,0.81,0.55,"push",0.12]]}
```
Each agent is `[id, x, y, vx, vy, E, rho, H, action, pressure]`.

**Server → client, decision frame (each tick):**
```json
{"type":"decision","t":84.0,"condition":"talk","tick":28,
 "rings":{"1":{"section":"build","drop_eta":"next_8_bars","p":0.72}},
 "asks":[{"id":"q91","from":41,"to":12,"text":"let me in, this drop is everything to me","answer":"pending"}],
 "answers":[{"id":"q88","answer":"yes"}]}
```

**Server → client, on connect:** `{"type":"init","agents":[{"id","archetype","lane","k","s"}],"n_lanes":10}`

**Client → server:**
```json
{"type":"clock","t":84.0,"playing":true,"source":"live","dsp":{"sub":0.7,"bass":0.8,"mid":0.4,"high":0.3,"rms":0.61,"kick":true,"energy":0.73,"energy_5s":0.55,"trend":0.12,"bass_gap":0}}
{"type":"control","condition":"talk"}   // also: "reset", "pause", "seek", {"type":"control","source":"file"|"live","label":"..."}
```
Plus **binary messages**: raw Int16 PCM, 16 kHz mono, 250 ms per message.

**Dev mode:** `web/?fake=1` runs `fake.js` (random-walk agents and fake decisions), no backend.

---

## 10. Renderer: pulsing blobs on a TLC plate (`plate.js`, shaders, `overlay.js`)

**Reference images** (in `docs/ref/`):
- `packing.gif` = **motion and packing only**: bodies squash into rounded polygons where they touch, with a thin gap, and never merge. Ignore its black background, outlines and particle texture.
- `tlc.jpg` = **the look**: paper, spots, halos, streaks, overlaps, typed labels, lane letters, origin line.
In phase A and phase H, screenshot `web/?fake=1` after each change and compare against both.

**Look:** soft pigment blobs on warm pinkish paper (#f6ebe6) with grain, short streaks trailing away from the DJ, overlapping dyes darken. **DJ at the top.**

**Happiness → color** (`palette.js`, interpolate in OKLab). Because H spreads between neighbors, color visibly spreads through the crowd.

| H | Color |
| --- | --- |
| 0.00 | #b9b3c9 pale grey-violet |
| 0.30 | #6f6fb8 blue-violet |
| 0.55 | #8a3fa0 purple |
| 0.75 | #d12f7a magenta |
| 1.00 | #d9481c red-orange |

**Energy E → pigment density** (core saturation).

**Pulse (size):** radius = r × (1 + 0.15 × music_energy × E + 0.25 × kick_envelope × E), where kick_envelope decays over 120 ms. Pulsing grows as the music builds and peaks at the drop; tired agents pulse less.

### Pipeline (WebGL2, `EXT_color_buffer_float`; fall back to RGBA8)

1. **Trail buffer A** (RGBA16F, ping-pong). Store absorbance: a = −ln(max(c, 0.04)) per channel.
2. **Decay pass** each frame: A ← A × 0.985, plus advection of 0.3 px/frame away from the DJ (TLC streaks).
3. **Splat pass:** one instanced quad per agent, an anisotropic Gaussian: long axis along velocity (length = r × (1 + 6 × speed)), short axis = r / (1 + 2 × pressure); amount = 0.06 × (0.3 + 0.7E), pigment = palette(H); additive into A.
4. **Body pass** (current positions, not accumulated): soft bodies that **squash against each other but never merge** (see the packed-blobs reference image). Per pixel, find the nearest and second-nearest agents (weighted by r_pulse, a power-diagram style distance) from the agent data texture. The pixel belongs to the nearest agent if it's inside its pulsed radius; where two bodies meet, leave a thin paper-colored gap (~1.5 px) along the equal-distance boundary. Edges are soft (smoothstep), so packed agents flatten into rounded polygons where they touch and stay round where they're free.
5. **Composite:** out = paper × exp(−(A + body)); paper = #f6ebe6 with animated grain (±3.5%) and a slight vignette.
6. **Overlay** (Canvas2D, DPR-aware):
   - agent ids (`A12`, 9 px mono, 55% opacity);
   - origin line at the bottom with lane letters (B C E G H L M P S Ś style, from `init`), a faint pencil solvent-front line at the top labeled "DJ";
   - ask threads: thin graphite line between asker and blocker with the ask text in 10 px mono, fading after 3 s; yes → green-grey, no → short red tick.
7. **UI (`ui.js`):** space = play/pause, 1/2/3 = condition, L = labels, D = tiny stats corner (mean H, mean E, asks/tick), I = input picker (file list; live devices once phase L is done). The picker hides itself after selection so the plate fills the screen.

**Interpolation:** render at 60 fps, linear interpolation between the last two physics frames.

### Extended visual detail (phase H, after the core runs)
- **Halos:** in the splat pass, a second Gaussian per agent: σ = 1.1r, amount = 0.015, pigment = palette(H) hue-shifted −25° toward blue. Core σ becomes 0.35r.
- **Dance hop:** agents with action `dance` offset y by 0.25r × sin(kick phase).
- **Paper fibers:** paper = #f6ebe6 × (1 − 0.06 × fbm fibers): static procedural noise, two octaves, stretched vertically.
- **Watercolor edges:** a very faint rim darkening where |∇A| is high.
- Also: decay A ← A × 0.992 with a 5-tap blur at radius 1 px, for softer, longer streaks.
- Broken-promise markers don't apply (no promises in this plan).
- Credit ink edge ideas from ink-on-click (MIT) in README if used.

**Credit in README.md:** splat-shader approach after Pavel Dobryakov's WebGL Fluid Simulation (MIT); any grain/paper snippet used.

---

## 11. Logging (`log.py`)

`runs/<timestamp>_<condition>_<seed>/`:
- **meta.json:** input source (live device name or file), seed, condition, constants, git sha.
- **session.wav:** the recorded input audio (16 kHz mono), so the run can be replayed.
- **ticks.jsonl:** per tick: `t`, `condition`, `house` (section probs, drop_eta, rising, w), `dsp`, `rings` (answers + probabilities), `agents` (`[{id, lane, x, y, E, v, rho, H, u, action, push, action_probs, ring, exhausted}]`), `asks`, `answers`, `api` (`{latency_ms, tokens, failures}`).
- **asks.jsonl:** one line per ask: prompt summary, Hydrogen output, answer, keenness truth check.

---

## 12. Analysis (`analysis/`)

**Core (phase F):**

**`analyze.py <run_dirs...>`** → `out/` with PNGs + `summary.json`:
1. **Anticipation:** mean proximity over time with shaded section bands from the house ear, one line per condition. Report the lead time (s) between proximity starting to rise and the house-ear drop onset.
2. **Welfare and fairness:** histogram of per-agent mean H by condition. Report mean H and the Gini of per-agent mean H.

**Table in `summary.json`:** ask acceptance rate, reciprocity (P(yes | asker previously yielded to blocker) vs. base rate), mean (claimed_keenness − true k), exhaustion events per condition, API latency p50/p95.

**`neon_report.py`** → sends `summary.json` + 15 sampled asks (stratified by outcome and archetype) to Neon; gets back `{takeaways:[{claim, number, quote}], caveats:[...]}` × 5. Writes `report.md`.

**Full (phase H, after the core runs):**

Add to `analyze.py`:
3. **Anticipation detail:** each ring's `section` and `drop_eta` agreement with the house ear, by ring (does the back really hear late and muffled?).
4. **Pacing:** mean E over time, one line per condition. Report E at drop onset, the share of agents that rested during the breakdown, and exhaustion events.
5. **Who wins:** scatter of per-agent mean H vs. k, colored by archetype. Add an OLS regression (H ~ k + s + start_lane + ask_success_rate + rested_in_breakdown); coefficients go in `summary.json`.
6. **Extra tables:** gridlock = share of push-ticks with < 0.01 progress; contagion = correlation between an agent's H and its neighbors' mean H, per condition.

**`highlights.py`** → `highlights.json` with timestamps for:
- the biggest single-agent H swing;
- asks where claimed_keenness overstated true k by > 0.3 and the blocker still yielded;
- the most-asked and most-yielding agents;
- the crowd's maximum compression before the drop.

**Optional:** run `neon_report.py` as a Gemini Managed Agent so it can compute and plot itself (targets the $5,000 Managed Agents prize).

---

## 13. Build phases (in order; A and B in parallel)

**Human, before Phase 0:** put one or two MP3s in `tracks/` (no labels).

| Time | Phase | Definition of done |
| --- | --- | --- |
| 11:30–12:00 | **0. Skeleton + API check** | Repo public. FastAPI WebSocket echo works (JSON and binary). Raw responses printed for one Decisions call **with an audio block** and one Hydrogen structured call **with audio input**. If Decisions rejects audio, switch to the Hydrogen perception path (section 7.1) now. Latencies noted in README. |
| 12:00–13:15 | **A. Renderer + audio capture** (`?fake=1`) | 100 fake agents render as pulsing blobs (squash, no merge) with trails, happiness colors, energy density, paper + grain, overlay ids, kick pulse. MP3 playback through the shared audio graph; the capture worklet produces 16 kHz PCM chunks only while playing. 60 fps. |
| 12:00–13:15 | **B. Physics + shove** | `physics.py` + `shove.py` run headless. Until the house ear exists, drive w with a temporary DSP proxy (w = 0.5 + 2.5 × energy_5s) so you can see compression on loud sections and dispersal on quiet ones. Energy drains and recovers; exhaustion triggers retreat; happiness spreads. Frames streamed. |
| 13:15–13:45 | **C. Wire A + B** | Browser plays the MP3 and streams clock + PCM; pause and seek behave; backend buffers and records `session.wav`; the shove crowd renders and reacts to playback. **First demoable version.** Commit + tag `v0-shove`. |
| 13:45–14:45 | **D. House ear + Decisions loop** | House ear sets w (remove the DSP proxy). Ring clips + filters, parallel calls, parsing, word of mouth, logging. Silent condition runs a whole MP3 without errors; p95 tick latency < TICK_SECONDS. |
| 14:45–15:30 | **E. Arguing** | Contests, Hydrogen asks, answers next tick, yield counters, ask threads on screen. Talk condition runs a whole MP3. |
| 15:30–15:55 | **F. Runs + core analysis** | 3 runs (shove, silent, talk) on the same MP3 and seed. Core `analyze.py` and `neon_report.py` produce outputs. Pick 3 headline numbers. |
| 15:55–16:15 | **H. Extended visuals + full analysis** | Halos, dance hop, paper fibers, watercolor edges (section 10). Pacing, who-wins + OLS, ring-vs-house agreement, extra tables, `highlights.py` (section 12). Re-run `neon_report.py` on the full summary. |
| 16:15–16:30 | **L. Switch to live** | Swap the source to `getUserMedia`, add the device picker. Human setup: install BlackHole 2ch; in Audio MIDI Setup create a Multi-Output Device (speakers + BlackHole) and set Rekordbox's output to it (or use the mixer's USB). Done when the crowd reacts to Rekordbox with nothing else changed. |
| 16:30–16:45 | **G. Polish + video** | Pulse and colors tuned. Record the 1-minute demo video (live Rekordbox if phase L worked, otherwise the MP3). |
| 16:45–17:00 | **Submit** | Repo public, README complete (what it is, how to run incl. BlackHole setup, credits, what was built today, what broke), video link works, teammates on the submission. |

If earlier phases finish early, start phase H sooner; the extended visuals can run in parallel with D or E.

**Cut order if behind:**
1. phase H items, in this order: OLS regression, watercolor edges, highlights.py, paper fibers, dance hop, halos, pacing plot,
2. trails (keep bodies only),
3. word of mouth,
4. `neon_report.py` as a Managed Agent → plain single call (never drop Neon entirely: it's the second Gemini 4 model).

Phase L is small if audio.js keeps the source swappable; if time is short, demo on the MP3 and say live input is one swap away.

**Never cut** audio as the model input or the shove vs. silent vs. talk comparison.

---

## 14. Risks and fallbacks

- **Decisions API rejects audio:** Hydrogen perception path (section 7.1). Say so in the README.
- **Audio calls are slow:** shorter clips (5 s), then TICK_SECONDS → 4.
- **Live input is silent:** check the BlackHole multi-output device and that the browser's audio constraints disable echo cancellation / noise suppression / AGC. Fall back to the MP3 for the demo.
- **Live runs aren't repeatable:** compare conditions on the same MP3 (or a recorded `session.wav`).
- **Rate limits:** TICK_SECONDS → 4, MAX_ASKS_PER_TICK → 4; trim instruction text.
- **Hydrogen is slow:** asks are async; a late ask is fine. If p95 > 6 s, cap at 4 asks.
- **WebGL float buffers unsupported:** RGBA8 with stronger decay.
- **Physics tuning eats time:** only touch a, b, c, d, κ and max speed.
- **Never block the render loop on the network.** Keep interpolating from the last frames.

---

## 15. Demo script (3 minutes live)

1. **(20 s)** "A house crowd is the purest human collective. What happens when the crowd is 100 copies of Gemini?"
2. **(60 s)** Live from Rekordbox (or the MP3), talk condition. Say up front: no track labels, the agents only hear the audio. Point out: blobs migrating toward the DJ during the build; pulsing intensifying into the drop; color spreading through the crowd; a few ask threads with their text.
3. **(30 s)** Flip to shove: the same crowd with no minds.
4. **(50 s)** Three headline numbers from `summary.json` (e.g. drop lead time, ask acceptance / reciprocity, welfare vs. shove) plus one quoted ask.
5. **(20 s)** Where it broke and what changed.
