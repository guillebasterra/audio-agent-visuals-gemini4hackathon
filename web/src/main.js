// Agent Crowd frontend entry point (Phase A: TLC WebGL2 renderer + AudioWorklet + ?fake=1).

import { PlateRenderer } from "./plate.js";
import { OverlayRenderer } from "./overlay.js";
import { AudioEngine } from "./audio.js";
import { FakeCrowd } from "./fake.js";
import { CrowdUI } from "./ui.js";

const params = new URLSearchParams(window.location.search);
const isFakeMode = params.get("fake") === "1";

const plateCanvas = document.getElementById("plate");
const overlayCanvas = document.getElementById("overlay");

const plate = new PlateRenderer(plateCanvas);
const overlay = new OverlayRenderer(overlayCanvas);

// Frame interpolation state between the last two 30 Hz physics frames (plan.md section 10)
let prevFrame = null;
let currFrame = null;
let prevFrameMs = 0;
let currFrameMs = 0;
let interpAgents = [];

let condition = "talk";
let asksById = new Map();
let asksLastTick = 0;

function unpackPhysicsAgents(rawAgents) {
  const out = new Array(rawAgents.length);
  for (let i = 0; i < rawAgents.length; i++) {
    const row = rawAgents[i];
    out[i] = {
      id: row[0],
      x: row[1],
      y: row[2],
      vx: row[3],
      vy: row[4],
      E: row[5],
      rho: row[6],
      H: row[7],
      action: row[8],
      pressure: row[9],
    };
  }
  return out;
}

function handleInit(msg) {
  if (msg.lane_labels) {
    overlay.setLaneLabels(msg.lane_labels);
  }
}

function handlePhysics(msg) {
  const now = performance.now();
  prevFrame = currFrame;
  prevFrameMs = currFrameMs;
  currFrame = unpackPhysicsAgents(msg.agents || []);
  currFrameMs = now;
  if (!prevFrame) {
    prevFrame = currFrame;
    prevFrameMs = now - 33.33;
  }
}

function handleDecision(msg) {
  const now = performance.now();
  if (msg.condition) condition = msg.condition;
  const newAsks = msg.asks || [];
  asksLastTick = newAsks.length;

  for (const a of newAsks) {
    asksById.set(a.id, {
      ...a,
      createdAt: now,
      updatedAt: now,
    });
  }
  for (const ans of msg.answers || []) {
    const existing = asksById.get(ans.id);
    if (existing) {
      existing.answer = ans.answer;
      existing.updatedAt = now;
    }
  }
  // Prune old asks after fadeout
  for (const [id, ask] of asksById.entries()) {
    if (now - ask.updatedAt > 5000) {
      asksById.delete(id);
    }
  }
}

function getInterpolatedAgents(nowMs) {
  if (!currFrame) return [];
  if (!prevFrame || prevFrame.length !== currFrame.length) return currFrame;

  const span = Math.max(16, Math.min(100, currFrameMs - prevFrameMs));
  const alpha = Math.max(0, Math.min(1, (nowMs - currFrameMs) / span + 1.0));
  // Smoothly interpolate from prevFrame -> currFrame over one physics interval (~33.3 ms)
  const t = Math.max(0, Math.min(1, (nowMs - currFrameMs) / 33.333));

  if (interpAgents.length !== currFrame.length) {
    interpAgents = new Array(currFrame.length);
    for (let i = 0; i < currFrame.length; i++) {
      interpAgents[i] = { ...currFrame[i] };
    }
  }

  for (let i = 0; i < currFrame.length; i++) {
    const a0 = prevFrame[i];
    const a1 = currFrame[i];
    const dst = interpAgents[i];
    dst.id = a1.id;
    dst.x = a0.x + (a1.x - a0.x) * t;
    dst.y = a0.y + (a1.y - a0.y) * t;
    dst.vx = a0.vx + (a1.vx - a0.vx) * t;
    dst.vy = a0.vy + (a1.vy - a0.vy) * t;
    dst.E = a0.E + (a1.E - a0.E) * t;
    dst.rho = a0.rho + (a1.rho - a0.rho) * t;
    dst.H = a0.H + (a1.H - a0.H) * t;
    dst.action = a1.action;
    dst.pressure = a0.pressure + (a1.pressure - a0.pressure) * t;
  }
  void alpha;
  return interpAgents;
}

let fakeCrowd = null;

// Audio engine (file playback + shared AnalyserNode + 16 kHz mono Int16 AudioWorklet)
const audio = new AudioEngine({
  onClock: (clockMsg) => {
    ui.setPlaying(clockMsg.playing);
    if (fakeCrowd && clockMsg.playing) {
      fakeCrowd.setMusicEnergy(clockMsg.dsp.energy);
    }
  },
  onPcmChunk: (_pcmBuffer) => {
    // Phase C forwards this 8,000-byte binary chunk over WebSocket.
  },
  onControl: (ctrl) => {
    if (ctrl.action === "seek") {
      plate.clearTrails();
    }
  },
});

let fileLoaded = false;
async function ensureDefaultTrackLoaded() {
  if (fileLoaded) return true;
  const tracks = ui.tracks.length > 0 ? ui.tracks : await ui.refreshTracks();
  if (tracks.length === 0) return false;
  const first = tracks[0];
  ui.selectedTrack = first;
  await audio.loadFile(`/tracks/${encodeURIComponent(first)}`, first);
  fileLoaded = true;
  return true;
}

const ui = new CrowdUI({
  onTogglePlay: async () => {
    await ensureDefaultTrackLoaded();
    await audio.togglePlay();
  },
  onCondition: (cond) => {
    condition = cond;
    if (fakeCrowd) fakeCrowd.setCondition(cond);
  },
  onToggleLabels: () => {
    overlay.showLabels = !overlay.showLabels;
  },
  onSelectFile: async (url, label) => {
    await audio.loadFile(url, label);
    fileLoaded = true;
    await audio.play();
  },
});

if (isFakeMode) {
  fakeCrowd = new FakeCrowd({
    seed: 42,
    onInit: handleInit,
    onPhysics: handlePhysics,
    onDecision: handleDecision,
  });
  fakeCrowd.start();
}

function handleResize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  plate.resize(w, h);
  overlay.resize(w, h);
}
window.addEventListener("resize", handleResize);
handleResize();

// 60 fps render loop (never blocks on network)
let lastRafMs = performance.now();
let fpsSmooth = 60;
let framesRendered = 0;

function frame(nowMs) {
  const dtSec = Math.min(0.05, Math.max(0.001, (nowMs - lastRafMs) / 1000));
  lastRafMs = nowMs;
  fpsSmooth = fpsSmooth * 0.92 + (1 / dtSec) * 0.08;
  framesRendered += 1;

  const dsp = audio.update(nowMs);
  const agents = getInterpolatedAgents(nowMs);

  // In ?fake=1 before user hits Space, optional gentle synthetic pulse if ?beat=1; otherwise real DSP
  let musicEnergy = dsp.energy;
  let kickEnv = audio.kickEnvelope;
  if (isFakeMode && !audio.playing && params.get("beat") === "1") {
    const beatPhase = ((nowMs / 1000) * 2.1) % 1.0;
    musicEnergy = 0.65;
    kickEnv = Math.exp(-beatPhase / 0.25);
  }

  plate.render(agents, {
    musicEnergy,
    kickEnvelope: kickEnv,
    timeSec: nowMs / 1000,
    dtSec,
  });

  overlay.render(agents, Array.from(asksById.values()), nowMs);

  if (agents.length > 0) {
    let sumH = 0;
    let sumE = 0;
    for (let i = 0; i < agents.length; i++) {
      sumH += agents[i].H;
      sumE += agents[i].E;
    }
    ui.updateStats({
      meanH: sumH / agents.length,
      meanE: sumE / agents.length,
      asksPerTick: asksLastTick,
      fps: fpsSmooth,
      condition,
      playing: audio.playing,
      energy: musicEnergy,
      pcmRate: audio.getPcmChunksPerSec(),
      pcmTotal: audio.pcmChunksTotal,
      track: audio.currentLabel,
    });
  }

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);

// Diagnostic hook for headless visual & AudioWorklet verification
window.__agentCrowd = {
  isFakeMode,
  plate,
  overlay,
  audio,
  ui,
  ensureDefaultTrackLoaded,
  getMetrics: () => ({
    fps: +fpsSmooth.toFixed(1),
    framesRendered,
    agentCount: currFrame ? currFrame.length : 0,
    useFloatTrails: plate.useFloatTrails,
    playing: audio.playing,
    workletReady: audio.workletReady,
    pcmChunksTotal: audio.pcmChunksTotal,
    lastChunkBytes: audio.lastChunkBytes,
    kicksTotal: audio.kicksTotal,
    dsp: audio.dsp,
    asksActive: asksById.size,
  }),
};
