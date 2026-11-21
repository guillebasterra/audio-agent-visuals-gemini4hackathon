// ?fake=1 local crowd simulator (plan.md sections 9 & 10).
// Produces Section-9-compatible init, 30 Hz physics frames, and 3 s decision frames with zero backend.

import { AGENT_RADIUS } from "./plate.js";

const LANE_LABELS = ["B", "C", "E", "G", "H", "L", "M", "P", "S", "Ś"];
const FAKE_ASK_LINES = [
  "let me in, this drop is everything to me",
  "just two steps forward, my friend is up front",
  "hey mind if I slide past for the peak?",
  "been waiting all night for this track",
  "trade spots? I'll let you back in after",
  "please, one meter closer to the speakers",
];

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class FakeCrowd {
  constructor({ seed = 42, onInit, onPhysics, onDecision } = {}) {
    this.rng = mulberry32(seed);
    this.onInit = onInit || (() => {});
    this.onPhysics = onPhysics || (() => {});
    this.onDecision = onDecision || (() => {});

    this.condition = "talk";
    this.t = 0;
    this.tick = 0;
    this.askSeq = 1;
    this.pendingAsks = [];
    this.agents = [];
    this._timer = null;
    this._lastDecisionT = 0;
    this._musicEnergy = 0.55;

    this._initAgents();
    // Pre-warm ~15 s so opening ?fake=1 immediately shows packed front blobs squashing
    // near the DJ booth, free-floating mid/back spots, downward trails, and full OKLab H/E range.
    for (let i = 0; i < 450; i++) {
      this._step(1 / 30, 0.72 + 0.22 * Math.sin(i * 0.035));
    }
  }

  _initAgents() {
    const archetypes = [];
    for (let i = 0; i < 30; i++) archetypes.push("diehard");
    for (let i = 0; i < 30; i++) archetypes.push("dancer");
    for (let i = 0; i < 40; i++) archetypes.push("between");

    // Fisher-Yates shuffle
    for (let i = archetypes.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      [archetypes[i], archetypes[j]] = [archetypes[j], archetypes[i]];
    }

    const clamp01 = (v) => Math.max(0, Math.min(1, v));
    this.agents = [];

    for (let id = 0; id < 100; id++) {
      const lane = id % 10;
      const arch = archetypes[id];
      let baseK = 0.5;
      let baseS = 0.5;
      if (arch === "diehard") {
        baseK = 0.9;
        baseS = 0.1;
      } else if (arch === "dancer") {
        baseK = 0.3;
        baseS = 0.9;
      }
      const k = clamp01(baseK + (this.rng() * 0.2 - 0.1));
      const s = clamp01(baseS + (this.rng() * 0.2 - 0.1));
      const archSize = arch === "diehard" ? 1.2 : arch === "dancer" ? 0.85 : 1.0;
      const sizeMult = archSize * (0.9 + this.rng() * 0.2);

      const laneX = (lane + 0.5) / 10;
      const x = Math.max(0.04, Math.min(0.96, laneX + (this.rng() - 0.5) * 0.06));
      const y = 0.85 + this.rng() * 0.11;

      this.agents.push({
        id,
        lane,
        archetype: arch,
        k,
        s,
        sizeMult,
        r: AGENT_RADIUS * sizeMult,
        localMusic: 0.4,
        x,
        y,
        vx: 0,
        vy: 0,
        E: 0.45 + this.rng() * 0.52,
        H: 0.25 + this.rng() * 0.5,
        rho: 0,
        v: 0,
        pressure: 0,
        action: arch === "diehard" ? "push" : arch === "dancer" ? "dance" : "hold",
        pushiness: arch === "diehard" ? 0.85 : 0.35,
        phase: this.rng() * Math.PI * 2,
      });
    }
  }

  setMusicEnergy(energy) {
    this._musicEnergy = Math.max(0, Math.min(1, energy));
  }

  setCondition(cond) {
    this.condition = cond;
  }

  start() {
    this.onInit({
      type: "init",
      n_lanes: 10,
      lane_labels: LANE_LABELS,
      dj_segment: [0.3, 0.7],
      floor: { x_min: 0.0, x_max: 1.0, y_min: 0.0, y_max: 1.0, dj_y: 0.03 },
      agents: this.agents.map((a) => ({
        id: a.id,
        archetype: a.archetype,
        lane: a.lane,
        k: +a.k.toFixed(3),
        s: +a.s.toFixed(3),
      })),
    });

    this.onPhysics(this._buildPhysicsFrame());
    this._emitFakeDecision();

    // 30 Hz physics frames
    this._timer = window.setInterval(() => {
      this._step(1 / 30, this._musicEnergy);
      this.onPhysics(this._buildPhysicsFrame());
      if (this.t - this._lastDecisionT >= 3.0) {
        this._lastDecisionT = this.t;
        this._emitFakeDecision();
      }
    }, 1000 / 30);
  }

  stop() {
    if (this._timer) {
      clearInterval(this._timer);
      this._timer = null;
    }
  }

  _step(dt, musicEnergy) {
    this.t += dt;
    const agents = this.agents;
    const n = agents.length;
    const djX = 0.5;
    const djY = 0.0;
    const w = 0.5 + 2.5 * Math.max(0.2, musicEnergy);

    // 1. Choose targets & steer
    for (let i = 0; i < n; i++) {
      const a = agents[i];
      const dxDj = djX - a.x;
      const dyDj = djY - a.y;
      const dDj = Math.hypot(dxDj, dyDj) || 1;
      const ux = dxDj / dDj;
      const uy = dyDj / dDj;

      // Simple expressive crowd roles so front packs tight, mid dances, tired retreats
      if (a.E < 0.18) {
        a.action = "retreat";
        a.pushiness = 0.2;
      } else if (a.action === "retreat" && a.E < 0.35) {
        a.action = "retreat";
      } else if (a.archetype === "diehard") {
        a.action = "push";
        a.pushiness = 0.85;
      } else if (a.archetype === "dancer") {
        a.action = a.rho > 0.45 ? "rest" : "dance";
        a.pushiness = 0.3;
      } else {
        a.action = musicEnergy > 0.48 && a.E > 0.35 ? "push" : "hold";
        a.pushiness = a.action === "push" ? 0.55 : 0.35;
      }

      let tx = a.x;
      let ty = a.y;
      const wobble = 0.028 * Math.sin(this.t * 1.4 + a.phase);
      const wobbleY = 0.02 * Math.cos(this.t * 1.1 + a.phase * 1.3);

      if (a.action === "push") {
        tx = a.x + ux * 0.1 + wobble * 0.45;
        ty = a.y + uy * 0.1;
      } else if (a.action === "retreat") {
        tx = a.x + wobble * 0.8;
        ty = 0.92;
      } else if (a.action === "rest") {
        // Drift away from DJ center toward open flanks/back
        tx = a.x + (a.x < 0.5 ? -0.05 : 0.05) + wobble;
        ty = Math.min(0.92, a.y + 0.06);
      } else if (a.action === "dance") {
        tx = a.x + wobble * 1.1;
        ty = Math.max(0.28, Math.min(0.84, a.y + wobbleY - 0.015));
      } else {
        tx = a.x + wobble * 0.5;
        ty = Math.max(0.22, a.y - 0.01 + wobbleY * 0.5);
      }

      const maxSpd = 0.055 * (0.4 + 0.6 * a.E);
      const toX = tx - a.x;
      const toY = ty - a.y;
      const dist = Math.hypot(toX, toY);
      const dvx = dist > 1e-4 ? (toX / dist) * Math.min(maxSpd, dist * 3.0) : 0;
      const dvy = dist > 1e-4 ? (toY / dist) * Math.min(maxSpd, dist * 3.0) : 0;

      a.vx += (dvx - a.vx) * Math.min(1, dt * 8.0);
      a.vy += (dvy - a.vy) * Math.min(1, dt * 8.0);
      a.x += a.vx * dt;
      a.y += a.vy * dt;
      a.pressure = 0;
    }

    // 2. Collision separation (4 iterations) with inverse mass
    const minDist = AGENT_RADIUS * 2.02;
    const minDist2 = minDist * minDist;
    for (let iter = 0; iter < 4; iter++) {
      for (let i = 0; i < n; i++) {
        const ai = agents[i];
        const invMi = 1 / (0.2 + ai.pushiness * ai.E);
        for (let j = i + 1; j < n; j++) {
          const aj = agents[j];
          const dx = aj.x - ai.x;
          const dy = aj.y - ai.y;
          const d2 = dx * dx + dy * dy;
          if (d2 >= minDist2 || d2 < 1e-10) continue;
          const d = Math.sqrt(d2);
          const overlap = minDist - d;
          const nx = dx / d;
          const ny = dy / d;
          const invMj = 1 / (0.2 + aj.pushiness * aj.E);
          const invSum = invMi + invMj;
          const wi = invMi / invSum;
          const wj = invMj / invSum;
          ai.x -= nx * overlap * wi;
          ai.y -= ny * overlap * wi;
          aj.x += nx * overlap * wj;
          aj.y += ny * overlap * wj;
          ai.pressure += overlap;
          aj.pressure += overlap;
        }
      }
      for (let i = 0; i < n; i++) {
        const a = agents[i];
        a.x = Math.max(AGENT_RADIUS, Math.min(1 - AGENT_RADIUS, a.x));
        a.y = Math.max(0.03 + AGENT_RADIUS, Math.min(1 - AGENT_RADIUS, a.y));
      }
    }

    // 3. Density rho, proximity v, energy E, happiness H + contagion
    const rhoR2 = (2.5 * AGENT_RADIUS) ** 2;
    const contR2 = (3.0 * AGENT_RADIUS) ** 2;
    const prevH = new Float32Array(n);
    for (let i = 0; i < n; i++) prevH[i] = agents[i].H;

    const emaAlpha = 1 - Math.exp(-dt / 10.0);
    for (let i = 0; i < n; i++) {
      const a = agents[i];
      let closeCnt = 0;
      let hSum = 0;
      let hCnt = 0;
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const dx = agents[j].x - a.x;
        const dy = agents[j].y - a.y;
        const d2 = dx * dx + dy * dy;
        if (d2 <= rhoR2) closeCnt++;
        if (d2 <= contR2) {
          hSum += prevH[j];
          hCnt++;
        }
      }
      a.rho = Math.min(8, closeCnt) / 8;
      const segX = Math.max(0.3, Math.min(0.7, a.x));
      const distDj = Math.hypot(a.x - segX, a.y - djY);
      a.v = Math.max(0, Math.min(1, 1 - distDj / 1.04403));
      const ringGain = distDj < 0.2 ? 1.0 : distDj < 0.4 ? 0.9 : distDj < 0.6 ? 0.75 : distDj < 0.8 ? 0.6 : 0.45;
      a.localMusic = musicEnergy * ringGain;
      a.r = AGENT_RADIUS * a.sizeMult * (0.8 + 0.45 * a.v) * (0.85 + 0.15 * a.E);

      const isPush = a.action === "push" ? 1 : 0;
      const isDance = a.action === "dance" ? 1 : 0;
      const isRecover =
        a.action === "rest" || a.action === "retreat" || (a.action === "hold" && a.rho < 0.2)
          ? 1
          : 0;
      const dE =
        -0.015 * a.v * a.rho -
        0.008 * isPush -
        0.006 * isDance +
        0.036 * (1 - a.v) * (1 - a.rho) * isRecover;
      a.E = Math.max(0, Math.min(1, a.E + dE * dt));

      const u = (a.E * (a.k * a.v * w + a.s * (1 - a.rho) * Math.max(0.45, musicEnergy))) / (a.k + a.s);
      // Map fake-mode target H across the full [0, 1] OKLab spectrum:
      // front drop diehards -> red-orange/magenta (0.78..0.98), happy dancers -> purple/magenta (0.52..0.82),
      // crowded/waiting -> blue-violet (0.25..0.48), low-energy/back -> pale grey-violet (0.04..0.22).
      let targetH;
      if (a.E < 0.25) {
        targetH = 0.06 + 0.35 * a.E;
      } else if (a.archetype === "diehard") {
        targetH = Math.min(1.0, 0.30 + 1.05 * a.v * (0.55 + 0.45 * a.E));
      } else if (a.archetype === "dancer") {
        targetH = Math.min(0.88, 0.18 + 0.72 * (1 - a.rho) * (0.4 + 0.6 * Math.max(0.4, musicEnergy)));
      } else {
        targetH = Math.max(0.08, Math.min(0.85, u / 1.1));
      }
      let nextH = a.H + emaAlpha * (targetH - a.H);
      if (hCnt > 0) {
        nextH += 0.14 * (hSum / hCnt - prevH[i]) * dt;
      }
      a.H = Math.max(0, Math.min(1, nextH));
    }
  }

  _buildPhysicsFrame() {
    return {
      type: "physics",
      t: +this.t.toFixed(3),
      agents: this.agents.map((a) => [
        a.id,
        +a.x.toFixed(4),
        +a.y.toFixed(4),
        +a.vx.toFixed(4),
        +a.vy.toFixed(4),
        +a.E.toFixed(3),
        +a.rho.toFixed(3),
        +a.H.toFixed(3),
        a.action,
        +a.pressure.toFixed(4),
        +a.r.toFixed(4),
        +a.v.toFixed(4),
        +a.localMusic.toFixed(4),
      ]),
    };
  }

  _emitFakeDecision() {
    this.tick += 1;
    const answers = this.pendingAsks.map((p) => ({
      id: p.id,
      from: p.from,
      to: p.to,
      answer: this.rng() < 0.55 ? "yes" : "no",
    }));

    const newAsks = [];
    if (this.condition === "talk") {
      // Pick up to 2 well-spaced front/mid pushing pairs so ask labels stay clean & readable
      const pushers = this.agents.filter((a) => a.action === "push" && a.y < 0.65);
      const picked = [];
      for (let tries = 0; tries < 20 && newAsks.length < 2 && pushers.length > 0; tries++) {
        const asker = pushers[Math.floor(this.rng() * pushers.length)];
        if (picked.some((p) => Math.hypot(p.x - asker.x, p.y - asker.y) < 0.22)) continue;
        let blocker = null;
        let bestD = 0.09;
        for (const other of this.agents) {
          if (other.id === asker.id || other.y >= asker.y) continue;
          const d = Math.hypot(other.x - asker.x, other.y - asker.y);
          if (d < bestD) {
            bestD = d;
            blocker = other;
          }
        }
        if (blocker) {
          picked.push(asker);
          const qid = `q${this.askSeq++}`;
          newAsks.push({
            id: qid,
            from: asker.id,
            to: blocker.id,
            text: FAKE_ASK_LINES[this.askSeq % FAKE_ASK_LINES.length],
            answer: "pending",
          });
        }
      }
    }

    this.pendingAsks = newAsks.map((a) => ({ id: a.id, from: a.from, to: a.to }));

    const tired = this.agents.find((a) => a.action === "retreat");
    const pusher = this.agents.find((a) => a.action === "push" && a.pushiness > 0.7);
    const notable = [];
    if (tired) {
      notable.push({ kind: "exhausted", agent: tired.id, text: `A${tired.id} exhausted, retreating` });
    }
    if (pusher) {
      notable.push({ kind: "push_hard", agent: pusher.id, text: `A${pusher.id} pushes hard` });
    }

    this.onDecision({
      type: "decision",
      t: +this.t.toFixed(2),
      condition: this.condition,
      tick: this.tick,
      house: {
        section: this._musicEnergy > 0.62 ? "drop" : "build",
        p: 0.81,
        drop_eta: this._musicEnergy > 0.62 ? "now" : "next_8_bars",
        w: this._musicEnergy > 0.62 ? 3.0 : 1.5,
      },
      rings: {
        1: { section: "drop", drop_eta: "now", p: 0.84 },
        3: { section: "build", drop_eta: "next_8_bars", p: 0.72 },
      },
      asks: newAsks,
      answers,
      notable,
    });
  }
}
