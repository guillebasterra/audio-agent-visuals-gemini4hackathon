// TLC pigment color mapping (plan.md section 10 + Phase C2):
// - Hue follows happiness H cool -> warm across the dyes in docs/ref/tlc.jpg:
//     0.00  #b9b3c9  pale grey-violet (unhappy / B2, M1)
//     0.22  #6b6eb8  blue-violet      (E1, M2)
//     0.46  #8a3fa0  purple           (C1, L1, H1)
//     0.72  #d12f7a  magenta-pink     (G2, S1, H4)
//     1.00  #d9481c  red-orange       (G1, happiest)
// - Interpolated in OKLab, with an adaptive crowd-spread + absolute hybrid curve so
//   clustered H values use the full TLC spectrum while preserving warm-loud / cool-quiet meaning.
// - Per-agent deterministic OKLab hue jitter (+-8 deg) so no two blobs look identical.

const STOPS_HEX = [
  [0.0, "#b9b3c9"],
  [0.22, "#6b6eb8"],
  [0.46, "#8a3fa0"],
  [0.72, "#d12f7a"],
  [1.0, "#d9481c"],
];

const MAX_HUE_JITTER_RAD = (8.0 * Math.PI) / 180.0;

function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function linearToSrgb(c) {
  const clamped = Math.max(0, Math.min(1, c));
  return clamped <= 0.0031308 ? 12.92 * clamped : 1.055 * Math.pow(clamped, 1 / 2.4) - 0.055;
}

function hexToLinearRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [
    srgbToLinear(((n >> 16) & 255) / 255),
    srgbToLinear(((n >> 8) & 255) / 255),
    srgbToLinear((n & 255) / 255),
  ];
}

// Björn Ottosson's OKLab transform (MIT / public domain).
function linearRgbToOklab(r, g, b) {
  const l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
  const m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
  const s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

  const l_ = Math.cbrt(l);
  const m_ = Math.cbrt(m);
  const s_ = Math.cbrt(s);

  return [
    0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
    1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
    0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_,
  ];
}

function oklabToLinearRgb(L, a, b) {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;

  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;

  return [
    +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
}

const STOPS_OKLAB = STOPS_HEX.map(([h, hex]) => {
  const [r, g, b] = hexToLinearRgb(hex);
  return { h, lab: linearRgbToOklab(r, g, b) };
});

function agentHueJitterRad(agentId) {
  if (agentId == null) return 0;
  // Knuth multiplicative hash -> deterministic [-1, 1] per agent ID
  const h = (Math.imul((agentId + 1) | 0, 2654435761) >>> 0) / 4294967295;
  return (h * 2.0 - 1.0) * MAX_HUE_JITTER_RAD;
}

/**
 * Tracks a temporally smoothed crowd H spread (p05..p95) and stretches each agent's H
 * through a hybrid relative + absolute contrast curve so the floor spans the full TLC
 * palette while keeping absolute mood meaning (warm on happy loud drops, cool on quiet).
 */
export class CrowdColorMapper {
  constructor() {
    this.smoothLo = 0.12;
    this.smoothHi = 0.42;
    this.initialized = false;
  }

  updateSpread(agents, dtSec = 1 / 60) {
    const n = agents.length;
    if (n === 0) return;

    const vals = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      vals[i] = Math.max(0, Math.min(1, agents[i].H || 0));
    }
    vals.sort();

    const lo = vals[Math.floor(0.05 * (n - 1))];
    const hi = vals[Math.floor(0.95 * (n - 1))];

    if (!this.initialized) {
      this.smoothLo = lo;
      this.smoothHi = hi;
      this.initialized = true;
      return;
    }

    const alpha = 1.0 - Math.exp(-Math.max(0.001, dtSec) / 1.4);
    this.smoothLo += alpha * (lo - this.smoothLo);
    this.smoothHi += alpha * (hi - this.smoothHi);
  }

  mapH(H) {
    const h = Math.max(0, Math.min(1, Number.isFinite(H) ? H : 0.3));
    const span = Math.max(this.smoothHi - this.smoothLo, 0.045);
    const rel = Math.max(0, Math.min(1, (h - this.smoothLo) / span));
    // Absolute anchor: ~0.03 -> cool floor, ~0.39+ -> full red-orange euphoria
    const abs = Math.max(0, Math.min(1, (h - 0.03) / 0.36));
    // Hybrid blend: stretches relative spread while preserving global warm vs. cool mood
    const blended = 0.56 * rel + 0.44 * abs;
    // Gentle S-curve contrast around midpoint for clearer TLC pigment separation
    const sCurve = blended * blended * (3.0 - 2.0 * blended);
    return Math.max(0, Math.min(1, 0.65 * blended + 0.35 * sCurve));
  }

  absorbanceForAgent(H, agentId) {
    return happinessToAbsorbance(this.mapH(H), agentId);
  }
}

/**
 * Map normalized happiness t in [0, 1] to sRGB [r, g, b] in [0, 1] via OKLab interpolation,
 * with optional deterministic per-agent hue jitter (+-8 deg).
 */
export function happinessToSrgb(H, agentId = null) {
  const h = Math.max(0, Math.min(1, Number.isFinite(H) ? H : 0.5));
  let i = 0;
  while (i < STOPS_OKLAB.length - 2 && h > STOPS_OKLAB[i + 1].h) {
    i++;
  }
  const s0 = STOPS_OKLAB[i];
  const s1 = STOPS_OKLAB[i + 1];
  const t = s1.h > s0.h ? (h - s0.h) / (s1.h - s0.h) : 0;

  const L = s0.lab[0] + (s1.lab[0] - s0.lab[0]) * t;
  let a = s0.lab[1] + (s1.lab[1] - s0.lab[1]) * t;
  let b = s0.lab[2] + (s1.lab[2] - s0.lab[2]) * t;

  if (agentId !== null) {
    const theta = agentHueJitterRad(agentId);
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);
    const aRot = a * cosT - b * sinT;
    const bRot = a * sinT + b * cosT;
    a = aRot;
    b = bRot;
  }

  const [lr, lg, lb] = oklabToLinearRgb(L, a, b);
  return [linearToSrgb(lr), linearToSrgb(lg), linearToSrgb(lb)];
}

/**
 * Convert pigment sRGB color c to Beer-Lambert absorbance a = -ln(max(c, 0.04)).
 */
export function happinessToAbsorbance(H, agentId = null) {
  const [r, g, b] = happinessToSrgb(H, agentId);
  return [
    -Math.log(Math.max(r, 0.04)),
    -Math.log(Math.max(g, 0.04)),
    -Math.log(Math.max(b, 0.04)),
  ];
}
