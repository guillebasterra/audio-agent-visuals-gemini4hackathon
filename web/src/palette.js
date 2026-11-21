// Happiness H in [0, 1] -> pigment sRGB color interpolated in OKLab (plan.md section 10).

const STOPS_HEX = [
  [0.0, "#b9b3c9"], // pale grey-violet
  [0.3, "#6f6fb8"], // blue-violet
  [0.55, "#8a3fa0"], // purple
  [0.75, "#d12f7a"], // magenta
  [1.0, "#d9481c"], // red-orange
];

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

/**
 * Map happiness H in [0, 1] to sRGB [r, g, b] in [0, 1] via OKLab interpolation.
 */
export function happinessToSrgb(H) {
  const h = Math.max(0, Math.min(1, Number.isFinite(H) ? H : 0.5));
  let i = 0;
  while (i < STOPS_OKLAB.length - 2 && h > STOPS_OKLAB[i + 1].h) {
    i++;
  }
  const s0 = STOPS_OKLAB[i];
  const s1 = STOPS_OKLAB[i + 1];
  const t = s1.h > s0.h ? (h - s0.h) / (s1.h - s0.h) : 0;

  const L = s0.lab[0] + (s1.lab[0] - s0.lab[0]) * t;
  const a = s0.lab[1] + (s1.lab[1] - s0.lab[1]) * t;
  const b = s0.lab[2] + (s1.lab[2] - s0.lab[2]) * t;

  const [lr, lg, lb] = oklabToLinearRgb(L, a, b);
  return [linearToSrgb(lr), linearToSrgb(lg), linearToSrgb(lb)];
}

/**
 * Convert pigment sRGB color c to Beer-Lambert absorbance a = -ln(max(c, 0.04)).
 */
export function happinessToAbsorbance(H) {
  const [r, g, b] = happinessToSrgb(H);
  return [
    -Math.log(Math.max(r, 0.04)),
    -Math.log(Math.max(g, 0.04)),
    -Math.log(Math.max(b, 0.04)),
  ];
}
