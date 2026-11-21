// DPR-aware Canvas2D overlay: TLC solvent front ("DJ"), origin line + lane letters,
// agent IDs (A12), and ask negotiation threads (plan.md section 10).

import { computePlateRectCss } from "./plate.js";

const DEFAULT_LANE_LABELS = ["B", "C", "E", "G", "H", "L", "M", "P", "S", "Ś"];
const ASK_LIFETIME_MS = 3000;

export class OverlayRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.cssWidth = 1;
    this.cssHeight = 1;
    this.dpr = 1;
    this.plateRect = { ox: 0, oy: 0, size: 1 };
    this.laneLabels = DEFAULT_LANE_LABELS;
    this.showLabels = true;
    this.resize(canvas.clientWidth || window.innerWidth, canvas.clientHeight || window.innerHeight);
  }

  setLaneLabels(labels) {
    if (Array.isArray(labels) && labels.length > 0) {
      this.laneLabels = labels;
    }
  }

  resize(cssWidth, cssHeight) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.cssWidth = cssWidth;
    this.cssHeight = cssHeight;
    this.dpr = dpr;
    this.canvas.width = Math.max(2, Math.round(cssWidth * dpr));
    this.canvas.height = Math.max(2, Math.round(cssHeight * dpr));
    this.plateRect = computePlateRectCss(cssWidth, cssHeight);
  }

  _toScreen(xPhys, yPhys) {
    const { ox, oy, size } = this.plateRect;
    return [ox + xPhys * size, oy + yPhys * size];
  }

  render(agents, asks = [], nowMs = performance.now()) {
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.cssWidth, this.cssHeight);

    const { ox, oy, size } = this.plateRect;

    // 1. Subtle TLC plate framing + faint pencil solvent-front line at top (y = 0.03, "DJ")
    ctx.save();
    ctx.strokeStyle = "rgba(65, 55, 58, 0.22)";
    ctx.lineWidth = 0.9;
    const djY = oy + 0.03 * size;
    ctx.beginPath();
    ctx.moveTo(ox, djY);
    ctx.lineTo(ox + size, djY);
    ctx.stroke();

    ctx.font = '500 10px "IBM Plex Mono", monospace';
    ctx.fillStyle = "rgba(55, 46, 50, 0.58)";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText("DJ", ox + size * 0.5, djY - 4);

    // 2. Faint pencil origin line near bottom (y = 0.955) + lane letters
    const originY = oy + 0.955 * size;
    ctx.strokeStyle = "rgba(65, 55, 58, 0.26)";
    ctx.lineWidth = 0.9;
    ctx.beginPath();
    ctx.moveTo(ox, originY);
    ctx.lineTo(ox + size, originY);
    ctx.stroke();

    const nLanes = this.laneLabels.length;
    ctx.font = '400 11px "IBM Plex Mono", monospace';
    ctx.fillStyle = "rgba(70, 58, 60, 0.52)";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i < nLanes; i++) {
      const lx = ox + ((i + 0.5) / nLanes) * size;
      // Small vertical pencil tick on origin line
      ctx.beginPath();
      ctx.moveTo(lx, originY - 2.5);
      ctx.lineTo(lx, originY + 2.5);
      ctx.stroke();
      ctx.fillText(this.laneLabels[i], lx, originY + 5);
    }
    ctx.restore();

    // Build quick lookup map by agent id for ask threads
    const byId = new Map();
    for (let i = 0; i < agents.length; i++) {
      byId.set(agents[i].id, agents[i]);
    }

    // 3. Agent IDs (A12, 9px mono, 55% opacity)
    if (this.showLabels) {
      ctx.save();
      ctx.font = '500 9px "IBM Plex Mono", monospace';
      ctx.fillStyle = "rgba(28, 22, 26, 0.55)";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let i = 0; i < agents.length; i++) {
        const a = agents[i];
        const [sx, sy] = this._toScreen(a.x, a.y);
        ctx.fillText(`A${a.id}`, sx, sy);
      }
      ctx.restore();
    }

    // 4. Ask threads: thin graphite line, 10px mono text, fading after 3 s;
    //    yes -> green-grey, no -> short red tick.
    if (asks && asks.length > 0) {
      ctx.save();
      ctx.font = '400 10px "IBM Plex Mono", monospace';
      // Keep only freshest visible asks and prevent text label overlap
      const visible = asks
        .map((ask) => ({
          ask,
          age: nowMs - (ask.updatedAt || ask.createdAt || nowMs),
        }))
        .filter((item) => item.age <= ASK_LIFETIME_MS + 700)
        .sort((a, b) => a.age - b.age);

      const placedBoxes = [];

      for (let i = 0; i < visible.length; i++) {
        const { ask, age } = visible[i];
        const fade =
          age <= ASK_LIFETIME_MS ? 1.0 : Math.max(0, 1.0 - (age - ASK_LIFETIME_MS) / 700);
        if (fade <= 0.01) continue;

        const fromA = byId.get(ask.from);
        const toA = byId.get(ask.to);
        if (!fromA || !toA) continue;

        const [x1, y1] = this._toScreen(fromA.x, fromA.y);
        const [x2, y2] = this._toScreen(toA.x, toA.y);
        const mx = (x1 + x2) * 0.5;
        const my = (y1 + y2) * 0.5;

        const ans = ask.answer || "pending";
        const lineColor =
          ans === "yes"
            ? `rgba(62, 118, 86, ${(0.84 * fade).toFixed(3)})`
            : ans === "no"
              ? `rgba(74, 64, 68, ${(0.72 * fade).toFixed(3)})`
              : `rgba(52, 46, 50, ${(0.76 * fade).toFixed(3)})`;

        ctx.strokeStyle = lineColor;
        ctx.lineWidth = ans === "yes" ? 1.35 : 1.05;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        // Small endpoint dot on asker
        ctx.fillStyle = lineColor;
        ctx.beginPath();
        ctx.arc(x1, y1, 2.0, 0, Math.PI * 2);
        ctx.fill();

        // Short red tick across midpoint when answer === "no"
        if (ans === "no") {
          const dx = x2 - x1;
          const dy = y2 - y1;
          const len = Math.hypot(dx, dy) || 1;
          const nx = -dy / len;
          const ny = dx / len;
          ctx.strokeStyle = `rgba(196, 52, 42, ${(0.92 * fade).toFixed(3)})`;
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          ctx.moveTo(mx - nx * 5.5, my - ny * 5.5);
          ctx.lineTo(mx + nx * 5.5, my + ny * 5.5);
          ctx.stroke();
        }

        // Draw up to 5 clean non-overlapping ask quote labels
        if (ask.text && placedBoxes.length < 5) {
          const label = `A${ask.from}→A${ask.to}: ${ask.text}`;
          const textW = ctx.measureText(label).width;
          const textH = 13;
          const alignRight = mx > ox + size * 0.58;
          const tx = alignRight ? mx - 7 : mx + 7;
          let ty = my - 5;

          const left = alignRight ? tx - textW : tx;
          // Nudge vertically if overlapping an already-placed ask label
          for (let attempt = 0; attempt < 6; attempt++) {
            const top = ty - textH;
            const overlaps = placedBoxes.some(
              (b) => left < b.right + 4 && left + textW > b.left - 4 && top < b.bottom + 2 && ty > b.top - 2
            );
            if (!overlaps) break;
            ty += attempt % 2 === 0 ? -(textH + 4) : (textH + 4) * Math.ceil((attempt + 1) / 2);
          }

          placedBoxes.push({ left, right: left + textW, top: ty - textH, bottom: ty });
          ctx.textAlign = alignRight ? "right" : "left";
          ctx.textBaseline = "bottom";

          ctx.lineWidth = 3.2;
          ctx.strokeStyle = `rgba(246, 235, 230, ${(0.88 * fade).toFixed(3)})`;
          ctx.strokeText(label, tx, ty);

          ctx.fillStyle =
            ans === "yes"
              ? `rgba(46, 98, 68, ${(0.95 * fade).toFixed(3)})`
              : `rgba(38, 32, 36, ${(0.90 * fade).toFixed(3)})`;
          ctx.fillText(label, tx, ty);
        }
      }
      ctx.restore();
    }
  }
}
