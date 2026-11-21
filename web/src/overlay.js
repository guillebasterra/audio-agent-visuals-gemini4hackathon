// DPR-aware Canvas2D overlay: TLC solvent front ("DJ" + hot segment), origin line + lane letters,
// agent IDs (A12), non-overlapping ask bubbles (max 3, clear of DJ line), and hover highlight rings.

import { AGENT_RADIUS, computePlateRectCss } from "./plate.js";

const DEFAULT_LANE_LABELS = ["B", "C", "E", "G", "H", "L", "M", "P", "S", "Ś"];
const DEFAULT_FLOOR = { x_min: 0.15, x_max: 0.85, y_min: 0.0, y_max: 0.70, dj_y: 0.03 };
const ASK_LIFETIME_MS = 3000;
const MAX_ASK_BUBBLES = 3;

export class OverlayRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.cssWidth = 1;
    this.cssHeight = 1;
    this.dpr = 1;
    this.rightReservePx = 0;
    this.plateRect = { ox: 0, oy: 0, size: 1 };
    this.laneLabels = DEFAULT_LANE_LABELS;
    this.floor = { ...DEFAULT_FLOOR };
    this.djSegment = [0.36, 0.64];
    this.showLabels = true;
    this.highlightedIds = [];
    this.resize(canvas.clientWidth || window.innerWidth, canvas.clientHeight || window.innerHeight);
  }

  setLaneLabels(labels) {
    if (Array.isArray(labels) && labels.length > 0) {
      this.laneLabels = labels;
    }
  }

  setDjSegment(seg) {
    if (Array.isArray(seg) && seg.length === 2) {
      this.djSegment = [Number(seg[0]), Number(seg[1])];
    }
  }

  setFloor(floor) {
    if (floor && Number.isFinite(floor.x_min) && Number.isFinite(floor.x_max) && floor.x_max > floor.x_min) {
      this.floor = {
        x_min: Number(floor.x_min),
        x_max: Number(floor.x_max),
        y_min: Number.isFinite(floor.y_min) ? Number(floor.y_min) : 0.0,
        y_max: Number.isFinite(floor.y_max) ? Number(floor.y_max) : 0.70,
        dj_y: Number.isFinite(floor.dj_y) ? Number(floor.dj_y) : 0.03,
      };
    }
  }

  setHighlightedAgents(ids) {
    this.highlightedIds = Array.isArray(ids) ? ids : [];
  }

  resize(cssWidth, cssHeight, rightReservePx = this.rightReservePx) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.cssWidth = cssWidth;
    this.cssHeight = cssHeight;
    this.dpr = dpr;
    this.rightReservePx = rightReservePx;
    this.canvas.width = Math.max(2, Math.round(cssWidth * dpr));
    this.canvas.height = Math.max(2, Math.round(cssHeight * dpr));
    this.plateRect = computePlateRectCss(cssWidth, cssHeight, rightReservePx);
  }

  _floorSpan() {
    return Math.max(0.1, this.floor.x_max - this.floor.x_min);
  }

  _toScreen(xPhys, yPhys) {
    // Isotropic map from compact floor [x_min..x_max] x [y_min..y_max] to full square TLC plate;
    // DJ (y_min) at bottom of screen, origin line (y_max) at top.
    const { ox, oy, size } = this.plateRect;
    const invSpan = 1.0 / this._floorSpan();
    const nx = (xPhys - this.floor.x_min) * invSpan;
    const ny = (yPhys - this.floor.y_min) * invSpan;
    return [ox + nx * size, oy + (1.0 - ny) * size];
  }

  render(agents, asks = [], nowMs = performance.now()) {
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.cssWidth, this.cssHeight);

    const { ox, oy, size } = this.plateRect;
    const invSpan = 1.0 / this._floorSpan();
    const djNormY = (this.floor.dj_y - this.floor.y_min) * invSpan;

    // 1. Faint pencil DJ line at bottom + hot-zone segment
    ctx.save();
    ctx.strokeStyle = "rgba(65, 55, 58, 0.22)";
    ctx.lineWidth = 0.9;
    const djY = oy + (1.0 - djNormY) * size;
    ctx.beginPath();
    ctx.moveTo(ox, djY);
    ctx.lineTo(ox + size, djY);
    ctx.stroke();

    const segX0 = ox + (this.djSegment[0] - this.floor.x_min) * invSpan * size;
    const segX1 = ox + (this.djSegment[1] - this.floor.x_min) * invSpan * size;
    ctx.strokeStyle = "rgba(84, 48, 56, 0.52)";
    ctx.lineWidth = 1.7;
    ctx.beginPath();
    ctx.moveTo(segX0, djY);
    ctx.lineTo(segX1, djY);
    ctx.moveTo(segX0, djY - 3);
    ctx.lineTo(segX0, djY + 3);
    ctx.moveTo(segX1, djY - 3);
    ctx.lineTo(segX1, djY + 3);
    ctx.stroke();

    ctx.font = '500 10px "IBM Plex Mono", monospace';
    ctx.fillStyle = "rgba(55, 46, 50, 0.65)";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("DJ", (segX0 + segX1) * 0.5, djY + 4);

    // 2. Faint pencil origin line at top + lane letters
    const originY = oy + (1.0 - 0.955) * size;
    ctx.strokeStyle = "rgba(65, 55, 58, 0.26)";
    ctx.lineWidth = 0.9;
    ctx.beginPath();
    ctx.moveTo(ox, originY);
    ctx.lineTo(ox + size, originY);
    ctx.stroke();

    const nLanes = this.laneLabels.length;
    ctx.font = '400 11px "IBM Plex Mono", monospace';
    ctx.fillStyle = "rgba(70, 58, 60, 0.54)";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    for (let i = 0; i < nLanes; i++) {
      const lx = ox + ((i + 0.5) / nLanes) * size;
      ctx.beginPath();
      ctx.moveTo(lx, originY - 2.5);
      ctx.lineTo(lx, originY + 2.5);
      ctx.stroke();
      ctx.fillText(this.laneLabels[i], lx, originY - 5);
    }
    ctx.restore();

    // Build quick lookup map by agent id
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

    // 4. Ask threads + up to 3 non-overlapping ask bubbles strictly clear of the DJ line
    if (asks && asks.length > 0) {
      ctx.save();
      ctx.font = '400 10px "IBM Plex Mono", monospace';
      const visible = asks
        .map((ask) => ({
          ask,
          age: nowMs - (ask.updatedAt || ask.createdAt || nowMs),
        }))
        .filter((item) => item.age <= ASK_LIFETIME_MS + 700)
        .sort((a, b) => a.age - b.age);

      const placedBoxes = [];
      const bubblesToDraw = [];
      const maxBubbleBottom = djY - 28; // keep ask bubbles strictly clear of DJ line & bottom rail
      const minBubbleTop = originY + 12;

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
            ? `rgba(56, 118, 82, ${(0.86 * fade).toFixed(3)})`
            : ans === "no"
              ? `rgba(74, 64, 68, ${(0.72 * fade).toFixed(3)})`
              : `rgba(52, 46, 50, ${(0.78 * fade).toFixed(3)})`;

        ctx.strokeStyle = lineColor;
        ctx.lineWidth = ans === "yes" ? 1.45 : 1.05;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        // Small endpoint dot on asker
        ctx.fillStyle = lineColor;
        ctx.beginPath();
        ctx.arc(x1, y1, 2.1, 0, Math.PI * 2);
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

        // Schedule up to 3 strictly non-overlapping bubbles clear of the DJ line
        if (ask.text && placedBoxes.length < MAX_ASK_BUBBLES) {
          const rawLabel = `A${ask.from}→A${ask.to}: ${ask.text}`;
          const maxLabelW = Math.min(size * 0.72, 420);
          let label = rawLabel;
          while (label.length > 24 && ctx.measureText(label).width > maxLabelW) {
            label = label.slice(0, -2) + "…";
          }
          const padX = 6;
          const padY = 3;
          const textW = ctx.measureText(label).width;
          const boxW = textW + padX * 2;
          const boxH = 12 + padY * 2;

          const minLeft = ox + 6;
          const maxLeft = Math.max(minLeft, ox + size - boxW - 6);
          const prefLeft = Math.max(minLeft, Math.min(maxLeft, mx - boxW * 0.5));
          const prefBottom = Math.min(maxBubbleBottom, my - 10);

          const overlapsAny = (l, t, r, b) =>
            placedBoxes.some((p) => l < p.right + 6 && r > p.left - 6 && t < p.bottom + 5 && b > p.top - 5);

          let chosen = null;
          // Scan upward from preferred position (with slight horizontal offsets) so bubbles never overlap
          const xCandidates = [
            prefLeft,
            Math.max(minLeft, Math.min(maxLeft, prefLeft - boxW * 0.35)),
            Math.max(minLeft, Math.min(maxLeft, prefLeft + boxW * 0.35)),
          ];
          for (let row = 0; row < 14 && !chosen; row++) {
            const bottom = prefBottom - row * (boxH + 6);
            const top = bottom - boxH;
            if (top < minBubbleTop) break;
            for (const left of xCandidates) {
              const right = left + boxW;
              if (!overlapsAny(left, top, right, bottom)) {
                chosen = { left, top, right, bottom };
                break;
              }
            }
          }

          if (chosen) {
            placedBoxes.push(chosen);
            bubblesToDraw.push({
              box: chosen,
              label,
              padX,
              padY,
              mx,
              my,
              ans,
              fade,
            });
          }
        }
      }

      // Draw crisp translucent paper pills on top of connector lines
      for (const b of bubblesToDraw) {
        const { left, top, right, bottom } = b.box;
        const cx = (left + right) * 0.5;
        // Subtle leader line if bubble was offset noticeably above its speaker midpoint
        if (b.my - bottom > 16) {
          ctx.strokeStyle = `rgba(60, 50, 54, ${(0.32 * b.fade).toFixed(3)})`;
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(b.mx, b.my);
          ctx.lineTo(Math.max(left + 8, Math.min(right - 8, b.mx)), bottom);
          ctx.stroke();
        }

        ctx.fillStyle = `rgba(246, 235, 230, ${(0.88 * b.fade).toFixed(3)})`;
        ctx.strokeStyle =
          b.ans === "yes"
            ? `rgba(48, 110, 74, ${(0.55 * b.fade).toFixed(3)})`
            : `rgba(55, 45, 48, ${(0.22 * b.fade).toFixed(3)})`;
        ctx.lineWidth = 0.9;
        ctx.beginPath();
        ctx.roundRect(left, top, right - left, bottom - top, 3);
        ctx.fill();
        ctx.stroke();

        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillStyle =
          b.ans === "yes"
            ? `rgba(38, 94, 62, ${(0.96 * b.fade).toFixed(3)})`
            : `rgba(34, 28, 32, ${(0.92 * b.fade).toFixed(3)})`;
        ctx.fillText(b.label, left + b.padX, (top + bottom) * 0.5 + 0.5);
      }

      ctx.restore();
    }

    // 5. Chat-hover highlight rings (thin crisp ring fitted to each agent's scaled radius)
    if (this.highlightedIds.length > 0) {
      ctx.save();
      const pts = [];
      for (let i = 0; i < this.highlightedIds.length; i++) {
        const ag = byId.get(this.highlightedIds[i]);
        if (!ag) continue;
        const [sx, sy] = this._toScreen(ag.x, ag.y);
        const rNorm = (Number.isFinite(ag.r) && ag.r > 0.004 ? ag.r : AGENT_RADIUS) * invSpan;
        const ringR = rNorm * 1.28 * size;
        pts.push([sx, sy]);

        ctx.strokeStyle = "rgba(246, 235, 230, 0.85)";
        ctx.lineWidth = 3.2;
        ctx.beginPath();
        ctx.arc(sx, sy, ringR, 0, Math.PI * 2);
        ctx.stroke();

        ctx.strokeStyle = i === 0 ? "rgba(38, 28, 34, 0.88)" : "rgba(92, 68, 82, 0.78)";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(sx, sy, ringR, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (pts.length === 2) {
        ctx.strokeStyle = "rgba(42, 34, 38, 0.55)";
        ctx.lineWidth = 1.0;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        ctx.lineTo(pts[1][0], pts[1][1]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.restore();
    }
  }
}
