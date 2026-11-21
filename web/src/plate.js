// WebGL2 TLC plate blob renderer (plan.md section 10).
// Splat-shader ping-pong approach inspired by Pavel Dobryakov's WebGL Fluid Simulation (MIT).

import fullscreenVert from "./shaders/fullscreen.vert.glsl?raw";
import decayFrag from "./shaders/decay.frag.glsl?raw";
import splatVert from "./shaders/splat.vert.glsl?raw";
import splatFrag from "./shaders/splat.frag.glsl?raw";
import compositeFrag from "./shaders/composite.frag.glsl?raw";
import { CrowdColorMapper } from "./palette.js";

export const AGENT_RADIUS = 0.027;
const MAX_AGENTS = 128;
const AGENT_TEX_ROWS = 3;
const DEFAULT_FLOOR = { x_min: 0.15, x_max: 0.85, y_min: 0.0, y_max: 0.70, dj_y: 0.03 };

/**
 * Shared square plate transform in CSS pixels so WebGL2 and Canvas2D overlay align pixel-perfectly.
 * When `rightReservePx > 0` (e.g. chat panel open), fits the plate inside the remaining left region
 * so the chat panel never covers any agent on the dancefloor.
 */
export function computePlateRectCss(cssWidth, cssHeight, rightReservePx = 0) {
  const availW = Math.max(120, cssWidth - rightReservePx);
  const pad = Math.max(12, Math.min(availW, cssHeight) * 0.03);
  const size = Math.max(64, Math.min(availW, cssHeight) - pad * 2);
  const ox = (availW - size) * 0.5;
  const oy = (cssHeight - size) * 0.5;
  return { ox, oy, size };
}

function compileShader(gl, type, source, name) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error (${name}): ${log}`);
  }
  return shader;
}

function linkProgram(gl, vertSrc, fragSrc, name) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc, `${name}.vert`);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc, `${name}.frag`);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`Program link error (${name}): ${log}`);
  }
  return prog;
}

export class PlateRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: true, // allows reliable screenshots without overhead
      powerPreference: "high-performance",
    });
    if (!gl) {
      throw new Error("WebGL2 is required but unavailable.");
    }
    this.gl = gl;

    // Check float render target support; fall back cleanly to RGBA8 if unavailable.
    this.useFloatTrails = Boolean(gl.getExtension("EXT_color_buffer_float"));
    this.useLinearFloat = this.useFloatTrails && Boolean(gl.getExtension("OES_texture_float_linear"));

    this.width = 1;
    this.height = 1;
    this.dpr = 1;
    this.rightReservePx = 0;
    this.plateRectCss = { ox: 0, oy: 0, size: 1 };

    // Programs
    this.decayProg = linkProgram(gl, fullscreenVert, decayFrag, "decay");
    this.splatProg = linkProgram(gl, splatVert, splatFrag, "splat");
    this.compProg = linkProgram(gl, fullscreenVert, compositeFrag, "composite");

    // Uniform locations
    this.uDecay = {
      prev: gl.getUniformLocation(this.decayProg, "uPrevTrail"),
      res: gl.getUniformLocation(this.decayProg, "uResolution"),
      decay: gl.getUniformLocation(this.decayProg, "uDecay"),
      advect: gl.getUniformLocation(this.decayProg, "uAdvectPx"),
    };
    this.uSplat = {
      res: gl.getUniformLocation(this.splatProg, "uResolution"),
      plateRect: gl.getUniformLocation(this.splatProg, "uPlateRect"),
      storeScale: gl.getUniformLocation(this.splatProg, "uStoreScale"),
    };
    this.uComp = {
      trail: gl.getUniformLocation(this.compProg, "uTrailTex"),
      agents: gl.getUniformLocation(this.compProg, "uAgentTex"),
      count: gl.getUniformLocation(this.compProg, "uAgentCount"),
      res: gl.getUniformLocation(this.compProg, "uResolution"),
      plateRect: gl.getUniformLocation(this.compProg, "uPlateRect"),
      dpr: gl.getUniformLocation(this.compProg, "uDpr"),
      time: gl.getUniformLocation(this.compProg, "uTime"),
      trailDecode: gl.getUniformLocation(this.compProg, "uTrailDecode"),
      djWallY: gl.getUniformLocation(this.compProg, "uDjWallY"),
    };

    // Fullscreen empty VAO
    this.emptyVao = gl.createVertexArray();

    // Instanced splat VAO (quad + per-instance attributes)
    this.splatVao = gl.createVertexArray();
    gl.bindVertexArray(this.splatVao);

    const quadVerts = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
    this.quadBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    // 11 floats per instance: posVel(4) + shape(4) + absorbance(3)
    this.instanceFloats = 11;
    this.instanceData = new Float32Array(MAX_AGENTS * this.instanceFloats);
    this.instanceBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.instanceData.byteLength, gl.DYNAMIC_DRAW);

    const stride = this.instanceFloats * 4;
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, stride, 0);
    gl.vertexAttribDivisor(1, 1);

    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 4, gl.FLOAT, false, stride, 16);
    gl.vertexAttribDivisor(2, 1);

    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 3, gl.FLOAT, false, stride, 32);
    gl.vertexAttribDivisor(3, 1);

    gl.bindVertexArray(null);

    // 128 x 3 RGBA32F agent data texture (core WebGL2, NEAREST filtering)
    // row 0: (x, y, rPulse, density)
    // row 1: (absR, absG, absB, djSquish)
    // row 2: (vx, vy, wobble2, wobble3)
    this.agentTexData = new Float32Array(MAX_AGENTS * AGENT_TEX_ROWS * 4);
    this.agentTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.agentTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA32F,
      MAX_AGENTS,
      AGENT_TEX_ROWS,
      0,
      gl.RGBA,
      gl.FLOAT,
      this.agentTexData
    );

    // Per-agent visual spring-damper wobble & DJ-squish state keyed by agent id
    this._blobState = new Map();
    this._prevKickEnv = 0;
    this.floor = { ...DEFAULT_FLOOR };

    // Ping-pong trail FBOs (created in resize)
    this.trails = [null, null];
    this.trailIdx = 0;
    this.colorMapper = new CrowdColorMapper();

    this.resize(canvas.clientWidth || window.innerWidth, canvas.clientHeight || window.innerHeight);
  }

  _createTrailTarget(w, h) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    const filter = this.useFloatTrails && !this.useLinearFloat ? gl.NEAREST : gl.LINEAR;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    if (this.useFloatTrails) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }

    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE && this.useFloatTrails) {
      // Runtime fallback to RGBA8 if RGBA16F attachment fails on driver.
      this.useFloatTrails = false;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }

    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { tex, fbo };
  }

  _destroyTrails() {
    const gl = this.gl;
    for (const t of this.trails) {
      if (!t) continue;
      gl.deleteFramebuffer(t.fbo);
      gl.deleteTexture(t.tex);
    }
    this.trails = [null, null];
  }

  resize(cssWidth, cssHeight, rightReservePx = this.rightReservePx) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(2, Math.round(cssWidth * dpr));
    const h = Math.max(2, Math.round(cssHeight * dpr));
    const reserveChanged = rightReservePx !== this.rightReservePx;
    this.rightReservePx = rightReservePx;
    this.plateRectCss = computePlateRectCss(cssWidth, cssHeight, rightReservePx);

    if (w === this.width && h === this.height && dpr === this.dpr && this.trails[0]) {
      if (reserveChanged) this.clearTrails();
      return;
    }
    this.dpr = dpr;
    this.width = w;
    this.height = h;
    this.canvas.width = w;
    this.canvas.height = h;

    this._destroyTrails();
    this.trails = [this._createTrailTarget(w, h), this._createTrailTarget(w, h)];
    this.trailIdx = 0;
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

  clearTrails() {
    const gl = this.gl;
    gl.clearColor(0, 0, 0, 0);
    for (const t of this.trails) {
      if (!t) continue;
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Render one frame from interpolated agent list.
   * Each agent: { id, x, y, vx, vy, E, rho, H, action, pressure }
   */
  render(agents, { musicEnergy = 0, kickEnvelope = 0, timeSec = 0, dtSec = 1 / 60 } = {}) {
    const gl = this.gl;
    const count = Math.min(agents.length, MAX_AGENTS);

    // Update smoothed crowd H spread for full-range TLC hue contrast
    this.colorMapper.updateSpread(agents, dtSec);

    // Detect kick onset for wobble spring impulse
    const kickDelta = Math.max(0, kickEnvelope - this._prevKickEnv);
    this._prevKickEnv = kickEnvelope;
    const dt = Math.min(0.04, Math.max(0.002, dtSec));

    // Map compact physical floor isotropically onto [0, 1] x [0, 1] plate so blobs read bigger
    const floorSpan = Math.max(0.1, this.floor.x_max - this.floor.x_min);
    const invSpan = 1.0 / floorSpan;
    const djWallNormY = (this.floor.dj_y - this.floor.y_min) * invSpan;

    // Pack agent buffers (DJ at bottom: physics y=0 maps directly to GL y=0; origin y_max maps to top).
    const inst = this.instanceData;
    const tex = this.agentTexData;
    const row1Offset = MAX_AGENTS * 4;
    const row2Offset = MAX_AGENTS * 8;

    for (let i = 0; i < count; i++) {
      const a = agents[i];
      const xGl = (a.x - this.floor.x_min) * invSpan;
      const yGl = (a.y - this.floor.y_min) * invSpan;
      const vxGl = a.vx * invSpan;
      const vyGl = a.vy * invSpan;
      const E = Math.max(0, Math.min(1, a.E));
      const pressure = Math.max(0, a.pressure || 0);

      // Proximity to wider DJ segment if not already supplied by backend
      let vProx = a.v;
      if (!Number.isFinite(vProx)) {
        const cx = Math.max(0.3, Math.min(0.7, xGl));
        const dSeg = Math.hypot(xGl - cx, yGl);
        vProx = Math.max(0, Math.min(1, 1.0 - dSeg / 1.04403));
      }
      // Ring-attenuated local_music if not supplied by backend
      let localMusic = a.localMusic;
      if (!Number.isFinite(localMusic)) {
        const ringGain = 0.45 + 0.55 * vProx;
        localMusic = musicEnergy * ringGain;
      }

      // Per-agent radius scaled cleanly by 1 / floorSpan so compact floor fills the plate
      const physR =
        Number.isFinite(a.r) && a.r > 0.004
          ? a.r * invSpan
          : AGENT_RADIUS * invSpan * (0.8 + 0.45 * vProx) * (0.85 + 0.15 * E);
      const bulge = 1.0 + 0.10 * Math.min(1.0, pressure * 12.0);
      const rBase = physR * 1.24 * bulge;
      const pulseScale = 1.0 + 0.14 * localMusic * E + 0.30 * kickEnvelope * localMusic * E;
      const rPulse = rBase * pulseScale;

      // Update per-agent 2-mode spring-damper wobble + DJ-line squish spring
      let bs = this._blobState.get(a.id);
      if (!bs) {
        bs = { w2: 0, v2: 0, w3: 0, v3: 0, djSquish: 0, djVel: 0, prevP: pressure, prevVx: vxGl, prevVy: vyGl };
        this._blobState.set(a.id, bs);
      }
      const pImpulse = Math.max(0, pressure - bs.prevP) * 28.0;
      const dvImpulse = Math.hypot(vxGl - bs.prevVx, vyGl - bs.prevVy) * 4.0;
      const kImpulse = kickDelta * localMusic * E * 3.8;
      const totalImp = Math.min(2.2, pImpulse + dvImpulse + kImpulse);
      bs.prevP = pressure;
      bs.prevVx = vxGl;
      bs.prevVy = vyGl;

      // Mode 2 (quadrupole jelly oval): w=18 rad/s, zeta=0.26
      const w2Freq = 18.0;
      const a2 = -w2Freq * w2Freq * bs.w2 - 2.0 * 0.26 * w2Freq * bs.v2;
      bs.v2 += a2 * dt + totalImp * 1.15;
      bs.w2 = Math.max(-0.18, Math.min(0.18, bs.w2 + bs.v2 * dt));

      // Mode 3 (organic trefoil wobble): w=25 rad/s, zeta=0.30
      const w3Freq = 25.0;
      const a3 = -w3Freq * w3Freq * bs.w3 - 2.0 * 0.30 * w3Freq * bs.v3;
      bs.v3 += a3 * dt + totalImp * 0.72 * (a.id % 2 === 0 ? 1 : -1);
      bs.w3 = Math.max(-0.13, Math.min(0.13, bs.w3 + bs.v3 * dt));

      // DJ-line squish spring: flattens when pressed against djWallNormY, springs back round when moving away
      const djDist = yGl - djWallNormY;
      const targetSquish = Math.max(0, Math.min(1, (rPulse * 1.18 - djDist) / Math.max(rPulse * 0.65, 0.01)));
      const djFreq = 14.0;
      const djAcc = djFreq * djFreq * (targetSquish - bs.djSquish) - 2.0 * 0.52 * djFreq * bs.djVel;
      bs.djVel += djAcc * dt;
      bs.djSquish = Math.max(0, Math.min(1.15, bs.djSquish + bs.djVel * dt));

      // Proximity vibrancy: near DJ = vivid & saturated; back = paler, fading toward paper
      const vibrancy = 0.22 + 0.78 * Math.pow(vProx, 0.85);
      const density = (0.28 + 1.28 * E) * (0.34 + 0.66 * vibrancy);
      const [ar, ag, ab] = this.colorMapper.absorbanceForAgent(a.H, a.id, vibrancy);

      const io = i * this.instanceFloats;
      inst[io] = xGl;
      inst[io + 1] = yGl;
      inst[io + 2] = vxGl;
      inst[io + 3] = vyGl;
      inst[io + 4] = rPulse;
      inst[io + 5] = E;
      inst[io + 6] = pressure;
      inst[io + 7] = vibrancy;
      inst[io + 8] = ar;
      inst[io + 9] = ag;
      inst[io + 10] = ab;

      const r0 = i * 4;
      tex[r0] = xGl;
      tex[r0 + 1] = yGl;
      tex[r0 + 2] = rPulse;
      tex[r0 + 3] = density;

      const r1 = row1Offset + i * 4;
      tex[r1] = ar;
      tex[r1 + 1] = ag;
      tex[r1 + 2] = ab;
      tex[r1 + 3] = bs.djSquish;

      const r2 = row2Offset + i * 4;
      tex[r2] = vxGl;
      tex[r2 + 1] = vyGl;
      tex[r2 + 2] = bs.w2;
      tex[r2 + 3] = bs.w3;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, inst.subarray(0, count * this.instanceFloats));

    gl.bindTexture(gl.TEXTURE_2D, this.agentTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, MAX_AGENTS, AGENT_TEX_ROWS, gl.RGBA, gl.FLOAT, tex);

    // Plate rect in GL framebuffer pixels (bottom-left origin)
    const { ox, oy, size } = this.plateRectCss;
    const oxGl = ox * this.dpr;
    const sizeGl = size * this.dpr;
    const oyGl = this.height - (oy + size) * this.dpr;

    const readTrail = this.trails[this.trailIdx];
    const writeTrail = this.trails[1 - this.trailIdx];

    // Normalize decay/advection to 60 fps step
    const frameScale = Math.min(2.5, Math.max(0.25, dtSec * 60.0));
    const baseDecay = this.useFloatTrails ? 0.985 : 0.975;
    const decay = Math.pow(baseDecay, frameScale);
    const advectPx = 0.3 * this.dpr * frameScale;
    const storeScale = this.useFloatTrails ? 1.0 : 0.25;
    const trailDecode = this.useFloatTrails ? 1.0 : 4.0;

    gl.viewport(0, 0, this.width, this.height);

    // --- Pass 1: Decay + downward advection into writeTrail ---
    gl.bindFramebuffer(gl.FRAMEBUFFER, writeTrail.fbo);
    gl.disable(gl.BLEND);
    gl.useProgram(this.decayProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, readTrail.tex);
    gl.uniform1i(this.uDecay.prev, 0);
    gl.uniform2f(this.uDecay.res, this.width, this.height);
    gl.uniform1f(this.uDecay.decay, decay);
    gl.uniform1f(this.uDecay.advect, advectPx);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // --- Pass 2: Additive anisotropic Gaussian splats into writeTrail ---
    if (count > 0) {
      gl.enable(gl.BLEND);
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.useProgram(this.splatProg);
      gl.uniform2f(this.uSplat.res, this.width, this.height);
      gl.uniform4f(this.uSplat.plateRect, oxGl, oyGl, sizeGl, sizeGl);
      gl.uniform1f(this.uSplat.storeScale, storeScale);
      gl.bindVertexArray(this.splatVao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, count);
      gl.disable(gl.BLEND);
    }

    // Flip ping-pong index
    this.trailIdx = 1 - this.trailIdx;

    // --- Pass 3: Body (power-diagram non-merging soft cells) + Beer-Lambert composite ---
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(this.compProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, writeTrail.tex);
    gl.uniform1i(this.uComp.trail, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.agentTex);
    gl.uniform1i(this.uComp.agents, 1);
    gl.uniform1i(this.uComp.count, count);
    gl.uniform2f(this.uComp.res, this.width, this.height);
    gl.uniform4f(this.uComp.plateRect, oxGl, oyGl, sizeGl, sizeGl);
    gl.uniform1f(this.uComp.dpr, this.dpr);
    gl.uniform1f(this.uComp.time, timeSec);
    gl.uniform1f(this.uComp.trailDecode, trailDecode);
    gl.uniform1f(this.uComp.djWallY, djWallNormY);

    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }
}
