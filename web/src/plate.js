// WebGL2 TLC plate blob renderer (plan.md section 10).
// Splat-shader ping-pong approach inspired by Pavel Dobryakov's WebGL Fluid Simulation (MIT).

import fullscreenVert from "./shaders/fullscreen.vert.glsl?raw";
import decayFrag from "./shaders/decay.frag.glsl?raw";
import splatVert from "./shaders/splat.vert.glsl?raw";
import splatFrag from "./shaders/splat.frag.glsl?raw";
import compositeFrag from "./shaders/composite.frag.glsl?raw";
import { happinessToAbsorbance } from "./palette.js";

export const AGENT_RADIUS = 0.028;
const MAX_AGENTS = 128;

/**
 * Shared square plate transform in CSS pixels so WebGL2 and Canvas2D overlay align pixel-perfectly.
 * Physics y=0 (DJ) maps to top of plateRect; physics y=1 (origin) maps to bottom.
 */
export function computePlateRectCss(cssWidth, cssHeight) {
  const pad = Math.max(12, Math.min(cssWidth, cssHeight) * 0.03);
  const size = Math.max(64, Math.min(cssWidth, cssHeight) - pad * 2);
  const ox = (cssWidth - size) * 0.5;
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

    // 128 x 2 RGBA32F agent data texture (core WebGL2, NEAREST filtering)
    this.agentTexData = new Float32Array(MAX_AGENTS * 2 * 4);
    this.agentTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.agentTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, MAX_AGENTS, 2, 0, gl.RGBA, gl.FLOAT, this.agentTexData);

    // Ping-pong trail FBOs (created in resize)
    this.trails = [null, null];
    this.trailIdx = 0;

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

  resize(cssWidth, cssHeight) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(2, Math.round(cssWidth * dpr));
    const h = Math.max(2, Math.round(cssHeight * dpr));
    if (w === this.width && h === this.height && dpr === this.dpr && this.trails[0]) {
      return;
    }
    this.dpr = dpr;
    this.width = w;
    this.height = h;
    this.canvas.width = w;
    this.canvas.height = h;
    this.plateRectCss = computePlateRectCss(cssWidth, cssHeight);

    this._destroyTrails();
    this.trails = [this._createTrailTarget(w, h), this._createTrailTarget(w, h)];
    this.trailIdx = 0;
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

    // Pack agent buffers (flip physics y -> GL y = 1 - y so DJ is at top).
    const inst = this.instanceData;
    const tex = this.agentTexData;
    const row1Offset = MAX_AGENTS * 4;

    for (let i = 0; i < count; i++) {
      const a = agents[i];
      const xGl = a.x;
      const yGl = 1.0 - a.y;
      const vxGl = a.vx;
      const vyGl = -a.vy;
      const E = Math.max(0, Math.min(1, a.E));
      const pressure = Math.max(0, a.pressure || 0);
      // Section 10 pulse formula with generous soft-body scale (1.28x physics radius)
      // so touching neighbors press into flat-sided rounded polygons with a ~1.5px gap,
      // while isolated agents stay soft round TLC spots.
      const rBase = AGENT_RADIUS * 1.28;
      const rPulse = rBase * (1.0 + 0.15 * musicEnergy * E + 0.25 * kickEnvelope * E);
      const [ar, ag, ab] = happinessToAbsorbance(a.H);

      const io = i * this.instanceFloats;
      inst[io] = xGl;
      inst[io + 1] = yGl;
      inst[io + 2] = vxGl;
      inst[io + 3] = vyGl;
      inst[io + 4] = rPulse;
      inst[io + 5] = E;
      inst[io + 6] = pressure;
      inst[io + 7] = 0;
      inst[io + 8] = ar;
      inst[io + 9] = ag;
      inst[io + 10] = ab;

      const r0 = i * 4;
      tex[r0] = xGl;
      tex[r0 + 1] = yGl;
      tex[r0 + 2] = rPulse;
      tex[r0 + 3] = E;

      const r1 = row1Offset + i * 4;
      tex[r1] = ar;
      tex[r1 + 1] = ag;
      tex[r1 + 2] = ab;
      tex[r1 + 3] = pressure;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, inst.subarray(0, count * this.instanceFloats));

    gl.bindTexture(gl.TEXTURE_2D, this.agentTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, MAX_AGENTS, 2, gl.RGBA, gl.FLOAT, tex);

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

    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }
}
