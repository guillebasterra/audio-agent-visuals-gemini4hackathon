// Shared Web Audio analysis (60 fps FFT features) + 16 kHz mono Int16 AudioWorklet capture.
// Single swappable source entry point (`connectSource`) shared by file playback and live getUserMedia input.

const FFT_SIZE = 2048;
const KICK_REFRACTORY_SEC = 0.12;
const KICK_DECAY_TAU_SEC = 0.12;
const CLOCK_INTERVAL_MS = 250; // 4 Hz
const SILENCE_RMS_THRESHOLD = 0.0004;

const DJ_DEVICE_PATTERN =
  /(blackhole|loopback|rekordbox|pioneer|alphatheta|ddj|djm|cdj|xdj|xone|allen|traktor|serato|scarlett|focusrite|soundflower|vb-cable|multi-output|aggregate|usb\s+audio|mixer)/i;

export class AudioEngine {
  constructor({ onClock, onPcmChunk, onControl, onDevicesChanged, onLiveEnded } = {}) {
    this.onClock = onClock || (() => {});
    this.onPcmChunk = onPcmChunk || (() => {});
    this.onControl = onControl || (() => {});
    this.onDevicesChanged = onDevicesChanged || (() => {});
    this.onLiveEnded = onLiveEnded || (() => {});

    this.ctx = null;
    this.analyser = null;
    this.workletNode = null;
    this.workletReady = false;

    this.currentSourceNode = null;
    this.currentSourceMode = "file"; // "file" | "live"
    this.currentLabel = "";
    this.currentDeviceId = "";
    this.liveStream = null;
    this.liveTrackSettings = null;
    this.monitorLive = false;
    this._monitorConnected = false;

    this.audioEl = document.createElement("audio");
    this.audioEl.crossOrigin = "anonymous";
    this.audioEl.preload = "auto";
    this.audioEl.loop = false;
    this.mediaElSource = null;

    this.playing = false;
    this.liveElapsedMs = 0;
    this.liveResumeMs = 0;

    // FFT buffers
    this.freqData = null;
    this.timeData = null;
    this.prevLowSpectrum = null;

    // Decaying running maxima for normalization
    this.maxBands = { sub: 0.002, bass: 0.002, mid: 0.001, high: 0.0004, rms: 0.03 };
    this.smoothBands = null;
    this.fluxMean = 0.002;
    this.lastKickSec = -10;
    this.kickEnvelope = 0;
    this.lastBassAboveSec = 0;

    // Live input level & silence monitor
    this.rawRms = 0;
    this.inputLevel = 0; // 0..1 log VU meter
    this.inputDb = -96;
    this.liveSilentSec = 0;

    // Rolling 10 s history of { t, energy } for energy_5s and trend
    this.energyHistory = [];

    // Latest DSP snapshot
    this.dsp = {
      sub: 0,
      bass: 0,
      mid: 0,
      high: 0,
      rms: 0,
      kick: false,
      energy: 0,
      energy_5s: 0,
      trend: 0,
      bass_gap: 0,
    };

    // Telemetry for verification & stats corner
    this.pcmChunksTotal = 0;
    this.pcmChunkTimestamps = [];
    this.lastChunkBytes = 0;
    this.kicksTotal = 0;
    this._lastClockSentMs = 0;
    this._lastUpdateMs = 0;
    this._kickSinceLastClock = false;

    this._bindAudioElementEvents();
    this._bindDeviceChange();
  }

  _bindAudioElementEvents() {
    this.audioEl.addEventListener("play", () => {
      if (this.currentSourceMode === "file") {
        this._setPlaying(true);
      }
    });
    this.audioEl.addEventListener("pause", () => {
      if (this.currentSourceMode === "file") {
        this._setPlaying(false);
      }
    });
    this.audioEl.addEventListener("ended", () => {
      if (this.currentSourceMode === "file") {
        this._setPlaying(false);
      }
    });
    this.audioEl.addEventListener("seeked", () => {
      if (this.currentSourceMode !== "file") return;
      if (this.workletNode) {
        this.workletNode.port.postMessage({ reset: true });
      }
      this.energyHistory = [];
      this.lastBassAboveSec = this.getPlaybackTimeSec();
      this.onControl({ type: "control", action: "seek" });
    });
  }

  _bindDeviceChange() {
    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
      navigator.mediaDevices.addEventListener("devicechange", () => {
        this.onDevicesChanged();
      });
    }
  }

  _resetDspState() {
    this.maxBands = { sub: 0.002, bass: 0.002, mid: 0.001, high: 0.0004, rms: 0.03 };
    this.smoothBands = null;
    this.prevLowSpectrum = null;
    this.fluxMean = 0.002;
    this.kickEnvelope = 0;
    this.lastBassAboveSec = 0;
    this.energyHistory = [];
    this.rawRms = 0;
    this.inputLevel = 0;
    this.inputDb = -96;
    this.liveSilentSec = 0;
    this._kickSinceLastClock = false;
    this.dsp = {
      sub: 0,
      bass: 0,
      mid: 0,
      high: 0,
      rms: 0,
      kick: false,
      energy: 0,
      energy_5s: 0,
      trend: 0,
      bass_gap: 0,
    };
    if (this.workletNode) {
      this.workletNode.port.postMessage({ reset: true });
    }
  }

  async _ensureContext() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioCtx();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = FFT_SIZE;
      this.analyser.smoothingTimeConstant = 0.35;

      this.freqData = new Float32Array(this.analyser.frequencyBinCount);
      this.timeData = new Float32Array(this.analyser.fftSize);

      // Shared zero-gain sink so Chromium/WebKit always pulls AnalyserNode & AudioWorklet
      // in live mode even when speaker monitoring is disabled.
      const silentGain = this.ctx.createGain();
      silentGain.gain.value = 0;
      silentGain.connect(this.ctx.destination);
      this.analyser.connect(silentGain);

      if (this.ctx.audioWorklet) {
        try {
          await this.ctx.audioWorklet.addModule("/pcm-worklet.js");
          this.workletNode = new AudioWorkletNode(this.ctx, "pcm-capture-processor", {
            numberOfInputs: 1,
            numberOfOutputs: 1,
            channelCount: 2,
          });
          this.workletNode.connect(silentGain);

          this.workletNode.port.onmessage = (evt) => {
            if (!this.playing) return;
            const buf = evt.data;
            if (buf instanceof ArrayBuffer) {
              const now = performance.now();
              this.pcmChunksTotal += 1;
              this.lastChunkBytes = buf.byteLength;
              this.pcmChunkTimestamps.push(now);
              while (
                this.pcmChunkTimestamps.length > 0 &&
                now - this.pcmChunkTimestamps[0] > 4000
              ) {
                this.pcmChunkTimestamps.shift();
              }
              this.onPcmChunk(buf);
            }
          };
          this.workletReady = true;
        } catch (err) {
          console.warn("AudioWorklet setup failed:", err);
        }
      }
    }
    if (this.ctx.state === "suspended") {
      await this.ctx.resume();
    }
  }

  /**
   * Single swappable source entry point: everything downstream (AnalyserNode + PCM capture worklet)
   * is identical for file playback and live getUserMedia input.
   */
  connectSource(sourceNode, { monitorToSpeakers = false } = {}) {
    if (this.currentSourceNode) {
      try {
        this.currentSourceNode.disconnect();
      } catch {
        // ignore if already disconnected
      }
    }
    this.currentSourceNode = sourceNode;
    sourceNode.connect(this.analyser);
    if (this.workletNode) {
      sourceNode.connect(this.workletNode);
    }
    this._monitorConnected = Boolean(monitorToSpeakers);
    if (monitorToSpeakers) {
      sourceNode.connect(this.ctx.destination);
    }
  }

  setMonitorToSpeakers(enabled) {
    this.monitorLive = Boolean(enabled);
    if (this.currentSourceMode !== "live" || !this.currentSourceNode || !this.ctx) return;
    if (this.monitorLive && !this._monitorConnected) {
      this.currentSourceNode.connect(this.ctx.destination);
      this._monitorConnected = true;
    } else if (!this.monitorLive && this._monitorConnected) {
      try {
        this.currentSourceNode.disconnect(this.ctx.destination);
      } catch {
        // Fallback: disconnect all and reconnect analysis + worklet
        this.connectSource(this.currentSourceNode, { monitorToSpeakers: false });
      }
      this._monitorConnected = false;
    }
  }

  /**
   * Enumerate browser audio input devices (BlackHole 2ch, USB DJ mixer, mic).
   * If `requestPermission` is true and labels are hidden by browser privacy rules,
   * requests brief getUserMedia access to unlock real hardware names.
   */
  async listLiveDevices({ requestPermission = false } = {}) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      return {
        supported: false,
        devices: [],
        hasLabels: false,
        error: "Live audio input unavailable in this browser context (requires localhost or HTTPS).",
      };
    }

    let error = "";
    try {
      let raw = await navigator.mediaDevices.enumerateDevices();
      let inputs = raw.filter((d) => d.kind === "audioinput");
      let hasLabels = inputs.some((d) => d.label && d.label.trim().length > 0);

      if (requestPermission && (!hasLabels || inputs.length === 0) && navigator.mediaDevices.getUserMedia) {
        try {
          const probeStream = await navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: false,
            },
          });
          for (const tr of probeStream.getTracks()) tr.stop();
          raw = await navigator.mediaDevices.enumerateDevices();
          inputs = raw.filter((d) => d.kind === "audioinput");
          hasLabels = inputs.some((d) => d.label && d.label.trim().length > 0);
        } catch (err) {
          error = err && err.message ? err.message : String(err);
        }
      }

      const devices = inputs.map((d, idx) => {
        const cleanLabel = d.label && d.label.trim() ? d.label.trim() : `Audio Input ${idx + 1}`;
        const recommended = DJ_DEVICE_PATTERN.test(cleanLabel);
        return {
          deviceId: d.deviceId,
          groupId: d.groupId || "",
          label: cleanLabel,
          recommended,
        };
      });

      // Float recommended loopback / USB DJ mixer devices to the top while preserving order otherwise
      devices.sort((a, b) => Number(b.recommended) - Number(a.recommended));

      return {
        supported: true,
        devices,
        hasLabels,
        error,
      };
    } catch (err) {
      return {
        supported: true,
        devices: [],
        hasLabels: false,
        error: err && err.message ? err.message : String(err),
      };
    }
  }

  async loadFile(url, label = "") {
    await this._ensureContext();
    this._stopLiveStream();
    this._resetDspState();

    this.currentSourceMode = "file";
    this.currentLabel = label || url.split("/").pop() || "file";
    this.currentDeviceId = "";
    this.liveTrackSettings = null;

    if (!this.mediaElSource) {
      this.mediaElSource = this.ctx.createMediaElementSource(this.audioEl);
    }
    this.connectSource(this.mediaElSource, { monitorToSpeakers: true });

    this.audioEl.pause();
    this.audioEl.src = url;
    this.audioEl.load();
    this.onControl({ type: "control", source: "file", label: this.currentLabel });
  }

  async loadLiveDevice(deviceId = "", label = "", { monitorToSpeakers } = {}) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("getUserMedia is not available (use localhost or HTTPS).");
    }
    await this._ensureContext();
    this.audioEl.pause();
    this._stopLiveStream();
    this._resetDspState();

    const baseConstraints = {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      voiceIsolation: false,
      channelCount: { ideal: 2 },
    };

    const primaryConstraints = { ...baseConstraints };
    if (deviceId && deviceId !== "default") {
      primaryConstraints.deviceId = { exact: deviceId };
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: primaryConstraints });
    } catch (err) {
      // Fallback to ideal deviceId if the device ID changed after re-enumeration
      if (deviceId && err && (err.name === "OverconstrainedError" || err.name === "NotFoundError")) {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { ...baseConstraints, deviceId: { ideal: deviceId } },
        });
      } else {
        throw err;
      }
    }

    const track = stream.getAudioTracks()[0] || null;
    const settings = track && track.getSettings ? track.getSettings() : {};
    const resolvedLabel =
      (label && !/^Audio Input \d+$/.test(label) ? label : "") ||
      (track && track.label ? track.label.trim() : "") ||
      label ||
      "Live Audio";

    this.liveStream = stream;
    this.currentSourceMode = "live";
    this.currentDeviceId = settings.deviceId || deviceId || "";
    this.currentLabel = resolvedLabel;
    this.liveTrackSettings = {
      deviceId: this.currentDeviceId,
      label: resolvedLabel,
      sampleRate: settings.sampleRate || this.ctx.sampleRate,
      channelCount: settings.channelCount || 2,
      echoCancellation: Boolean(settings.echoCancellation),
      noiseSuppression: Boolean(settings.noiseSuppression),
      autoGainControl: Boolean(settings.autoGainControl),
    };

    this.liveElapsedMs = 0;
    this.liveResumeMs = performance.now();

    if (track) {
      track.onended = () => {
        if (this.liveStream === stream) {
          this._setPlaying(false);
          this.onLiveEnded(resolvedLabel);
        }
      };
    }

    const shouldMonitor =
      typeof monitorToSpeakers === "boolean" ? monitorToSpeakers : this.monitorLive;
    this.monitorLive = shouldMonitor;

    const streamSource = this.ctx.createMediaStreamSource(stream);
    // Live mode defaults to monitorToSpeakers: false (DJ setup already feeds speakers).
    this.connectSource(streamSource, { monitorToSpeakers: shouldMonitor });
    this.onControl({ type: "control", source: "live", label: this.currentLabel });
    this._setPlaying(true);

    return this.liveTrackSettings;
  }

  _stopLiveStream() {
    if (this.liveStream) {
      for (const track of this.liveStream.getTracks()) {
        track.onended = null;
        track.stop();
      }
      this.liveStream = null;
    }
  }

  _setPlaying(isPlaying) {
    const next = Boolean(isPlaying);
    if (this.currentSourceMode === "live") {
      const now = performance.now();
      if (this.playing && !next && this.liveResumeMs > 0) {
        this.liveElapsedMs += Math.max(0, now - this.liveResumeMs);
        this.liveResumeMs = 0;
      } else if (!this.playing && next) {
        this.liveResumeMs = now;
      }
    }
    this.playing = next;
    if (this.workletNode) {
      this.workletNode.port.postMessage({ playing: next });
    }
    this._emitClock(performance.now());
  }

  async play() {
    await this._ensureContext();
    if (this.currentSourceMode === "file") {
      if (!this.audioEl.src) return;
      await this.audioEl.play();
    } else if (this.liveStream) {
      this._setPlaying(true);
    }
  }

  pause() {
    if (this.currentSourceMode === "file") {
      this.audioEl.pause();
    } else {
      this._setPlaying(false);
    }
  }

  async togglePlay() {
    if (this.playing) {
      this.pause();
    } else {
      await this.play();
    }
  }

  getPlaybackTimeSec() {
    if (this.currentSourceMode === "live") {
      const running = this.playing && this.liveResumeMs > 0 ? performance.now() - this.liveResumeMs : 0;
      return Math.max(0, (this.liveElapsedMs + running) / 1000);
    }
    return Number.isFinite(this.audioEl.currentTime) ? this.audioEl.currentTime : 0;
  }

  getPcmChunksPerSec() {
    const now = performance.now();
    while (this.pcmChunkTimestamps.length > 0 && now - this.pcmChunkTimestamps[0] > 4000) {
      this.pcmChunkTimestamps.shift();
    }
    return this.pcmChunkTimestamps.length / 4.0;
  }

  update(nowMs) {
    const dt = this._lastUpdateMs > 0 ? Math.min(0.1, (nowMs - this._lastUpdateMs) / 1000) : 1 / 60;
    this._lastUpdateMs = nowMs;

    // Decay kick envelope smoothly over 120 ms
    this.kickEnvelope *= Math.exp(-dt / KICK_DECAY_TAU_SEC);

    if (!this.analyser || !this.playing) {
      this.dsp.kick = false;
      this.dsp.energy *= Math.exp(-dt / 0.25);
      this.inputLevel *= Math.exp(-dt / 0.15);
      this.rawRms = 0;
      this.liveSilentSec = 0;
      return this.dsp;
    }

    this.analyser.getFloatFrequencyData(this.freqData);
    this.analyser.getFloatTimeDomainData(this.timeData);

    const sr = this.ctx.sampleRate;
    const binHz = sr / FFT_SIZE;
    const nBins = this.freqData.length;

    let subSum = 0, subCnt = 0;
    let bassSum = 0, bassCnt = 0;
    let midSum = 0, midCnt = 0;
    let highSum = 0, highCnt = 0;

    // Convert dB to linear magnitude for energy integration & low-band spectral flux
    const lowEndBin = Math.min(nBins, Math.ceil(250 / binHz));
    if (!this.prevLowSpectrum || this.prevLowSpectrum.length !== lowEndBin) {
      this.prevLowSpectrum = new Float32Array(lowEndBin);
    }

    let lowFlux = 0;
    for (let i = 1; i < nBins; i++) {
      const freq = i * binHz;
      const db = this.freqData[i];
      const mag = db > -100 ? Math.pow(10, db / 20) : 0;

      if (i < lowEndBin) {
        const diff = mag - this.prevLowSpectrum[i];
        if (diff > 0) lowFlux += diff;
        this.prevLowSpectrum[i] = mag;
      }

      if (freq < 60) {
        subSum += mag;
        subCnt++;
      } else if (freq < 250) {
        bassSum += mag;
        bassCnt++;
      } else if (freq < 2000) {
        midSum += mag;
        midCnt++;
      } else {
        highSum += mag;
        highCnt++;
      }
    }

    const rawSub = subCnt ? subSum / subCnt : 0;
    const rawBass = bassCnt ? bassSum / bassCnt : 0;
    const rawMid = midCnt ? midSum / midCnt : 0;
    const rawHigh = highCnt ? highSum / highCnt : 0;

    let sq = 0;
    for (let i = 0; i < this.timeData.length; i++) {
      const s = this.timeData[i];
      sq += s * s;
    }
    const rawRms = Math.sqrt(sq / this.timeData.length);
    this.rawRms = rawRms;

    // Log-scaled VU input meter (-60 dBFS .. 0 dBFS) & silence detection for live setup diagnostics
    const dbFs = rawRms > 1e-6 ? 20 * Math.log10(rawRms) : -96;
    this.inputDb = dbFs;
    const normDb = Math.max(0, Math.min(1, (dbFs + 60) / 60));
    const meterAlpha = normDb > this.inputLevel ? 0.45 : 0.14;
    this.inputLevel += meterAlpha * (normDb - this.inputLevel);

    if (this.currentSourceMode === "live") {
      if (rawRms < SILENCE_RMS_THRESHOLD) {
        this.liveSilentSec += dt;
      } else {
        this.liveSilentSec = 0;
      }
    } else {
      this.liveSilentSec = 0;
    }

    // Track a mildly smoothed band envelope for normalization so single 42 ms kick
    // transients don't crush a loud drop down to ~0.40, while slow decay (~180 s)
    // keeps 30-60 s quiet breakdowns authentically quiet (~0.20-0.30).
    const smoothAlpha = 1.0 - Math.exp(-dt / 0.25);
    if (!this.smoothBands) {
      this.smoothBands = { sub: rawSub, bass: rawBass, mid: rawMid, high: rawHigh, rms: rawRms };
    }
    const decay = Math.exp(-dt / 180.0);
    const norm = (val, key, floor) => {
      this.smoothBands[key] += smoothAlpha * (val - this.smoothBands[key]);
      const refVal = this.smoothBands[key];
      this.maxBands[key] = Math.max(refVal, this.maxBands[key] * decay, floor);
      return Math.min(1, refVal / this.maxBands[key]);
    };

    const sub = norm(rawSub, "sub", 0.0008);
    const bass = norm(rawBass, "bass", 0.0008);
    const mid = norm(rawMid, "mid", 0.0004);
    const high = norm(rawHigh, "high", 0.00015);
    const rms = norm(rawRms, "rms", 0.015);

    // Kick onset on low band: positive spectral flux above adaptive threshold, 120 ms refractory
    const avgFlux = lowFlux / Math.max(1, lowEndBin);
    this.fluxMean = this.fluxMean * 0.92 + avgFlux * 0.08;
    const tSec = nowMs / 1000;
    const isKick =
      avgFlux > Math.max(this.fluxMean * 1.45, 0.0006) &&
      (bass > 0.25 || sub > 0.25) &&
      tSec - this.lastKickSec >= KICK_REFRACTORY_SEC;

    if (isKick) {
      this.lastKickSec = tSec;
      this.kickEnvelope = 1.0;
      this.kicksTotal += 1;
      this._kickSinceLastClock = true;
    }

    const energy = 0.6 * bass + 0.4 * rms;
    const playbackT = this.getPlaybackTimeSec();

    if (bass > 0.35) {
      this.lastBassAboveSec = playbackT;
    }
    const bassGap = Math.max(0, playbackT - this.lastBassAboveSec);

    // Rolling 10 s history for energy_5s and trend
    this.energyHistory.push({ t: playbackT, e: energy });
    while (this.energyHistory.length > 0 && playbackT - this.energyHistory[0].t > 10.0) {
      this.energyHistory.shift();
    }
    let sumRecent = 0, cntRecent = 0;
    let sumPrev = 0, cntPrev = 0;
    for (let i = 0; i < this.energyHistory.length; i++) {
      const age = playbackT - this.energyHistory[i].t;
      if (age <= 5.0) {
        sumRecent += this.energyHistory[i].e;
        cntRecent++;
      } else {
        sumPrev += this.energyHistory[i].e;
        cntPrev++;
      }
    }
    const energy5s = cntRecent ? sumRecent / cntRecent : energy;
    const prev5s = cntPrev ? sumPrev / cntPrev : energy5s;
    const trend = energy5s - prev5s;

    this.dsp = {
      sub: +sub.toFixed(3),
      bass: +bass.toFixed(3),
      mid: +mid.toFixed(3),
      high: +high.toFixed(3),
      rms: +rms.toFixed(3),
      kick: isKick,
      energy: +energy.toFixed(3),
      energy_5s: +energy5s.toFixed(3),
      trend: +trend.toFixed(3),
      bass_gap: +bassGap.toFixed(2),
    };

    if (nowMs - this._lastClockSentMs >= CLOCK_INTERVAL_MS) {
      this._emitClock(nowMs);
    }

    return this.dsp;
  }

  _emitClock(nowMs) {
    this._lastClockSentMs = nowMs;
    const dspPayload = {
      ...this.dsp,
      kick: this._kickSinceLastClock || this.dsp.kick,
    };
    this._kickSinceLastClock = false;
    this.onClock({
      type: "clock",
      t: +this.getPlaybackTimeSec().toFixed(3),
      playing: this.playing,
      source: this.currentSourceMode,
      dsp: dspPayload,
    });
  }
}
