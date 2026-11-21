// Shared Web Audio analysis (60 fps FFT features) + 16 kHz mono Int16 AudioWorklet capture.
// Designed so input source (MP3 file vs. live getUserMedia in Phase L) is a single swappable function.

const FFT_SIZE = 2048;
const KICK_REFRACTORY_SEC = 0.12;
const KICK_DECAY_TAU_SEC = 0.12;
const CLOCK_INTERVAL_MS = 250; // 4 Hz

export class AudioEngine {
  constructor({ onClock, onPcmChunk, onControl } = {}) {
    this.onClock = onClock || (() => {});
    this.onPcmChunk = onPcmChunk || (() => {});
    this.onControl = onControl || (() => {});

    this.ctx = null;
    this.analyser = null;
    this.workletNode = null;
    this.workletReady = false;

    this.currentSourceNode = null;
    this.currentSourceMode = "file"; // "file" | "live"
    this.currentLabel = "";
    this.liveStream = null;

    this.audioEl = document.createElement("audio");
    this.audioEl.crossOrigin = "anonymous";
    this.audioEl.preload = "auto";
    this.audioEl.loop = false;
    this.mediaElSource = null;

    this.playing = false;
    this.sessionStartMs = 0;

    // FFT buffers
    this.freqData = null;
    this.timeData = null;
    this.prevLowSpectrum = null;

    // Decaying running maxima for normalization
    this.maxBands = { sub: 0.002, bass: 0.002, mid: 0.001, high: 0.0004, rms: 0.03 };
    this.fluxMean = 0.002;
    this.lastKickSec = -10;
    this.kickEnvelope = 0;
    this.lastBassAboveSec = 0;

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

    // Telemetry for Phase A verification & stats corner
    this.pcmChunksTotal = 0;
    this.pcmChunkTimestamps = [];
    this.lastChunkBytes = 0;
    this.kicksTotal = 0;
    this._lastClockSentMs = 0;
    this._lastUpdateMs = 0;
    this._kickSinceLastClock = false;

    this._bindAudioElementEvents();
  }

  _bindAudioElementEvents() {
    this.audioEl.addEventListener("play", () => {
      this._setPlaying(true);
    });
    this.audioEl.addEventListener("pause", () => {
      this._setPlaying(false);
    });
    this.audioEl.addEventListener("ended", () => {
      this._setPlaying(false);
    });
    this.audioEl.addEventListener("seeked", () => {
      if (this.workletNode) {
        this.workletNode.port.postMessage({ reset: true });
      }
      this.energyHistory = [];
      this.lastBassAboveSec = this.getPlaybackTimeSec();
      this.onControl({ type: "control", action: "seek" });
    });
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

      if (this.ctx.audioWorklet) {
        try {
          await this.ctx.audioWorklet.addModule("/pcm-worklet.js");
          this.workletNode = new AudioWorkletNode(this.ctx, "pcm-capture-processor", {
            numberOfInputs: 1,
            numberOfOutputs: 1,
            channelCount: 2,
          });
          // Route silent output so Chromium/WebKit always pulls process() frames
          const silentGain = this.ctx.createGain();
          silentGain.gain.value = 0;
          this.workletNode.connect(silentGain);
          silentGain.connect(this.ctx.destination);

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
    if (monitorToSpeakers) {
      sourceNode.connect(this.ctx.destination);
    }
  }

  async loadFile(url, label = "") {
    await this._ensureContext();
    this._stopLiveStream();

    this.currentSourceMode = "file";
    this.currentLabel = label || url.split("/").pop() || "file";

    if (!this.mediaElSource) {
      this.mediaElSource = this.ctx.createMediaElementSource(this.audioEl);
    }
    this.connectSource(this.mediaElSource, { monitorToSpeakers: true });

    this.audioEl.pause();
    this.audioEl.src = url;
    this.audioEl.load();
    this.energyHistory = [];
    this.onControl({ type: "control", source: "file", label: this.currentLabel });
  }

  async loadLiveDevice(deviceId, label = "live") {
    await this._ensureContext();
    this.audioEl.pause();
    this._stopLiveStream();

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    this.liveStream = stream;
    this.currentSourceMode = "live";
    this.currentLabel = label;
    this.sessionStartMs = performance.now();

    const streamSource = this.ctx.createMediaStreamSource(stream);
    // Live mode does NOT play out of the browser (DJ setup already feeds speakers).
    this.connectSource(streamSource, { monitorToSpeakers: false });
    this._setPlaying(true);
    this.onControl({ type: "control", source: "live", label: this.currentLabel });
  }

  _stopLiveStream() {
    if (this.liveStream) {
      for (const track of this.liveStream.getTracks()) track.stop();
      this.liveStream = null;
    }
  }

  _setPlaying(isPlaying) {
    this.playing = isPlaying;
    if (this.workletNode) {
      this.workletNode.port.postMessage({ playing: isPlaying });
    }
    this._emitClock(performance.now());
  }

  async play() {
    await this._ensureContext();
    if (this.currentSourceMode === "file") {
      if (!this.audioEl.src) return;
      await this.audioEl.play();
    } else {
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
      return this.sessionStartMs > 0 ? (performance.now() - this.sessionStartMs) / 1000 : 0;
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

    // Normalize each band by a decaying running max (~10 s decay)
    const decay = Math.exp(-dt / 10.0);
    const norm = (val, key, floor) => {
      this.maxBands[key] = Math.max(val, this.maxBands[key] * decay, floor);
      return Math.min(1, val / this.maxBands[key]);
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
