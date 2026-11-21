// Minimal keyboard controls + top-left mode banner + bottom-left legend + stats corner (D) + input picker (I).
// Supports both live getUserMedia devices (BlackHole 2ch, USB mixer, mic) and file tracks in tracks/.
// Visual TLC plate remains unobstructed as the main thing on screen.

const MODE_DESCRIPTIONS = {
  shove: {
    label: "SHOVE",
    desc: "scripted crowd, no AI",
  },
  silent: {
    label: "SILENT",
    desc: "each agent listens and decides with Gemini",
  },
  talk: {
    label: "TALK",
    desc: "agents decide and argue to get past each other",
  },
};

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export class CrowdUI {
  constructor({
    onTogglePlay,
    onCondition,
    onToggleLabels,
    onToggleChat,
    onReset,
    onSelectFile,
    onSelectLive,
    onRequestLiveDevices,
    onToggleMonitor,
  } = {}) {
    this.onTogglePlay = onTogglePlay || (() => {});
    this.onCondition = onCondition || (() => {});
    this.onToggleLabels = onToggleLabels || (() => {});
    this.onToggleChat = onToggleChat || (() => {});
    this.onReset = onReset || (() => {});
    this.onSelectFile = onSelectFile || (() => {});
    this.onSelectLive = onSelectLive || (() => {});
    this.onRequestLiveDevices = onRequestLiveDevices || (async () => ({ supported: false, devices: [], hasLabels: false, error: "" }));
    this.onToggleMonitor = onToggleMonitor || (() => {});

    this.showStats = false;
    this.showPicker = false;
    this.chatOpen = false;
    this.condition = "shove";
    this._bannerTimer = null;

    this.tracks = [];
    this.liveDevices = [];
    this.liveSupported = true;
    this.liveHasLabels = false;
    this.liveError = "";
    this.liveMonitor = false;
    this.liveTrackSettings = null;

    this.sourceMode = "file"; // "file" | "live"
    this.selectedTrack = "";
    this.selectedDeviceId = "";
    this.sourceLabel = "";
    this._liveSilentWarning = false;
    this._livePlaying = false;
    this._meterFillEl = null;
    this._meterDbEl = null;

    this._injectStyles();
    this._buildDom();
    this._bindKeys();
    this.setCondition(this.condition, false);
    this.refreshInputs(false);
  }

  _injectStyles() {
    const style = document.createElement("style");
    style.textContent = `
      .crowd-banner {
        position: fixed;
        top: 8px;
        left: 12px;
        padding: 4px 9px;
        background: rgba(246, 235, 230, 0.76);
        border: 1px solid rgba(55, 45, 48, 0.16);
        border-left: 2.5px solid rgba(118, 52, 92, 0.52);
        color: rgba(42, 34, 38, 0.86);
        font: 10.5px/1.35 "IBM Plex Mono", monospace;
        pointer-events: none;
        z-index: 12;
        white-space: nowrap;
        transition: background 0.45s ease, border-color 0.45s ease, box-shadow 0.45s ease;
      }
      .crowd-banner.highlight {
        background: rgba(252, 234, 224, 0.96);
        border-color: rgba(196, 62, 68, 0.55);
        border-left-color: rgba(212, 66, 38, 0.92);
        box-shadow: 0 2px 12px rgba(192, 62, 72, 0.18);
      }
      .crowd-banner .mode-name {
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #2c1e26;
      }
      .crowd-banner .mode-sep {
        opacity: 0.45;
        margin: 0 6px;
      }
      .crowd-banner .mode-desc {
        color: rgba(52, 42, 47, 0.78);
      }
      .crowd-banner .live-badge {
        color: #b53418;
        font-weight: 600;
      }
      .crowd-banner .live-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #d9481c;
        margin-right: 5px;
        vertical-align: 1px;
      }
      .crowd-banner .live-dot.paused {
        background: rgba(90, 78, 84, 0.45);
      }
      .crowd-banner .live-warn {
        color: #8f3b2e;
        font-weight: 400;
        opacity: 0.85;
      }

      .crowd-stats {
        position: fixed;
        top: 12px;
        right: 12px;
        padding: 8px 10px;
        background: rgba(246, 235, 230, 0.86);
        border: 1px solid rgba(55, 45, 48, 0.22);
        color: #2b2327;
        font: 11px/1.45 "IBM Plex Mono", monospace;
        pointer-events: none;
        display: none;
        z-index: 10;
        min-width: 185px;
      }
      .crowd-stats.visible { display: block; }
      .crowd-stats .row { display: flex; justify-content: space-between; gap: 12px; }
      .crowd-stats .muted { opacity: 0.6; }

      .crowd-picker {
        position: fixed;
        top: 44px;
        left: 12px;
        padding: 10px 12px;
        background: rgba(246, 235, 230, 0.95);
        border: 1px solid rgba(55, 45, 48, 0.28);
        box-shadow: 0 6px 24px rgba(40, 26, 32, 0.12);
        color: #2b2327;
        font: 11px/1.5 "IBM Plex Mono", monospace;
        display: none;
        z-index: 20;
        min-width: 280px;
        max-width: 360px;
        max-height: calc(100vh - 80px);
        overflow-y: auto;
      }
      .crowd-picker.visible { display: block; }
      .crowd-picker .picker-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin: 0 0 6px 0;
      }
      .crowd-picker h4 {
        margin: 0;
        font-size: 10.5px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        opacity: 0.72;
      }
      .crowd-picker .section-gap {
        margin-top: 10px;
        padding-top: 8px;
        border-top: 1px solid rgba(55, 45, 48, 0.14);
      }
      .crowd-picker button.item-btn {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        width: 100%;
        text-align: left;
        background: transparent;
        border: 1px solid transparent;
        padding: 4px 6px;
        margin: 2px 0;
        font: 11px "IBM Plex Mono", monospace;
        color: #2b2327;
        cursor: pointer;
      }
      .crowd-picker button.item-btn:hover {
        background: rgba(138, 63, 160, 0.12);
        border-color: rgba(138, 63, 160, 0.35);
      }
      .crowd-picker button.item-btn.active {
        font-weight: 600;
        background: rgba(217, 72, 28, 0.10);
        border-color: rgba(217, 72, 28, 0.45);
      }
      .crowd-picker button.mini-btn {
        background: rgba(55, 45, 48, 0.07);
        border: 1px solid rgba(55, 45, 48, 0.22);
        padding: 2px 6px;
        font: 10px "IBM Plex Mono", monospace;
        color: #2b2327;
        cursor: pointer;
      }
      .crowd-picker button.mini-btn:hover {
        background: rgba(138, 63, 160, 0.14);
      }
      .crowd-picker .dev-tag {
        font-size: 9.5px;
        color: #b53418;
        opacity: 0.85;
        white-space: nowrap;
      }
      .crowd-picker .picker-note {
        font-size: 10px;
        opacity: 0.68;
        margin: 4px 0;
        line-height: 1.4;
      }
      .crowd-picker .picker-err {
        font-size: 10px;
        color: #b02828;
        background: rgba(176, 40, 40, 0.08);
        border: 1px solid rgba(176, 40, 40, 0.22);
        padding: 4px 6px;
        margin: 4px 0 6px 0;
      }
      .crowd-picker .live-meter-wrap {
        margin: 6px 0 4px 0;
        padding: 5px 6px;
        background: rgba(55, 45, 48, 0.05);
        border: 1px solid rgba(55, 45, 48, 0.14);
      }
      .crowd-picker .live-meter-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 10px;
        margin-bottom: 4px;
      }
      .crowd-picker .live-meter-bar {
        width: 100%;
        height: 5px;
        background: rgba(55, 45, 48, 0.14);
        overflow: hidden;
      }
      .crowd-picker .live-meter-fill {
        height: 100%;
        width: 0%;
        background: linear-gradient(90deg, #6b6eb8, #d12f7a 65%, #d9481c);
        transition: width 0.06s linear;
      }
      .crowd-picker .monitor-row {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 6px;
        font-size: 10px;
        opacity: 0.82;
        cursor: pointer;
      }

      .crowd-hint {
        position: fixed;
        bottom: 10px;
        left: 14px;
        font: 10.5px "IBM Plex Mono", monospace;
        color: rgba(52, 42, 47, 0.58);
        pointer-events: none;
        z-index: 8;
        transition: opacity 0.4s ease;
      }
      .crowd-hint.quiet { opacity: 0.36; }
      .crowd-stats.chat-open { right: 292px; }
    `;
    document.head.appendChild(style);
  }

  _buildDom() {
    this.bannerEl = document.createElement("div");
    this.bannerEl.className = "crowd-banner";
    document.body.appendChild(this.bannerEl);

    this.statsEl = document.createElement("div");
    this.statsEl.className = "crowd-stats";
    document.body.appendChild(this.statsEl);

    this.pickerEl = document.createElement("div");
    this.pickerEl.className = "crowd-picker";
    document.body.appendChild(this.pickerEl);

    this.hintEl = document.createElement("div");
    this.hintEl.className = "crowd-hint";
    this.hintEl.textContent =
      "space play · 1 shove · 2 silent · 3 talk · C chat · L labels · D stats · I input · R reset";
    document.body.appendChild(this.hintEl);
  }

  _renderBanner() {
    const meta = MODE_DESCRIPTIONS[this.condition] || MODE_DESCRIPTIONS.shove;
    let liveHtml = "";
    if (this.sourceMode === "live") {
      const dotCls = this._livePlaying ? "live-dot" : "live-dot paused";
      const labelShort = escapeHtml((this.sourceLabel || "Live").slice(0, 26));
      const warn =
        this._livePlaying && this._liveSilentWarning
          ? ` <span class="live-warn">(no signal — check BlackHole / Multi-Output)</span>`
          : "";
      liveHtml =
        `<span class="mode-sep">·</span>` +
        `<span class="live-badge"><span class="${dotCls}"></span>LIVE: ${labelShort}</span>${warn}`;
    }
    this.bannerEl.innerHTML =
      `<span class="mode-name">${meta.label}:</span> ` +
      `<span class="mode-desc">${meta.desc}</span>${liveHtml}`;
  }

  setCondition(cond, highlight = true) {
    const key = String(cond || "shove").toLowerCase();
    const changed = key !== this.condition;
    this.condition = key;
    this._renderBanner();

    if (highlight && changed) {
      this.bannerEl.classList.remove("highlight");
      void this.bannerEl.offsetWidth;
      this.bannerEl.classList.add("highlight");
      if (this._bannerTimer) clearTimeout(this._bannerTimer);
      this._bannerTimer = window.setTimeout(() => {
        this.bannerEl.classList.remove("highlight");
      }, 1500);
    }
  }

  setSource({ mode, label, deviceId = "", trackSettings = null, error = "" }) {
    this.sourceMode = mode || "file";
    this.sourceLabel = label || "";
    this.selectedDeviceId = mode === "live" ? deviceId : "";
    if (mode === "file") {
      this.selectedTrack = label || "";
    }
    this.liveTrackSettings = mode === "live" ? trackSettings : null;
    this.liveError = error || "";
    this._liveSilentWarning = false;
    this._renderBanner();
    if (this.showPicker) {
      this._renderPicker();
    }
  }

  setLiveError(message) {
    this.liveError = message || "";
    if (this.showPicker) {
      this._renderPicker();
    }
  }

  updateLiveStatus({ playing, inputLevel = 0, inputDb = -96, silentSec = 0 }) {
    const playingChanged = Boolean(playing) !== this._livePlaying;
    const silentWarn = this.sourceMode === "live" && Boolean(playing) && silentSec >= 3.0;
    const silentChanged = silentWarn !== this._liveSilentWarning;

    this._livePlaying = Boolean(playing);
    this._liveSilentWarning = silentWarn;

    if (this.sourceMode === "live" && (playingChanged || silentChanged)) {
      this._renderBanner();
    }

    if (this.showPicker && this._meterFillEl) {
      this._meterFillEl.style.width = `${Math.round(Math.max(0, Math.min(1, inputLevel)) * 100)}%`;
      if (this._meterDbEl) {
        this._meterDbEl.textContent = inputDb > -90 ? `${inputDb.toFixed(0)} dBFS` : "silence";
      }
    }
  }

  setChatOpen(open) {
    this.chatOpen = Boolean(open);
    this.statsEl.classList.toggle("chat-open", this.chatOpen);
  }

  _bindKeys() {
    window.addEventListener("keydown", (e) => {
      if (e.target && ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
      if (e.code === "Space") {
        e.preventDefault();
        this.onTogglePlay();
      } else if (e.key === "1") {
        this.setCondition("shove", true);
        this.onCondition("shove");
      } else if (e.key === "2") {
        this.setCondition("silent", true);
        this.onCondition("silent");
      } else if (e.key === "3") {
        this.setCondition("talk", true);
        this.onCondition("talk");
      } else if (e.key === "c" || e.key === "C") {
        this.onToggleChat();
      } else if (e.key === "l" || e.key === "L") {
        this.onToggleLabels();
      } else if (e.key === "r" || e.key === "R") {
        this.onReset();
      } else if (e.key === "d" || e.key === "D") {
        this.showStats = !this.showStats;
        this.statsEl.classList.toggle("visible", this.showStats);
      } else if (e.key === "i" || e.key === "I") {
        this.showPicker = !this.showPicker;
        this.pickerEl.classList.toggle("visible", this.showPicker);
        if (this.showPicker) this.refreshInputs(false);
      } else if (e.key === "Escape" && this.showPicker) {
        this.showPicker = false;
        this.pickerEl.classList.remove("visible");
      }
    });
  }

  async refreshTracks() {
    try {
      const res = await fetch("/api/tracks");
      if (res.ok) {
        const data = await res.json();
        this.tracks = Array.isArray(data.tracks) ? data.tracks : [];
      }
    } catch {
      this.tracks = [];
    }
    this._renderPicker();
    return this.tracks;
  }

  async refreshLiveDevices(requestPermission = false) {
    const res = await this.onRequestLiveDevices(requestPermission);
    this.liveSupported = Boolean(res && res.supported);
    this.liveDevices = Array.isArray(res && res.devices) ? res.devices : [];
    this.liveHasLabels = Boolean(res && res.hasLabels);
    if (res && res.error) {
      this.liveError = res.error;
    } else if (requestPermission) {
      this.liveError = "";
    }
    this._renderPicker();
    return this.liveDevices;
  }

  async refreshInputs(requestPermission = false) {
    await Promise.all([this.refreshTracks(), this.refreshLiveDevices(requestPermission)]);
  }

  hidePicker() {
    this.showPicker = false;
    this.pickerEl.classList.remove("visible");
  }

  _renderPicker() {
    this.pickerEl.innerHTML = "";
    this._meterFillEl = null;
    this._meterDbEl = null;

    // --- Section 1: Live Audio Inputs (BlackHole 2ch, USB DJ mixer, mic) ---
    const liveHead = document.createElement("div");
    liveHead.className = "picker-head";
    const liveTitle = document.createElement("h4");
    liveTitle.textContent = "Live Input (BlackHole / Mixer)";
    liveHead.appendChild(liveTitle);

    const refreshBtn = document.createElement("button");
    refreshBtn.className = "mini-btn";
    refreshBtn.textContent = this.liveHasLabels ? "Refresh" : "Unlock names";
    refreshBtn.title = "Request mic permission to show exact hardware names (e.g. BlackHole 2ch)";
    refreshBtn.addEventListener("click", () => {
      this.refreshLiveDevices(true);
    });
    liveHead.appendChild(refreshBtn);
    this.pickerEl.appendChild(liveHead);

    if (this.liveError) {
      const errEl = document.createElement("div");
      errEl.className = "picker-err";
      errEl.textContent = this.liveError;
      this.pickerEl.appendChild(errEl);
    }

    if (!this.liveSupported) {
      const note = document.createElement("div");
      note.className = "picker-note";
      note.textContent = "Live capture requires localhost or HTTPS.";
      this.pickerEl.appendChild(note);
    } else if (this.liveDevices.length === 0) {
      const note = document.createElement("div");
      note.className = "picker-note";
      note.textContent = "No audio inputs detected. Click 'Unlock names' to grant permission.";
      this.pickerEl.appendChild(note);
    } else {
      if (!this.liveHasLabels) {
        const note = document.createElement("div");
        note.className = "picker-note";
        note.textContent = "Click 'Unlock names' once to reveal BlackHole / USB mixer labels.";
        this.pickerEl.appendChild(note);
      }

      for (const dev of this.liveDevices) {
        const btn = document.createElement("button");
        btn.className = "item-btn";
        const isActive =
          this.sourceMode === "live" &&
          ((dev.deviceId && dev.deviceId === this.selectedDeviceId) ||
            (dev.label && dev.label === this.sourceLabel));
        if (isActive) btn.classList.add("active");

        const lblSpan = document.createElement("span");
        lblSpan.textContent = dev.label;
        btn.appendChild(lblSpan);

        if (dev.recommended) {
          const tag = document.createElement("span");
          tag.className = "dev-tag";
          tag.textContent = "[DJ / loopback]";
          btn.appendChild(tag);
        }

        btn.addEventListener("click", async () => {
          this.liveError = "";
          try {
            await this.onSelectLive(dev.deviceId, dev.label, { monitorToSpeakers: this.liveMonitor });
            this.hidePicker();
          } catch (err) {
            this.liveError = err && err.message ? err.message : String(err);
            this._renderPicker();
          }
        });
        this.pickerEl.appendChild(btn);
      }
    }

    // Active live input diagnostics + optional speaker monitor toggle (off by default)
    if (this.sourceMode === "live") {
      const wrap = document.createElement("div");
      wrap.className = "live-meter-wrap";

      const row = document.createElement("div");
      row.className = "live-meter-row";
      const s = this.liveTrackSettings;
      const dspOk = s && !s.echoCancellation && !s.noiseSuppression && !s.autoGainControl;
      const metaSpan = document.createElement("span");
      metaSpan.textContent = dspOk ? "raw audio (EC/NS/AGC off)" : "live level";
      const dbSpan = document.createElement("span");
      dbSpan.textContent = "—";
      this._meterDbEl = dbSpan;
      row.appendChild(metaSpan);
      row.appendChild(dbSpan);
      wrap.appendChild(row);

      const bar = document.createElement("div");
      bar.className = "live-meter-bar";
      const fill = document.createElement("div");
      fill.className = "live-meter-fill";
      this._meterFillEl = fill;
      bar.appendChild(fill);
      wrap.appendChild(bar);

      const monLabel = document.createElement("label");
      monLabel.className = "monitor-row";
      const monBox = document.createElement("input");
      monBox.type = "checkbox";
      monBox.checked = this.liveMonitor;
      monBox.addEventListener("change", () => {
        this.liveMonitor = monBox.checked;
        this.onToggleMonitor(this.liveMonitor);
      });
      monLabel.appendChild(monBox);
      monLabel.appendChild(document.createTextNode("Monitor in browser (off for DJ speakers)"));
      wrap.appendChild(monLabel);

      this.pickerEl.appendChild(wrap);
    }

    // --- Section 2: Dev / Replay Files (tracks/) ---
    const fileHead = document.createElement("div");
    fileHead.className = "picker-head section-gap";
    const fileTitle = document.createElement("h4");
    fileTitle.textContent = "File Playback (tracks/)";
    fileHead.appendChild(fileTitle);
    this.pickerEl.appendChild(fileHead);

    if (this.tracks.length === 0) {
      const empty = document.createElement("div");
      empty.className = "picker-note";
      empty.textContent = "No MP3/WAV files found in tracks/";
      this.pickerEl.appendChild(empty);
      return;
    }

    for (const file of this.tracks) {
      const btn = document.createElement("button");
      btn.className = "item-btn";
      btn.textContent = file;
      if (this.sourceMode === "file" && file === this.selectedTrack) {
        btn.classList.add("active");
      }
      btn.addEventListener("click", () => {
        this.selectedTrack = file;
        this.hidePicker();
        this.onSelectFile(`/tracks/${encodeURIComponent(file)}`, file);
      });
      this.pickerEl.appendChild(btn);
    }
  }

  setPlaying(playing) {
    this.hintEl.classList.toggle("quiet", Boolean(playing));
    if (this.sourceMode === "live" && Boolean(playing) !== this._livePlaying) {
      this._livePlaying = Boolean(playing);
      this._renderBanner();
    }
  }

  updateStats({
    meanH,
    meanE,
    asksPerTick,
    fps,
    condition,
    playing,
    energy,
    pcmRate,
    pcmTotal,
    track,
    sourceMode,
    inputDb,
  }) {
    if (!this.showStats) return;
    const srcText =
      sourceMode === "live"
        ? `LIVE ${(track || "").slice(0, 14)}`
        : (track || "").slice(0, 18);
    const levelRow =
      sourceMode === "live"
        ? `<div class="row"><span class="muted">in level</span><span>${inputDb > -90 ? `${inputDb.toFixed(0)} dBFS` : "silence"}</span></div>`
        : "";
    this.statsEl.innerHTML = `
      <div class="row"><span class="muted">cond</span><span>${condition}</span></div>
      <div class="row"><span class="muted">mean H</span><span>${meanH.toFixed(3)}</span></div>
      <div class="row"><span class="muted">mean E</span><span>${meanE.toFixed(3)}</span></div>
      <div class="row"><span class="muted">asks/tick</span><span>${asksPerTick}</span></div>
      <div class="row"><span class="muted">fps</span><span>${fps.toFixed(0)}</span></div>
      <div class="row"><span class="muted">audio</span><span>${playing ? `▶ ${energy.toFixed(2)}` : "❚❚ paused"}</span></div>
      ${levelRow}
      <div class="row"><span class="muted">pcm 16k</span><span>${pcmRate.toFixed(1)}/s (${pcmTotal})</span></div>
      ${srcText ? `<div class="row"><span class="muted">src</span><span>${escapeHtml(srcText)}</span></div>` : ""}
    `;
  }
}
