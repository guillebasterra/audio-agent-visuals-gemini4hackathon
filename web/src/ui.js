// Minimal keyboard controls + top-left mode banner + bottom-left legend + stats corner (D) + input picker (I).
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

export class CrowdUI {
  constructor({ onTogglePlay, onCondition, onToggleLabels, onToggleChat, onReset, onSelectFile } = {}) {
    this.onTogglePlay = onTogglePlay || (() => {});
    this.onCondition = onCondition || (() => {});
    this.onToggleLabels = onToggleLabels || (() => {});
    this.onToggleChat = onToggleChat || (() => {});
    this.onReset = onReset || (() => {});
    this.onSelectFile = onSelectFile || (() => {});

    this.showStats = false;
    this.showPicker = false;
    this.chatOpen = false;
    this.condition = "shove";
    this._bannerTimer = null;
    this.tracks = [];
    this.selectedTrack = "";

    this._injectStyles();
    this._buildDom();
    this._bindKeys();
    this.setCondition(this.condition, false);
    this.refreshTracks();
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
        margin: 0 5px;
      }
      .crowd-banner .mode-desc {
        color: rgba(52, 42, 47, 0.78);
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
        min-width: 170px;
      }
      .crowd-stats.visible { display: block; }
      .crowd-stats .row { display: flex; justify-content: space-between; gap: 12px; }
      .crowd-stats .muted { opacity: 0.6; }

      .crowd-picker {
        position: fixed;
        top: 56px;
        left: 14px;
        padding: 10px 12px;
        background: rgba(246, 235, 230, 0.94);
        border: 1px solid rgba(55, 45, 48, 0.28);
        color: #2b2327;
        font: 11px/1.5 "IBM Plex Mono", monospace;
        display: none;
        z-index: 20;
        min-width: 240px;
        max-width: 340px;
      }
      .crowd-picker.visible { display: block; }
      .crowd-picker h4 {
        margin: 0 0 6px 0;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        opacity: 0.72;
      }
      .crowd-picker button {
        display: block;
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
      .crowd-picker button:hover {
        background: rgba(138, 63, 160, 0.12);
        border-color: rgba(138, 63, 160, 0.35);
      }
      .crowd-picker button.active {
        font-weight: 600;
        border-color: rgba(55, 45, 48, 0.35);
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

  setCondition(cond, highlight = true) {
    const key = String(cond || "shove").toLowerCase();
    const meta = MODE_DESCRIPTIONS[key] || MODE_DESCRIPTIONS.shove;
    const changed = key !== this.condition;
    this.condition = key;
    this.bannerEl.innerHTML =
      `<span class="mode-name">${meta.label}:</span> ` +
      `<span class="mode-desc">${meta.desc}</span>`;

    if (highlight && changed) {
      this.bannerEl.classList.remove("highlight");
      // Force reflow so rapid switches re-trigger the highlight cleanly
      void this.bannerEl.offsetWidth;
      this.bannerEl.classList.add("highlight");
      if (this._bannerTimer) clearTimeout(this._bannerTimer);
      this._bannerTimer = window.setTimeout(() => {
        this.bannerEl.classList.remove("highlight");
      }, 1500);
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
        if (this.showPicker) this.refreshTracks();
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

  _renderPicker() {
    this.pickerEl.innerHTML = "";
    const title = document.createElement("h4");
    title.textContent = "Audio Input (tracks/)";
    this.pickerEl.appendChild(title);

    if (this.tracks.length === 0) {
      const empty = document.createElement("div");
      empty.textContent = "No MP3/WAV files found in tracks/";
      empty.style.opacity = "0.65";
      this.pickerEl.appendChild(empty);
      return;
    }

    for (const file of this.tracks) {
      const btn = document.createElement("button");
      btn.textContent = file;
      if (file === this.selectedTrack) btn.classList.add("active");
      btn.addEventListener("click", () => {
        this.selectedTrack = file;
        this.showPicker = false;
        this.pickerEl.classList.remove("visible");
        this.onSelectFile(`/tracks/${encodeURIComponent(file)}`, file);
      });
      this.pickerEl.appendChild(btn);
    }
  }

  setPlaying(playing) {
    this.hintEl.classList.toggle("quiet", Boolean(playing));
  }

  updateStats({ meanH, meanE, asksPerTick, fps, condition, playing, energy, pcmRate, pcmTotal, track }) {
    if (!this.showStats) return;
    this.statsEl.innerHTML = `
      <div class="row"><span class="muted">cond</span><span>${condition}</span></div>
      <div class="row"><span class="muted">mean H</span><span>${meanH.toFixed(3)}</span></div>
      <div class="row"><span class="muted">mean E</span><span>${meanE.toFixed(3)}</span></div>
      <div class="row"><span class="muted">asks/tick</span><span>${asksPerTick}</span></div>
      <div class="row"><span class="muted">fps</span><span>${fps.toFixed(0)}</span></div>
      <div class="row"><span class="muted">audio</span><span>${playing ? `▶ ${energy.toFixed(2)}` : "❚❚ paused"}</span></div>
      <div class="row"><span class="muted">pcm 16k</span><span>${pcmRate.toFixed(1)}/s (${pcmTotal})</span></div>
      ${track ? `<div class="row"><span class="muted">src</span><span>${track.slice(0, 18)}</span></div>` : ""}
    `;
  }
}
