// Minimal keyboard controls + toggleable tiny stats corner (D) + auto-hiding input picker (I).
// Visual TLC plate remains unobstructed as the main thing on screen (plan.md section 10).

export class CrowdUI {
  constructor({ onTogglePlay, onCondition, onToggleLabels, onToggleChat, onSelectFile } = {}) {
    this.onTogglePlay = onTogglePlay || (() => {});
    this.onCondition = onCondition || (() => {});
    this.onToggleLabels = onToggleLabels || (() => {});
    this.onToggleChat = onToggleChat || (() => {});
    this.onSelectFile = onSelectFile || (() => {});

    this.showStats = false;
    this.showPicker = false;
    this.chatOpen = false;
    this.tracks = [];
    this.selectedTrack = "";

    this._injectStyles();
    this._buildDom();
    this._bindKeys();
    this.refreshTracks();
  }

  _injectStyles() {
    const style = document.createElement("style");
    style.textContent = `
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
        top: 12px;
        left: 12px;
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
        right: 12px;
        font: 10px "IBM Plex Mono", monospace;
        color: rgba(55, 45, 48, 0.45);
        pointer-events: none;
        z-index: 5;
        transition: opacity 0.4s ease;
      }
      .crowd-hint.quiet { opacity: 0.18; }
      .crowd-stats.chat-open { right: 292px; }
      .crowd-hint.chat-open { right: 292px; }
    `;
    document.head.appendChild(style);
  }

  _buildDom() {
    this.statsEl = document.createElement("div");
    this.statsEl.className = "crowd-stats";
    document.body.appendChild(this.statsEl);

    this.pickerEl = document.createElement("div");
    this.pickerEl.className = "crowd-picker";
    document.body.appendChild(this.pickerEl);

    this.hintEl = document.createElement("div");
    this.hintEl.className = "crowd-hint";
    this.hintEl.textContent = "Space play · 1/2/3 cond · C chat · L labels · D stats · I input";
    document.body.appendChild(this.hintEl);
  }

  setChatOpen(open) {
    this.chatOpen = Boolean(open);
    this.statsEl.classList.toggle("chat-open", this.chatOpen);
    this.hintEl.classList.toggle("chat-open", this.chatOpen);
  }

  _bindKeys() {
    window.addEventListener("keydown", (e) => {
      if (e.target && ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
      if (e.code === "Space") {
        e.preventDefault();
        this.onTogglePlay();
      } else if (e.key === "1") {
        this.onCondition("shove");
      } else if (e.key === "2") {
        this.onCondition("silent");
      } else if (e.key === "3") {
        this.onCondition("talk");
      } else if (e.key === "c" || e.key === "C") {
        this.onToggleChat();
      } else if (e.key === "l" || e.key === "L") {
        this.onToggleLabels();
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
        // Hide picker immediately after selection so plate fills the screen
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
