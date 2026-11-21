// Live Twitch-style chat feed panel on the right edge (~280 px, IBM Plex Mono 11 px).
// Toggle with C. Never covers the crowd (plate resizes cleanly into the remaining area).

export const CHAT_PANEL_WIDTH = 280;
const MAX_LINES_PER_TICK = 6;
const MAX_PENDING_ASKS_PER_TICK = 2;
const MAX_HISTORY_LINES = 140;

function formatDropEta(eta) {
  if (!eta) return "not soon";
  return String(eta).replace(/_/g, " ");
}

export class ChatFeed {
  constructor({ onToggle, onHoverAgents } = {}) {
    this.onToggle = onToggle || (() => {});
    this.onHoverAgents = onHoverAgents || (() => {});

    this.isOpen = false;
    this.lineCount = 0;

    // Per-tick throttle state so pending-asks frame + resolved decision frame share the same 6-line budget.
    this._tickCountByTick = new Map();
    this._shownAskIds = new Set();
    this._shownAskOrder = [];
    this._ringCursor = 0;

    this._injectStyles();
    this._buildDom();
  }

  _injectStyles() {
    const style = document.createElement("style");
    style.textContent = `
      .crowd-chat {
        position: fixed;
        top: 0;
        right: 0;
        bottom: 0;
        width: ${CHAT_PANEL_WIDTH}px;
        box-sizing: border-box;
        background: rgba(246, 235, 230, 0.78);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        border-left: 1px solid rgba(55, 45, 48, 0.16);
        color: #2b2327;
        font: 11px/1.42 "IBM Plex Mono", monospace;
        display: none;
        flex-direction: column;
        z-index: 15;
        user-select: text;
      }
      .crowd-chat.visible {
        display: flex;
      }
      .crowd-chat-header {
        flex: 0 0 auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 9px 12px 7px 12px;
        border-bottom: 1px solid rgba(55, 45, 48, 0.12);
        font-size: 10px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: rgba(55, 45, 48, 0.62);
      }
      .crowd-chat-header .cond-tag {
        font-weight: 600;
        color: rgba(45, 34, 40, 0.78);
      }
      .crowd-chat-log {
        flex: 1 1 auto;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 8px 10px 12px 10px;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        gap: 4px;
        scrollbar-width: thin;
        scrollbar-color: rgba(55, 45, 48, 0.22) transparent;
      }
      .crowd-chat-inner {
        display: flex;
        flex-direction: column;
        gap: 4px;
        margin-top: auto;
      }
      .chat-line {
        padding: 2px 5px;
        border-radius: 2px;
        word-wrap: break-word;
        overflow-wrap: anywhere;
        color: rgba(40, 32, 36, 0.88);
        transition: background 0.12s ease;
      }
      .chat-line.interactive {
        cursor: pointer;
      }
      .chat-line.interactive:hover {
        background: rgba(55, 45, 48, 0.08);
      }
      .chat-line .agent-id {
        font-weight: 600;
      }
      .chat-line .arrow {
        color: rgba(55, 45, 48, 0.48);
        margin: 0 2px;
      }
      .chat-line.kind-ask {
        color: rgba(32, 24, 29, 0.94);
      }
      .chat-line.kind-answer .ans-yes {
        color: #2c6d47;
        font-weight: 500;
      }
      .chat-line.kind-answer .ans-no {
        color: #a63a31;
        font-weight: 500;
      }
      .chat-line.kind-notable {
        color: rgba(48, 39, 44, 0.82);
      }
      .chat-line.kind-ring {
        color: rgba(58, 48, 53, 0.54);
        font-size: 10.5px;
      }
      .chat-line.kind-house {
        margin-top: 2px;
        padding: 3px 6px;
        border-left: 2px solid rgba(92, 54, 86, 0.48);
        background: rgba(92, 54, 86, 0.055);
        color: rgba(44, 30, 42, 0.88);
        font-variant-caps: all-small-caps;
        letter-spacing: 0.04em;
        font-size: 11.5px;
        font-weight: 500;
      }
      .chat-line.kind-house .booth-label {
        font-weight: 600;
        letter-spacing: 0.08em;
      }
      .crowd-chat-empty {
        color: rgba(55, 45, 48, 0.44);
        font-size: 10.5px;
        padding: 6px 4px;
      }
    `;
    document.head.appendChild(style);
  }

  _buildDom() {
    this.panelEl = document.createElement("aside");
    this.panelEl.className = "crowd-chat";
    this.panelEl.setAttribute("aria-label", "Crowd chat feed");

    this.headerEl = document.createElement("div");
    this.headerEl.className = "crowd-chat-header";
    this.headerEl.innerHTML = `<span>FLOOR FEED</span><span class="cond-tag">SHOVE</span>`;
    this.condTagEl = this.headerEl.querySelector(".cond-tag");

    this.logEl = document.createElement("div");
    this.logEl.className = "crowd-chat-log";

    this.innerEl = document.createElement("div");
    this.innerEl.className = "crowd-chat-inner";

    this.emptyEl = document.createElement("div");
    this.emptyEl.className = "crowd-chat-empty";
    this.emptyEl.textContent = "Waiting for crowd ticks… (C to hide)";
    this.innerEl.appendChild(this.emptyEl);

    this.logEl.appendChild(this.innerEl);
    this.panelEl.appendChild(this.headerEl);
    this.panelEl.appendChild(this.logEl);
    document.body.appendChild(this.panelEl);
  }

  setOpen(open) {
    const next = Boolean(open);
    if (this.isOpen === next) return;
    this.isOpen = next;
    this.panelEl.classList.toggle("visible", this.isOpen);
    if (!this.isOpen) {
      this.onHoverAgents([]);
    } else {
      this._scrollToBottom(true);
    }
    this.onToggle(this.isOpen);
  }

  toggle() {
    this.setOpen(!this.isOpen);
  }

  setCondition(cond) {
    if (this.condTagEl && cond) {
      this.condTagEl.textContent = String(cond).toUpperCase();
    }
  }

  clear() {
    this.innerEl.innerHTML = "";
    this.innerEl.appendChild(this.emptyEl);
    this.lineCount = 0;
    this._tickCountByTick.clear();
    this._shownAskIds.clear();
    this._shownAskOrder = [];
    this.onHoverAgents([]);
  }

  _rememberShownAsk(askId) {
    if (!askId) return;
    this._shownAskIds.add(askId);
    this._shownAskOrder.push(askId);
    if (this._shownAskOrder.length > 200) {
      const old = this._shownAskOrder.shift();
      this._shownAskIds.delete(old);
    }
  }

  _getTickUsed(tick) {
    return this._tickCountByTick.get(tick) || 0;
  }

  _addTickUsed(tick, n = 1) {
    const next = this._getTickUsed(tick) + n;
    this._tickCountByTick.set(tick, next);
    if (this._tickCountByTick.size > 40) {
      const oldestKey = this._tickCountByTick.keys().next().value;
      this._tickCountByTick.delete(oldestKey);
    }
  }

  _scrollToBottom(force = false) {
    if (!this.isOpen) return;
    const distFromBottom =
      this.logEl.scrollHeight - this.logEl.scrollTop - this.logEl.clientHeight;
    if (force || distFromBottom < 48) {
      this.logEl.scrollTop = this.logEl.scrollHeight;
    }
  }

  _appendLine(el, hoverAgentIds = []) {
    if (this.emptyEl.parentNode === this.innerEl) {
      this.innerEl.removeChild(this.emptyEl);
    }
    if (hoverAgentIds.length > 0) {
      el.classList.add("interactive");
      el.dataset.agents = hoverAgentIds.join(",");
      el.addEventListener("mouseenter", () => {
        this.onHoverAgents(hoverAgentIds);
      });
      el.addEventListener("mouseleave", () => {
        this.onHoverAgents([]);
      });
    }
    this.innerEl.appendChild(el);
    this.lineCount += 1;

    while (this.innerEl.children.length > MAX_HISTORY_LINES) {
      this.innerEl.removeChild(this.innerEl.firstChild);
    }
    this._scrollToBottom(false);
  }

  _makeAgentSpan(agentId, colorFn) {
    const span = document.createElement("span");
    span.className = "agent-id";
    span.textContent = `A${agentId}`;
    span.style.color = colorFn(agentId);
    return span;
  }

  _emitAsk(ask, colorFn) {
    const line = document.createElement("div");
    line.className = "chat-line kind-ask";

    line.appendChild(this._makeAgentSpan(ask.from, colorFn));
    const arrow = document.createElement("span");
    arrow.className = "arrow";
    arrow.textContent = "→";
    line.appendChild(arrow);
    line.appendChild(this._makeAgentSpan(ask.to, colorFn));

    const body = document.createElement("span");
    body.textContent = `: ${ask.text || ""}`;
    line.appendChild(body);

    this._appendLine(line, [ask.from, ask.to]);
    this._rememberShownAsk(ask.id);
  }

  _emitAnswer(ans, colorFn) {
    const fromId = ans.to; // blocker answers
    const toId = ans.from; // to asker
    const line = document.createElement("div");
    line.className = "chat-line kind-answer";

    line.appendChild(this._makeAgentSpan(fromId, colorFn));
    const arrow = document.createElement("span");
    arrow.className = "arrow";
    arrow.textContent = "→";
    line.appendChild(arrow);
    line.appendChild(this._makeAgentSpan(toId, colorFn));

    const sep = document.createElement("span");
    sep.textContent = ": ";
    line.appendChild(sep);

    const verdict = document.createElement("span");
    if (ans.answer === "yes") {
      verdict.className = "ans-yes";
      verdict.textContent = "✓ yields";
    } else {
      verdict.className = "ans-no";
      verdict.textContent = "✗ holds";
    }
    line.appendChild(verdict);

    this._appendLine(line, [fromId, toId]);
  }

  _emitNotable(ev, colorFn) {
    const line = document.createElement("div");
    line.className = "chat-line kind-notable";

    if (ev.kind === "exhausted_summary") {
      const ids = Array.isArray(ev.agents) ? ev.agents : [];
      const body = document.createElement("span");
      body.textContent = String(ev.text || `${ids.length} agents retreating`);
      body.style.color = "rgba(92, 66, 82, 0.85)";
      line.appendChild(body);
      this._appendLine(line, ids);
      return;
    }

    const id = ev.agent;
    const prefix = `A${id}`;
    const rawText = String(ev.text || "");
    const rest = rawText.startsWith(prefix) ? rawText.slice(prefix.length) : ` ${rawText}`;

    line.appendChild(this._makeAgentSpan(id, colorFn));
    const body = document.createElement("span");
    body.textContent = rest;
    line.appendChild(body);

    this._appendLine(line, Number.isFinite(id) ? [id] : []);
  }

  _emitHouse(house) {
    const line = document.createElement("div");
    line.className = "chat-line kind-house";

    const lbl = document.createElement("span");
    lbl.className = "booth-label";
    lbl.textContent = "BOOTH: ";
    line.appendChild(lbl);

    const p = Number.isFinite(house.p) ? house.p.toFixed(2) : "0.00";
    const body = document.createElement("span");
    body.textContent = `${house.section || "groove"} (${p})`;
    line.appendChild(body);

    this._appendLine(line, []);
  }

  _emitRing(ringId, ringData) {
    const line = document.createElement("div");
    line.className = "chat-line kind-ring";

    const section = ringData.section || "groove";
    const eta = formatDropEta(ringData.drop_eta);
    const pVal = Number.isFinite(ringData.p) ? ringData.p : ringData.section_p || 0;
    line.textContent = `ring ${ringId} hears: ${section} · drop ${eta} (${pVal.toFixed(2)})`;

    this._appendLine(line, []);
  }

  /**
   * Ingest a Section-9 decision frame (either a pre-ring pending-asks announcement
   * or a full resolved tick frame) and emit up to ~6 priority-ordered lines per tick.
   */
  ingestDecision(msg, getAgentColor) {
    if (!msg) return;
    if (msg.condition) this.setCondition(msg.condition);

    const colorFn = (id) => {
      try {
        return getAgentColor ? getAgentColor(id) : "#4b3847";
      } catch {
        return "#4b3847";
      }
    };

    const tick = Number.isFinite(msg.tick) ? msg.tick : 0;
    const isPendingAnnouncement =
      Array.isArray(msg.asks) &&
      msg.asks.length > 0 &&
      !msg.house &&
      (!msg.answers || msg.answers.length === 0);

    let remaining = Math.max(0, MAX_LINES_PER_TICK - this._getTickUsed(tick));
    if (remaining <= 0) return;

    // Case 1: pre-ring pending-asks broadcast (reserve space for paired answers + BOOTH/ring)
    if (isPendingAnnouncement) {
      const cap = Math.min(remaining, MAX_PENDING_ASKS_PER_TICK);
      for (let i = 0; i < msg.asks.length && i < cap; i++) {
        this._emitAsk(msg.asks[i], colorFn);
        this._addTickUsed(tick, 1);
      }
      return;
    }

    // Case 2: full resolved tick decision frame (or ?fake=1 combined frame)
    // Priority 1: answers matching shown asks (plus any extra "yes" yields!), never orphan holds spam.
    const answers = Array.isArray(msg.answers) ? msg.answers : [];
    const pairedAnswers = answers.filter((a) => this._shownAskIds.has(a.id));
    const extraYesAnswers = answers.filter(
      (a) => !this._shownAskIds.has(a.id) && a.answer === "yes"
    );
    const chosenAnswers =
      pairedAnswers.length > 0
        ? [...pairedAnswers, ...extraYesAnswers.slice(0, 1)]
        : answers.slice(0, 2);

    // Reserve 1 slot for BOOTH house ear so BOOTH appears once per tick unless asks/answers alone fill the tick
    const maxAnsNow = Math.max(0, Math.min(remaining, MAX_PENDING_ASKS_PER_TICK + 1));
    let emittedAns = 0;
    for (const ans of chosenAnswers) {
      if (remaining <= 0 || emittedAns >= maxAnsNow) break;
      if (ans.from === undefined || ans.to === undefined) continue;
      this._emitAnswer(ans, colorFn);
      this._addTickUsed(tick, 1);
      remaining -= 1;
      emittedAns += 1;
    }

    const asks = Array.isArray(msg.asks) ? msg.asks : [];
    let emittedCombinedAsks = 0;
    for (const ask of asks) {
      if (remaining <= 0 || emittedCombinedAsks >= MAX_PENDING_ASKS_PER_TICK) break;
      if (this._shownAskIds.has(ask.id)) continue;
      this._emitAsk(ask, colorFn);
      this._addTickUsed(tick, 1);
      remaining -= 1;
      emittedCombinedAsks += 1;
    }

    // Priority 2: exhaustion (collapsed into one summary if multiple) and give-up notable decisions
    const notable = Array.isArray(msg.notable) ? msg.notable : [];
    const exhRaw = notable.filter((ev) => ev.kind === "exhausted" || ev.kind === "exhausted_summary");
    const highNotable = [];
    if (exhRaw.length === 1) {
      highNotable.push(exhRaw[0]);
    } else if (exhRaw.length > 1) {
      const mergedIds = [];
      for (const ev of exhRaw) {
        if (Array.isArray(ev.agents)) mergedIds.push(...ev.agents);
        else if (Number.isFinite(ev.agent)) mergedIds.push(ev.agent);
      }
      highNotable.push({
        kind: "exhausted_summary",
        agents: mergedIds,
        text: `${mergedIds.length} agents retreating`,
      });
    }
    for (const ev of notable) {
      if (ev.kind === "give_up") highNotable.push(ev);
    }
    const lowNotable = notable.filter(
      (ev) => ev.kind !== "exhausted" && ev.kind !== "exhausted_summary" && ev.kind !== "give_up"
    );

    for (const ev of highNotable) {
      if (remaining <= 0) break;
      this._emitNotable(ev, colorFn);
      this._addTickUsed(tick, 1);
      remaining -= 1;
    }

    // Priority 3: house ear (once per tick) & one ring ear (once per tick, rotating across rings)
    if (remaining > 0 && msg.house && msg.house.section) {
      this._emitHouse(msg.house);
      this._addTickUsed(tick, 1);
      remaining -= 1;
    }

    if (remaining > 0 && msg.rings && typeof msg.rings === "object") {
      const ringKeys = Object.keys(msg.rings).sort((a, b) => Number(a) - Number(b));
      if (ringKeys.length > 0) {
        const chosenKey = ringKeys[this._ringCursor % ringKeys.length];
        this._ringCursor += 1;
        this._emitRing(chosenKey, msg.rings[chosenKey]);
        this._addTickUsed(tick, 1);
        remaining -= 1;
      }
    }

    // Priority 4: push_hard notable events fill any leftover budget (still <= 6 lines/tick)
    for (const ev of lowNotable) {
      if (remaining <= 0) break;
      this._emitNotable(ev, colorFn);
      this._addTickUsed(tick, 1);
      remaining -= 1;
    }
  }
}
