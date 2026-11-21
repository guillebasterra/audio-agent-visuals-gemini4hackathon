// Section 9 WebSocket client: JSON control/clock/frames + binary Int16 PCM chunks.
// Automatically reconnects without blocking the 60 fps render loop.

export class CrowdSocket {
  constructor({ url, onInit, onPhysics, onDecision, onStatus } = {}) {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    this.url = url || `${proto}//${window.location.host}/ws`;
    this.onInit = onInit || (() => {});
    this.onPhysics = onPhysics || (() => {});
    this.onDecision = onDecision || (() => {});
    this.onStatus = onStatus || (() => {});

    this.ws = null;
    this.connected = false;
    this.closedManually = false;
    this.reconnectDelayMs = 500;
    this._reconnectTimer = null;

    this.physicsFramesReceived = 0;
    this.clockSent = 0;
    this.pcmChunksSent = 0;
    this.pcmBytesSent = 0;
  }

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    this.closedManually = false;
    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    ws.onopen = () => {
      this.connected = true;
      this.reconnectDelayMs = 500;
      this.onStatus(true);
    };

    ws.onmessage = (evt) => {
      if (typeof evt.data !== "string") return;
      let msg;
      try {
        msg = JSON.parse(evt.data);
      } catch {
        return;
      }
      if (msg.type === "init") {
        this.onInit(msg);
      } else if (msg.type === "physics") {
        this.physicsFramesReceived += 1;
        this.onPhysics(msg);
      } else if (msg.type === "decision") {
        this.onDecision(msg);
      }
    };

    ws.onclose = () => {
      this.connected = false;
      this.onStatus(false);
      if (!this.closedManually) {
        this._scheduleReconnect();
      }
    };

    ws.onerror = () => {
      // onclose handles reconnect cleanly
    };
  }

  _scheduleReconnect() {
    if (this._reconnectTimer) return;
    this._reconnectTimer = window.setTimeout(() => {
      this._reconnectTimer = null;
      this.reconnectDelayMs = Math.min(3000, Math.round(this.reconnectDelayMs * 1.5));
      this.connect();
    }, this.reconnectDelayMs);
  }

  sendJson(obj) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(obj));
    if (obj && obj.type === "clock") {
      this.clockSent += 1;
    }
    return true;
  }

  sendPcm(arrayBuffer) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.ws.send(arrayBuffer);
    this.pcmChunksSent += 1;
    this.pcmBytesSent += arrayBuffer.byteLength;
    return true;
  }

  close() {
    this.closedManually = true;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
