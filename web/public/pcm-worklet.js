// AudioWorkletProcessor: downsamples live/file input to 16 kHz mono Int16
// and posts 250 ms chunks (4000 samples = 8000 bytes) ONLY while playing.

const TARGET_SR = 16000;
const CHUNK_SAMPLES = 4000; // 250 ms at 16 kHz

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.playing = false;
    this.ratio = sampleRate / TARGET_SR;
    this.phase = 0;
    this.out = new Int16Array(CHUNK_SAMPLES);
    this.outIdx = 0;
    this.prevSample = 0;

    this.port.onmessage = (e) => {
      const msg = e.data || {};
      if (typeof msg.playing === "boolean") {
        this.playing = msg.playing;
        if (!this.playing) {
          this.outIdx = 0;
          this.phase = 0;
        }
      }
      if (msg.reset) {
        this.outIdx = 0;
        this.phase = 0;
        this.prevSample = 0;
      }
    };
  }

  process(inputs) {
    if (!this.playing) return true;
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    const ch0 = input[0];
    const ch1 = input.length > 1 ? input[1] : null;
    const len = ch0.length;
    if (len === 0) return true;

    // Mix down to mono and linearly resample to 16 kHz.
    let pos = this.phase;
    while (pos < len) {
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const s0 =
        idx === 0
          ? this.prevSample
          : ch1
            ? 0.5 * (ch0[idx - 1] + ch1[idx - 1])
            : ch0[idx - 1];
      const s1 = ch1 ? 0.5 * (ch0[idx] + ch1[idx]) : ch0[idx];
      const val = s0 + (s1 - s0) * frac;
      const clamped = val < -1 ? -1 : val > 1 ? 1 : val;
      this.out[this.outIdx++] = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);

      if (this.outIdx >= CHUNK_SAMPLES) {
        const buffer = this.out.buffer;
        this.port.postMessage(buffer, [buffer]);
        this.out = new Int16Array(CHUNK_SAMPLES);
        this.outIdx = 0;
      }
      pos += this.ratio;
    }

    this.phase = pos - len;
    this.prevSample = ch1 ? 0.5 * (ch0[len - 1] + ch1[len - 1]) : ch0[len - 1];
    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
