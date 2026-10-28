// Heartbeats from the MAX30102 IR signal (for the light rings), only while a finger is on it.
// Blood arriving with each beat absorbs more IR, so the IR reading dips: remove the slow baseline,
// flip the sign, smooth a little and take peaks above a fraction of the recent pulse size, at least
// a refractory time apart (max 180 bpm).

const DEFAULTS = {
  fingerOnIr: 50000,
  fingerOffIr: 40000,
  settleMs: 1000, // ignore the first second after the finger lands (baseline jump)
  baselineTauS: 1.5,
  smoothTauS: 0.06,
  envTauS: 2.0,
  peakFrac: 0.4,
  refractoryMs: 330,
  maxIbiMs: 1500,
  lostMs: 3000, // no beat for this long: bpm unknown
};

export class HeartDetector {
  constructor(cfg = {}) {
    this.cfg = { ...DEFAULTS, ...cfg };
    this.reset();
  }

  reset() {
    this.finger = false;
    this.fingerSince = null;
    this.bpm = null;
    this.resetSignal();
  }

  resetSignal() {
    this.lastT = null;
    this.base = null;
    this.sm = 0;
    this.env = 0;
    this.p1 = null; // previous smoothed value
    this.p2 = null; // the one before
    this.t1 = null;
    this.lastBeat = null;
    this.ibis = [];
    this.bpm = null;
  }

  // Returns {type:'beat', t} or null.
  push(s) {
    const c = this.cfg;
    const wasOn = this.finger;
    this.finger = wasOn ? s.ir > c.fingerOffIr : s.ir > c.fingerOnIr;
    if (!this.finger) {
      if (wasOn) this.resetSignal();
      this.fingerSince = null;
      return null;
    }
    if (!wasOn) this.fingerSince = s.t;

    const dt = this.lastT === null ? 0 : (s.t - this.lastT) / 1000;
    this.lastT = s.t;
    if (this.base === null || dt > 0.5) {
      this.base = s.ir;
      this.p1 = this.p2 = null;
      return null;
    }
    this.base += (s.ir - this.base) * Math.min(1, dt / c.baselineTauS);
    const x = -(s.ir - this.base);
    this.sm += (x - this.sm) * Math.min(1, dt / c.smoothTauS);
    this.env = Math.max(Math.abs(this.sm), this.env * Math.exp(-dt / c.envTauS));

    let beat = null;
    const settled = s.t - this.fingerSince >= c.settleMs;
    if (
      settled &&
      this.p1 !== null &&
      this.p2 !== null &&
      this.p1 > this.p2 &&
      this.p1 >= this.sm &&
      this.p1 > 0 &&
      this.p1 > c.peakFrac * this.env &&
      (this.lastBeat === null || this.t1 - this.lastBeat >= c.refractoryMs)
    ) {
      if (this.lastBeat !== null) {
        const ibi = this.t1 - this.lastBeat;
        if (ibi <= c.maxIbiMs) {
          this.ibis.push(ibi);
          if (this.ibis.length > 5) this.ibis.shift();
        } else {
          this.ibis = [];
        }
      }
      this.lastBeat = this.t1;
      if (this.ibis.length >= 3) {
        const sorted = [...this.ibis].sort((p, q) => p - q);
        this.bpm = 60000 / sorted[Math.floor(sorted.length / 2)];
      }
      beat = { type: 'beat', t: this.t1 };
    }
    if (this.lastBeat !== null && s.t - this.lastBeat > c.lostMs) {
      this.bpm = null;
      this.ibis = [];
    }
    this.p2 = this.p1;
    this.p1 = this.sm;
    this.t1 = s.t;
    return beat;
  }
}
