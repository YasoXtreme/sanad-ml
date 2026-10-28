// Jitter buffer: plays the band's samples back on the band's own clock, a little behind live.
//
// Wi-Fi delivers samples in uneven bursts. Instead of drawing each sample when it lands, we map band
// time (t_ms) onto laptop time and play back `delay` ms behind:
//   offset   = arrival - t_ms, its minimum over the last few seconds = the fastest path through the link
//   playhead = now - min offset - delay   (in band time)
// The playhead moves at ±10 % speed to absorb changes in the offset or delay, so it never jumps
// (except after a long outage, when it snaps). Each frame returns:
//   process   samples to run through the filters, in order, exactly once. Processing runs one sample
//             ahead of the playhead so the frame can always interpolate between two processed samples.
//   a, b, frac  draw a + (b - a) * frac
//   starvedMs   > 0 when the playhead ran past the newest sample (gap): extrapolate briefly, then hold

class WindowExtreme {
  // Sliding-window minimum (or maximum) over time, O(1) amortized.
  constructor(isMin) {
    this.isMin = isMin;
    this.items = []; // [time, value], monotonic values
  }

  push(time, value) {
    const it = this.items;
    while (it.length && (this.isMin ? it[it.length - 1][1] >= value : it[it.length - 1][1] <= value)) {
      it.pop();
    }
    it.push([time, value]);
  }

  trim(before) {
    const it = this.items;
    while (it.length > 1 && it[0][0] < before) it.shift();
  }

  get value() {
    return this.items.length ? this.items[0][1] : null;
  }
}

const DEFAULTS = {
  periodMs: 20,
  baseDelayMs: 120,
  maxDelayMs: 400,
  offsetWindowMs: 4000,
  jitterWindowMs: 5000,
  delayMarginMs: 30,
  delayShrinkPerS: 20, // ms of delay given back per second once the link is calm
  maxRateChange: 0.1, // playhead runs at 0.9x..1.1x speed while catching up
  snapMs: 600,
  keepMs: 3000,
};

export class Playback {
  constructor(cfg = {}) {
    this.cfg = { ...DEFAULTS, ...cfg };
    this.reset();
  }

  reset() {
    this.buf = [];
    this.next = 0; // buf[next] is the next sample to process
    this.offsets = new WindowExtreme(true);
    this.jitter = new WindowExtreme(false);
    this.playhead = null;
    this.lastNow = null;
    this.delay = this.cfg.baseDelayMs;
    this.lastT = null;
    this.lastSeq = null;
    this.drops = 0;
    this.received = 0;
  }

  // Add a sample that arrived at laptop time `now` (ms). Returns false normally, true if the stream
  // restarted (band rebooted, new recording: reset your filters too), null if the sample was ignored.
  push(s, now) {
    let restarted = false;
    if (this.lastT !== null) {
      const seqAhead = s.seq !== null && this.lastSeq !== null && s.seq > this.lastSeq;
      if (s.t < this.lastT - 1000) {
        this.reset();
        restarted = true;
      } else if (s.t <= this.lastT && !seqAhead) {
        return null; // duplicate or out of order: ignore
      } else if (seqAhead) {
        // v0.2 stamps FIFO samples to ~5 ms, so two can share a millisecond: keep time moving forward
        if (s.t <= this.lastT) s.t = this.lastT + 1;
        if (s.seq > this.lastSeq + 1) this.drops += s.seq - this.lastSeq - 1;
      } else {
        const missing = Math.round((s.t - this.lastT) / this.cfg.periodMs) - 1;
        if (missing > 0) this.drops += missing;
      }
    }
    this.lastT = s.t;
    this.lastSeq = s.seq;

    const off = now - s.t;
    this.offsets.push(now, off);
    this.offsets.trim(now - this.cfg.offsetWindowMs);
    this.jitter.push(now, off - this.offsets.value);
    this.jitter.trim(now - this.cfg.jitterWindowMs);

    this.buf.push(s);
    this.received++;
    // Nobody is drawing (hidden tab): don't let unplayed samples pile up forever.
    if (this.buf.length > 3000) {
      this.buf.splice(0, 1000);
      this.next = Math.max(0, this.next - 1000);
    }
    return restarted;
  }

  // Milliseconds of data buffered ahead of the playhead.
  get bufferedMs() {
    if (!this.buf.length || this.playhead === null) return 0;
    return Math.max(0, this.buf[this.buf.length - 1].t - this.playhead);
  }

  advance(now) {
    const c = this.cfg;
    const frameMs = this.lastNow === null ? 0 : Math.max(0, now - this.lastNow);
    this.lastNow = now;
    if (!this.buf.length) {
      return { process: [], a: null, b: null, frac: 0, starvedMs: 0, playhead: null };
    }

    // Delay: grows at once to cover the worst recent jitter, shrinks slowly when the link calms.
    // + one period: processing runs one sample ahead of the playhead.
    const need = this.jitter.value + c.periodMs + c.delayMarginMs;
    const want = Math.min(c.maxDelayMs, Math.max(c.baseDelayMs, need));
    if (want > this.delay) this.delay = want;
    else this.delay = Math.max(want, this.delay - (c.delayShrinkPerS * frameMs) / 1000);

    const target = now - this.offsets.value - this.delay;
    if (this.playhead === null || Math.abs(target - this.playhead) > c.snapMs) {
      this.playhead = target;
    } else {
      const err = target - this.playhead;
      const rate = 1 + Math.max(-c.maxRateChange, Math.min(c.maxRateChange, err / 200));
      this.playhead += frameMs * rate;
    }
    const P = this.playhead;

    const process = [];
    while (this.next < this.buf.length && (this.next === 0 || this.buf[this.next - 1].t <= P)) {
      process.push(this.buf[this.next]);
      this.next++;
    }

    let ia = this.next - 1;
    while (ia > 0 && this.buf[ia].t > P) ia--;
    const a = this.buf[ia];
    const b = ia + 1 < this.next ? this.buf[ia + 1] : null;
    let frac = 0;
    let starvedMs = 0;
    if (a.t <= P) {
      if (b) frac = (P - a.t) / (b.t - a.t);
      else starvedMs = P - a.t;
    }

    // Forget samples well behind the playhead.
    let cut = 0;
    while (cut < ia - 1 && this.buf[cut].t < P - c.keepMs) cut++;
    if (cut > 64) {
      this.buf.splice(0, cut);
      this.next -= cut;
    }

    return { process, a, b, frac, starvedMs, playhead: P };
  }
}
