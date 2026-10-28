// Where the lines come from: the band over Wi-Fi (WebSocket) or USB (Web Serial), the built-in
// demo band, or a recorded CSV dropped on the page. Everything ends in onLine(line, arrivalMs).

import { DemoGenerator, JitterSim } from './core/demo.js';
import { LineSplitter } from './core/parse.js';

const PICO_VID = 0x2e8a; // Raspberry Pi
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export class Transport {
  constructor({ onLine, onStatus, onRestart }) {
    this.onLine = onLine;
    this.onStatus = onStatus;
    this.onRestart = onRestart;
    this.kind = null; // 'wifi' | 'usb' | 'demo' | 'replay' | null
    this.token = 0; // bumps on every stop: old loops notice and quit
    this.demo = null;
    this.jitterOn = false;
    this.lastLineAt = 0;
  }

  status(state, detail) {
    this.state = state;
    this.onStatus({ kind: this.kind, state, detail });
  }

  line(line, now) {
    this.lastLineAt = now;
    this.onLine(line, now);
  }

  async stop(silent = false) {
    this.token++;
    clearInterval(this.timer);
    clearTimeout(this.timer);
    this.timer = null;
    if (this.ws) {
      const ws = this.ws;
      this.ws = null;
      ws.onopen = ws.onmessage = ws.onclose = ws.onerror = null;
      try {
        ws.close();
      } catch {
        /* already closed */
      }
    }
    if (this.reader) {
      try {
        await this.reader.cancel();
      } catch {
        /* port gone */
      }
    }
    this.demo = null;
    this.kind = null;
    if (!silent) this.status('idle', 'Not connected');
  }

  async begin(kind) {
    await this.stop(true);
    this.kind = kind;
    this.onRestart();
    return this.token;
  }

  send(cmd) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(cmd);
    if (this.writer) this.writer.write(new TextEncoder().encode(`${cmd}\n`)).catch(() => {});
  }

  // ---------- Wi-Fi ----------
  async connectWifi(host) {
    host = host.trim().replace(/^wss?:\/\//, '').replace(/\/.*$/, '');
    const token = await this.begin('wifi');
    let backoff = 500;
    const open = () => {
      if (token !== this.token) return;
      this.status('connecting', `Connecting to ${host}…`);
      let ws;
      try {
        ws = new WebSocket(`ws://${host}/ws`);
      } catch {
        this.status('error', `Bad address: ${host}`);
        return;
      }
      this.ws = ws;
      const split = new LineSplitter();
      let last = performance.now();
      let retried = false;
      const retry = (why) => {
        if (retried || token !== this.token) return;
        retried = true;
        clearInterval(watchdog);
        ws.onopen = ws.onmessage = ws.onclose = ws.onerror = null;
        try {
          ws.close();
        } catch {
          /* ignore */
        }
        this.status('reconnecting', `${why}, retrying…`);
        this.timer = setTimeout(open, backoff);
        backoff = Math.min(2000, backoff * 2);
      };
      ws.onopen = () => {
        backoff = 500;
        last = performance.now();
        this.status('live', `Wi-Fi · ${host}`);
      };
      ws.onmessage = (e) => {
        const now = performance.now();
        last = now;
        for (const l of split.push(`${e.data}\n`)) this.line(l, now);
      };
      ws.onclose = () => retry(`Lost ${host}`);
      ws.onerror = () => retry(`Can't reach ${host}`);
      // A Wi-Fi drop doesn't close the socket quickly: treat 3 s of silence as a drop.
      const watchdog = setInterval(() => {
        if (token !== this.token) return clearInterval(watchdog);
        if (ws.readyState === WebSocket.OPEN && performance.now() - last > 3000) retry('No data');
      }, 500);
    };
    open();
  }

  // ---------- USB ----------
  async connectUsb() {
    if (!('serial' in navigator)) {
      this.status('error', 'USB needs Chrome or Edge');
      return;
    }
    let port;
    try {
      port = await navigator.serial.requestPort({ filters: [{ usbVendorId: PICO_VID }] });
    } catch {
      return; // chooser closed
    }
    const token = await this.begin('usb');
    while (token === this.token) {
      const why = await this.readPort(port, token);
      if (token !== this.token) return;
      if (why === 'busy') {
        // Windows lets one program use a COM port at a time. Keep retrying: connects once it's free.
        this.status('error', 'USB port busy: close the PlatformIO Serial Monitor (or any app using the band)…');
      } else {
        this.status('reconnecting', 'USB unplugged? Waiting for the band…');
      }
      // Re-open the same (already allowed) port when it comes back.
      for (;;) {
        await sleep(1000);
        if (token !== this.token) return;
        const ports = await navigator.serial.getPorts();
        const again = ports.find((p) => p.getInfo().usbVendorId === PICO_VID);
        if (again) {
          port = again;
          break;
        }
      }
    }
  }

  // Streams until the port closes. Returns 'busy' if it couldn't be opened, else 'lost'.
  async readPort(port, token) {
    try {
      await port.open({ baudRate: 115200 });
    } catch {
      return 'busy';
    }
    try {
      this.status('connecting', 'Opening USB…');
      try {
        await port.setSignals({ dataTerminalReady: true, requestToSend: true });
      } catch {
        /* not all drivers support it */
      }
      this.writer = port.writable.getWriter();
      this.reader = port.readable.getReader();
      this.status('live', 'USB');
      this.send('r'); // raw 50 Hz stream
      const kick = setInterval(() => {
        // the band rebooted into human mode: ask again
        if (token === this.token && performance.now() - this.lastLineAt > 2000) this.send('r');
      }, 2000);
      const dec = new TextDecoder();
      const split = new LineSplitter();
      try {
        for (;;) {
          const { value, done } = await this.reader.read();
          if (done || token !== this.token) break;
          const now = performance.now();
          for (const l of split.push(dec.decode(value, { stream: true }))) this.line(l, now);
        }
      } finally {
        clearInterval(kick);
      }
    } catch (e) {
      if (token === this.token) this.status('error', `USB: ${e.message}`);
    }
    try {
      this.reader?.releaseLock();
      this.writer?.releaseLock();
    } catch {
      /* ignore */
    }
    this.reader = this.writer = null;
    try {
      await port.close();
    } catch {
      /* already closed */
    }
    return 'lost';
  }

  // ---------- Demo ----------
  async startDemo() {
    const token = await this.begin('demo');
    this.demo = new DemoGenerator(Date.now() & 0xffff);
    const jitter = new JitterSim();
    const t0 = performance.now();
    let gen = 0; // ms of demo generated
    const queue = [];
    this.status('live', 'Demo band');
    this.timer = setInterval(() => {
      if (token !== this.token) return;
      const now = performance.now();
      if (now - t0 - gen > 2000) gen = now - t0 - 20; // tab was asleep: skip ahead
      while (gen <= now - t0) {
        const line = this.demo.next();
        queue.push([this.jitterOn ? jitter.release(t0 + gen) : t0 + gen, line]);
        gen += 20;
      }
      while (queue.length && queue[0][0] <= now) this.line(queue.shift()[1], now);
    }, 5);
  }

  toggleJitter() {
    this.jitterOn = !this.jitterOn;
    return this.jitterOn;
  }

  // ---------- Replay a recording ----------
  async replay(text, name) {
    const items = [];
    for (const l of text.split(/\r?\n/)) {
      if (!l || l[0] === '#') continue;
      const t = Number(l.split(',')[0]);
      if (Number.isFinite(t)) items.push([t, l]);
    }
    if (!items.length) {
      this.status('error', `${name}: no samples found`);
      return;
    }
    const token = await this.begin('replay');
    const t0 = performance.now();
    const d0 = items[0][0];
    let i = 0;
    this.status('live', `Replay · ${name}`);
    this.timer = setInterval(() => {
      if (token !== this.token) return;
      const now = performance.now();
      while (i < items.length && items[i][0] - d0 <= now - t0) this.line(items[i++][1], now);
      if (i >= items.length) {
        clearInterval(this.timer);
        this.kind = null;
        this.status('idle', `Replay finished · ${name}`);
      }
    }, 5);
  }
}
