// The small overlay: connect buttons, status, stats, bpm. Fades out when the mouse rests.

const $ = (id) => document.getElementById(id);
const HOST_KEY = 'sanad.host';

function load(key, fallback) {
  try {
    return localStorage.getItem(key) || fallback;
  } catch {
    return fallback;
  }
}

function save(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* private window */
  }
}

export class Ui {
  constructor(h) {
    this.hud = $('hud');
    this.hostInput = $('host');
    this.hostInput.value = load(HOST_KEY, '192.168.42.1');
    const wifi = () => {
      save(HOST_KEY, this.hostInput.value.trim());
      h.onWifi(this.hostInput.value);
    };
    $('btn-wifi').onclick = wifi;
    this.hostInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') wifi();
    });
    $('btn-usb').onclick = h.onUsb;
    $('btn-demo').onclick = h.onDemo;
    $('btn-stop').onclick = h.onStop;
    this.buttons = { wifi: $('btn-wifi'), usb: $('btn-usb'), demo: $('btn-demo') };

    // Fade the overlay out after 3 s without mouse movement (and hide the cursor).
    this.hidden = false;
    const wake = () => {
      document.body.classList.remove('idle');
      clearTimeout(this.idleTimer);
      this.idleTimer = setTimeout(() => {
        if (document.activeElement !== this.hostInput) document.body.classList.add('idle');
      }, 3000);
    };
    window.addEventListener('pointermove', wake);
    window.addEventListener('pointerdown', wake); // a tap with no movement (touch screens) wakes it too
    window.addEventListener('keydown', wake);
    wake();
  }

  toggleHidden() {
    this.hidden = !this.hidden;
    document.body.classList.toggle('hud-off', this.hidden);
  }

  setStatus({ kind, state, detail }) {
    $('status').textContent = detail;
    $('status').dataset.state = state;
    for (const [k, b] of Object.entries(this.buttons)) b.classList.toggle('on', k === kind);
    document.body.classList.toggle('demo', kind === 'demo');
  }

  setStats(text) {
    $('stats').textContent = text;
  }

  setBpm(finger, bpm) {
    const el = $('bpm');
    el.classList.toggle('show', finger);
    $('bpm-val').textContent = bpm ? Math.round(bpm) : '··';
  }

  beat() {
    const el = $('bpm');
    el.classList.remove('beat');
    void el.offsetWidth; // restart the CSS animation
    el.classList.add('beat');
  }

  setRecording(on) {
    $('rec').classList.toggle('show', on);
  }

  // Firmware comment lines (# ...), shown briefly.
  comment(text) {
    const el = $('comment');
    el.textContent = text;
    el.classList.add('show');
    clearTimeout(this.commentTimer);
    this.commentTimer = setTimeout(() => el.classList.remove('show'), 6000);
  }

  toast(text) {
    this.comment(text);
  }
}
