// Band stream lines.
// v0.1: t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,ir,red
// v0.2: the same + ,seq (sample counter; gaps = dropped samples)
// Lines starting with '#' are comments from the firmware.

export const HEADER = 't_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,ir,red,seq';

// Returns a sample, {comment}, or null for anything else (blank, human-mode text, garbage).
export function parseLine(line) {
  const s = line.trim();
  if (!s) return null;
  if (s[0] === '#') return { comment: s.slice(1).trim() };
  const f = s.split(',');
  if (f.length !== 9 && f.length !== 10) return null;
  const v = new Array(f.length);
  for (let i = 0; i < f.length; i++) {
    if (f[i] === '') return null;
    v[i] = Number(f[i]);
    if (!Number.isFinite(v[i])) return null;
  }
  return {
    t: v[0],
    a: [v[1], v[2], v[3]],
    g: [v[4], v[5], v[6]],
    ir: v[7],
    red: v[8],
    seq: f.length === 10 ? v[9] : null,
  };
}

// Turns stream chunks (which may cut lines anywhere) into whole lines.
export class LineSplitter {
  constructor() {
    this.rest = '';
  }

  push(chunk) {
    const parts = (this.rest + chunk).split(/\r?\n/);
    this.rest = parts.pop();
    if (this.rest.length > 4096) this.rest = ''; // no newline for ages: not our stream
    return parts;
  }
}
