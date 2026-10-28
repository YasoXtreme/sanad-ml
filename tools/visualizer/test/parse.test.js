import assert from 'node:assert/strict';
import { test } from 'node:test';

import { LineSplitter, parseLine } from '../src/core/parse.js';

test('parses a v0.2 line with seq', () => {
  const s = parseLine('1234,0.010,-0.020,1.001,0.4,-0.8,0.2,81234,51000,61');
  assert.equal(s.t, 1234);
  assert.deepEqual(s.a, [0.01, -0.02, 1.001]);
  assert.deepEqual(s.g, [0.4, -0.8, 0.2]);
  assert.equal(s.ir, 81234);
  assert.equal(s.red, 51000);
  assert.equal(s.seq, 61);
});

test('parses a v0.1 line (no seq)', () => {
  const s = parseLine('20,0,0,1,0,0,0,1200,900\r');
  assert.equal(s.t, 20);
  assert.equal(s.seq, null);
});

test('comments, blanks and human-mode text', () => {
  assert.deepEqual(parseLine('# MPU-6050 init OK'), { comment: 'MPU-6050 init OK' });
  assert.equal(parseLine(''), null);
  assert.equal(parseLine('   '), null);
  assert.equal(parseLine('acc[g] x=+0.01 y=-0.02 z=+1.00 |a|=1.00'), null);
  assert.equal(parseLine('1,2,3'), null);
  assert.equal(parseLine('20,0,0,1,0,0,x,1200,900'), null);
  assert.equal(parseLine('20,0,0,1,0,0,,1200,900'), null);
});

test('LineSplitter joins lines cut across chunks', () => {
  const sp = new LineSplitter();
  assert.deepEqual(sp.push('a,b'), []);
  assert.deepEqual(sp.push('c\r\nd'), ['a,bc']);
  assert.deepEqual(sp.push('\ne\nf'), ['d', 'e']);
  assert.equal(sp.rest, 'f');
});
