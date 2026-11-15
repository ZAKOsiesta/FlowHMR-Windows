import assert from 'node:assert/strict';
import { sampleFrame } from './web/static/js/frame-sample.js';

const valid = [false, true, true, false, false, true, true, true];
const shots = [0, 0, 0, 0, 1, 1, 2, 2];
for (let f = 0; f < valid.length; f++) {
  for (const fraction of [0, .01, .25, .5, .99]) {
    const sample = sampleFrame(f + fraction, valid.length, valid, shots);
    assert.equal(sample.visible, valid[f]);
    if (!valid[f] || !valid[f + 1] || shots[f] !== shots[f + 1]) {
      assert.equal(sample.frame, f);
      assert.equal(sample.alpha, 0);
    } else assert.equal(sample.frame, f + fraction);
  }
}
for (const f of [-1, NaN, Infinity, 8]) assert.equal(sampleFrame(f, 8, valid, shots).visible, false);
assert.equal(sampleFrame(2.5, 10).frame, 2.5);
console.log('Fractional gap, cut, endpoint and legacy sampling: passed');
