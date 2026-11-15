// Run with Node; no server or browser dependencies are required.
import assert from 'node:assert/strict';
import { catalogs } from './web/static/js/i18n.js';
import { cameraStatusMessage, multishotMessage } from './web/static/js/camera-messages.js';
const canonical = catalogs['zh-CN'];
const keys = Object.keys(canonical).sort();
const placeholders = text => [...new Set(text.match(/\{\w+\}/g) ?? [])].sort();
for (const [language, catalog] of Object.entries(catalogs)) {
  assert.deepEqual(Object.keys(catalog).sort(), keys, `${language}: keys`);
  for (const key of keys) {
    assert.ok(catalog[key].trim(), `${language}: empty ${key}`);
    assert.deepEqual(placeholders(catalog[key]), placeholders(canonical[key]), `${language}: ${key}`);
  }
}
for (const state of ['none','queued','running','cancel_requested','ready','cancelled','failed']) {
  assert.ok(canonical[cameraStatusMessage({camera_state:state}).key], state);
}
for (const text of ['等待恢复摄像机','摄像机整体尺度 · 1/2','对齐原片机位 · 1/2',
  '补算人物观测 · 1/2 帧','复核人物可见性 · 1/2 个镜头','核对相邻画面 · 1/2 个镜头',
  '镜头条件 · 1/2','已自动连接 2 个镜头','多人加切镜暂不连接，已按原流程分别处理人物。']) {
  const message = multishotMessage(text);
  assert.ok(message && canonical[message.key], text);
  assert.deepEqual(Object.keys(message.params).sort(), placeholders(canonical[message.key]).map(x=>x.slice(1,-1)).sort());
}
console.log(JSON.stringify({languages:Object.keys(catalogs), keys:keys.length, progressMessages:'passed'}));
