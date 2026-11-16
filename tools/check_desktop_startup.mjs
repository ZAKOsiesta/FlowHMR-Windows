import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';

const noop = () => {};
const component = () => new Proxy({}, { get: () => noop });
function environment() {
  const timers = [], listeners = new Map(), nodes = new Map();
  const node = (id) => {
    if (!nodes.has(id)) nodes.set(id, { classList: { toggle: noop }, dataset: {},
      addEventListener: noop, setAttribute: noop, querySelectorAll: () => [] });
    return nodes.get(id);
  };
  const window = { addEventListener: (name, fn) => listeners.set(name, fn),
    dispatchEvent: noop, matchMedia: () => ({ matches: false, addEventListener: noop }) };
  const context = vm.createContext({ window, document: { documentElement: { dataset: {} },
    addEventListener: noop, querySelectorAll: () => [], querySelector: () => null },
    localStorage: { getItem: () => null }, CustomEvent: class {}, HTMLElement: class {},
    MutationObserver: class { observe() {} }, console,
    setTimeout: (fn) => { timers.push(fn); return fn; },
    clearTimeout: (fn) => { const i = timers.indexOf(fn); if (i >= 0) timers.splice(i, 1); },
    setInterval: noop, AbortSignal });
  return { context, window, timers, listeners, utils: { $: node, show: noop, el: noop,
    createBus: () => ({ on: noop }), jobLabel: noop },
    i18n: { message: (key) => key, setText: noop, setupLocalization: noop, appError: Error } };
}
async function module(file, env, imports) {
  const source = await fs.readFile(new URL(`./web/static/js/${file}`, import.meta.url), 'utf8');
  const actual = new vm.SourceTextModule(source, { context: env.context });
  await actual.link((specifier) => {
    const exports = imports[specifier];
    assert.ok(exports, `Missing fixture for ${specifier}`);
    return new vm.SyntheticModule(Object.keys(exports), function () {
      for (const [key, value] of Object.entries(exports)) this.setExport(key, value);
    }, { context: env.context });
  });
  return actual;
}
async function tick(env) {
  // Drain promise continuations before advancing the controlled retry timer.
  for (let i = 0; i < 12; i++) await Promise.resolve();
  env.timers.shift()?.();
  for (let i = 0; i < 12; i++) await Promise.resolve();
}

async function checkHandshake(firstResult) {
  const env = environment();
  let calls = 0;
  env.window.pywebview = { api: {
    check_updates: async () => ({ version: null }),
    confirm_update_boot: async () => {
      calls++;
      if (calls === 1 && firstResult === 'reject') throw new Error('bridge not ready');
      if (calls === 1 && firstResult === 'hang') return new Promise(noop);
      return calls === 1 ? firstResult : true;
    },
  } };
  const actual = await module('update.js', env, { './utils.js': env.utils, './i18n.js': env.i18n });
  await actual.evaluate();
  const ready = actual.namespace.initUpdates();
  await tick(env);
  assert.equal(calls, 0, 'Uninitialized workspace must never confirm boot');
  ready();
  for (let i = 0; i < 6; i++) await tick(env);
  assert.equal(calls, 2, 'A rejected or unacknowledged boot confirmation must retry');
  ready();
  await tick(env);
  assert.equal(calls, 2, 'Acknowledged boot must not be sent again');
}

async function checkStartupInfo(recover = true) {
  const env = environment();
  let infoCalls = 0, confirmations = 0;
  const imports = {
    './utils.js': env.utils, './i18n.js': env.i18n,
    './api.js': { getInfo: async () => {
      if (++infoCalls === 1 || !recover) throw new Error('temporary local connection failure');
      return { model_files: { items: [] }, defaults: {} };
    } },
    './upload.js': { initUpload: () => ({ getFile: () => null }) },
    './params.js': { initParams: component }, './progress.js': { initProgress: component },
    './person-select.js': { initPersonSelect: component }, './preview.js': { initPreview: component },
    './history.js': { initHistory: () => ({ refresh: async () => [] }) },
    './viewer.js': { initViewer: component }, './layout.js': { initLayout: noop },
    './trim.js': { initTrim: component }, './models.js': { initModelDialog: noop },
    './runtime.js': { initRuntimeDialog: component }, './motion-export.js': { initMotionExport: component },
    './update.js': { initUpdates: () => () => { confirmations++; } },
    './mcp.js': { initMcpDialog: noop }, './ui-events.js': { initUIEvents: noop },
  };
  const actual = await module('main.js', env, imports);
  const running = actual.evaluate();
  for (let i = 0; i < 10; i++) await tick(env);
  await running;
  assert.equal(infoCalls, recover ? 2 : 3, 'Local connection retries must be bounded');
  assert.equal(confirmations, recover ? 1 : 0, 'Only a recovered workspace may report ready');
}

const failures = [];
for (const [name, test] of [
  ['bridge rejection', () => checkHandshake('reject')],
  ['bridge acknowledgement', () => checkHandshake(false)],
  ['bridge timeout', () => checkHandshake('hang')],
  ['initial local connection', () => checkStartupInfo()],
  ['persistent local failure', () => checkStartupInfo(false)],
]) {
  try { await test(); console.log(`PASS ${name}`); }
  catch (error) { failures.push(name); console.error(`${name}: ${error.stack}`); }
}
assert.deepEqual(failures, []);
