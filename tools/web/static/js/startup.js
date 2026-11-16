// A classic script runs even when a module fails to download, parse or import.
// Keep startup diagnostics local and stop collecting errors once the UI is ready.
(() => {
  const pending = [];
  let sending = false, ready = false, errors = 0;
  async function flush() {
    const report = window.pywebview?.api?.report_workspace_boot;
    if (sending || !report) return;
    sending = true;
    try {
      while (pending.length) {
        if (await report(...pending[0]) !== true) break;
        pending.shift();
      }
    } catch { /* The next startup stage or bridge event will retry diagnostics. */ }
    finally { sending = false; }
  }
  function record(phase, detail = "") {
    pending.push([phase, detail]);
    void flush();
  }
  function error(detail) {
    if (!ready && errors++ < 4) record("error", String(detail).slice(0, 1000));
  }
  window.flowhmrStartup = {
    stage: (phase) => { record(phase); if (phase === "ready") ready = true; }, error,
  };
  window.addEventListener("error", (event) => {
    const source = event.filename || event.target?.src || "";
    let path = "";
    try { path = new URL(source, location.href).pathname; } catch { /* no source URL */ }
    error(`${event.message || "Resource failed to load"} (${path}:${event.lineno || 0})`);
  }, true);
  window.addEventListener("unhandledrejection", (event) => error(event.reason?.message || event.reason));
  window.addEventListener("pywebviewready", flush);
  record("modules");
})();
