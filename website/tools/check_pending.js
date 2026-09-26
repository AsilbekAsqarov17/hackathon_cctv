#!/usr/bin/env node
/**
 * check_pending.js — assert the site degrades honestly when the generated JSON
 * is missing. Runs the same DevTools-protocol harness as check_site.js but with
 * a data directory that contains no eda.json / results.json.
 *
 *   node website/tools/check_pending.js <url>
 */
"use strict";
const { spawn } = require("child_process");
const http = require("http");
const fs = require("fs");
const path = require("path");

const URL_ARG = process.argv[2] || "http://127.0.0.1:8130/";
const OUT = process.argv[3] || "/tmp/site-pending";
const PORT = 9900 + (process.pid % 90);
fs.mkdirSync(OUT, { recursive: true });
const chrome = spawn(process.env.CHROME || "google-chrome", [
  "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--hide-scrollbars",
  "--window-size=1280,900", "--user-data-dir=" + path.join("/tmp", "pending-" + process.pid),
  "--remote-debugging-port=" + PORT, "about:blank",
], { stdio: ["ignore", "pipe", "pipe"] });
let cerr = "";
chrome.stderr.on("data", (d) => (cerr += d.toString()));
const get = (p) => new Promise((res, rej) => {
  http.get({ host: "127.0.0.1", port: PORT, path: p }, (r) => {
    let b = ""; r.on("data", (c) => (b += c)); r.on("end", () => { try { res(JSON.parse(b)); } catch (e) { rej(e); } });
  }).on("error", rej);
});
async function waitFor(fn, ms = 25000) {
  const t0 = Date.now();
  for (;;) { try { return await fn(); } catch (e) {} if (Date.now() - t0 > ms) throw new Error("chrome timeout"); await new Promise((r) => setTimeout(r, 250)); }
}
(async () => {
  const t = await waitFor(() => get("/json/list"));
  const page = t.find((x) => x.type === "page");
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 0; const pending = new Map(); const errors = [];
  ws.addEventListener("message", (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); return; }
    if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") {
      errors.push("console.error: " + (m.params.args || []).map((a) => a.value ?? a.description).join(" "));
    }
    if (m.method === "Runtime.exceptionThrown") {
      const d = m.params.exceptionDetails;
      errors.push("uncaught: " + ((d.exception && (d.exception.description || d.exception.value)) || d.text));
    }
  });
  await new Promise((r) => ws.addEventListener("open", r));
  const send = (method, params) => new Promise((res) => { const n = ++id; pending.set(n, res); ws.send(JSON.stringify({ id: n, method, params: params || {} })); });
  const evaluate = async (expr) => {
    const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) return { error: r.exceptionDetails.text };
    return { value: r.result && r.result.value };
  };
  await send("Runtime.enable"); await send("Page.enable");
  await send("Page.navigate", { url: URL_ARG });
  await new Promise((r) => setTimeout(r, 7000));
  const probe = (await evaluate(`(function () {
    function n(s) { return document.querySelectorAll(s).length; }
    function txt(sel) { var e = document.querySelector(sel); return e ? e.textContent.replace(/\\s+/g," ").trim() : null; }
    return {
      heroSkeletonLeft: n("#heroStats .skeleton"),
      heroStats: n("#heroStats .stat"),
      classRows: n("#classTable tbody tr"),
      teamCards: n("#teamGrid .member"),
      linkCards: n("#linksGrid .link-card"),
      pipelineNodes: n("#pipeline .node"),
      edaPending: n("#edaBody .pending-box"),
      edaText: txt("#edaBody .pending-box"),
      edaProv: txt("#edaProv"),
      resPending: n("#resBody .pending-box"),
      resText: txt("#resBody .pending-box"),
      resProv: txt("#resProv"),
      reportPending: n("#reportBody .pending-box"),
      reportText: txt("#reportBody .pending-box"),
      footNote: txt("#footNote"),
      demoAlert: (txt("#demoAlert") || "").slice(0, 130),
      runDisabled: (document.getElementById("runBtn") || {}).disabled,
      brokenImages: [].slice.call(document.images).filter(function (i) { return i.complete && i.naturalWidth === 0; }).map(function (i) { return i.getAttribute("src"); }),
      horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth + 2
    };
  })()`)).value;
  const s = await send("Page.captureScreenshot", { format: "png" });
  fs.writeFileSync(path.join(OUT, "pending-results.png"), Buffer.from(s.data, "base64"));
  await evaluate("window.scrollTo(0, document.getElementById('results').getBoundingClientRect().top + scrollY)");
  await new Promise((r) => setTimeout(r, 700));
  const s2 = await send("Page.captureScreenshot", { format: "png" });
  fs.writeFileSync(path.join(OUT, "pending-results2.png"), Buffer.from(s2.data, "base64"));
  console.log(JSON.stringify({ url: URL_ARG, probe, errors }, null, 1));
  const bad = [];
  if (errors.length) bad.push(errors.length + " console error(s)");
  if (probe.horizontalOverflow) bad.push("horizontal overflow");
  if (probe.brokenImages && probe.brokenImages.length) bad.push("broken images: " + probe.brokenImages.join(", "));
  if (probe.heroSkeletonLeft) bad.push("hero stuck on skeletons");
  if (!probe.edaPending) bad.push("EDA has no pending state");
  if (!probe.resPending) bad.push("results has no pending state");
  if (!probe.reportPending) bad.push("report has no pending state");
  if (!probe.classRows) bad.push("class table empty even with no data");
  ws.close(); chrome.kill();
  console.log("\nscreenshots -> " + OUT);
  if (bad.length) { console.log("FAIL: " + bad.join("; ")); process.exit(1); }
  console.log("PASS: pending states shown, no console errors, no overflow, static content intact");
  process.exit(0);
})().catch((e) => { console.error("failed:", e.message); console.error(cerr.slice(-1200)); chrome.kill(); process.exit(1); });
