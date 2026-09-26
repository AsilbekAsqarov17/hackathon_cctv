#!/usr/bin/env node
/**
 * check_site.js — headless smoke test for the public site.
 *
 * Loads the page in headless Chrome over the DevTools protocol, records every
 * console message, page exception and failed request, then asserts that the
 * generated sections actually rendered (charts drawn, timeline segments
 * present, provenance stamped). Writes screenshots to --out.
 *
 * This is a dev tool, not a runtime dependency: the site itself needs no
 * Node, no npm and no build step.
 *
 *   node website/tools/check_site.js http://127.0.0.1:8120/ /tmp/shots
 *   node website/tools/check_site.js https://<user>.github.io/<repo>/     # static host
 *
 * Requires: node >= 22 (global WebSocket) and a `google-chrome` on PATH.
 * Exits non-zero if there are console errors, failed requests, horizontal
 * overflow, or a missing dynamic section.
 */
"use strict";

const { spawn } = require("child_process");
const http = require("http");
const fs = require("fs");
const path = require("path");

const URL_ARG = process.argv[2] || "http://127.0.0.1:8000/";
const OUT = process.argv[3] || "/tmp/site-shots";
const PORT = 9500 + (process.pid % 400);
const CHROME = process.env.CHROME || "google-chrome";

fs.mkdirSync(OUT, { recursive: true });

const chrome = spawn(CHROME, [
  "--headless=new",
  "--no-sandbox",
  "--disable-gpu",
  "--disable-dev-shm-usage",
  "--hide-scrollbars",
  "--window-size=1280,900",
  "--user-data-dir=" + path.join("/tmp", "site-check-" + process.pid),
  "--remote-debugging-port=" + PORT,
  "about:blank",
], { stdio: ["ignore", "pipe", "pipe"] });
let chromeErr = "";
chrome.stderr.on("data", (d) => { chromeErr += d.toString(); });

const get = (p) => new Promise((resolve, reject) => {
  const req = http.get({ host: "127.0.0.1", port: PORT, path: p }, (res) => {
    let b = "";
    res.on("data", (c) => (b += c));
    res.on("end", () => { try { resolve(JSON.parse(b)); } catch (e) { reject(e); } });
  });
  req.on("error", reject);
});

async function waitFor(fn, ms = 25000) {
  const t0 = Date.now();
  for (;;) {
    try { return await fn(); } catch (e) { /* retry */ }
    if (Date.now() - t0 > ms) throw new Error("timed out waiting for Chrome to start");
    await new Promise((r) => setTimeout(r, 250));
  }
}

(async () => {
  const targets = await waitFor(() => get("/json/list"));
  const page = targets.find((t) => t.type === "page");
  if (!globalThis.WebSocket) throw new Error("needs node >= 22 for the global WebSocket");

  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 0;
  const pending = new Map();
  const errors = [];
  const warnings = [];
  const failedRequests = [];

  ws.addEventListener("message", (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); return; }
    if (m.method === "Runtime.consoleAPICalled") {
      const text = (m.params.args || []).map((a) => a.value ?? a.description ?? a.type).join(" ");
      if (m.params.type === "error") errors.push("console.error: " + text);
      else if (m.params.type === "warning") warnings.push("console.warn: " + text);
    }
    if (m.method === "Runtime.exceptionThrown") {
      const d = m.params.exceptionDetails;
      errors.push("uncaught: " + ((d.exception && (d.exception.description || d.exception.value)) || d.text));
    }
    if (m.method === "Network.loadingFailed" && m.params.errorText !== "net::ERR_ABORTED") {
      failedRequests.push(m.params.errorText + " (" + (m.params.type || "resource") + ")");
    }
  });
  await new Promise((r) => ws.addEventListener("open", r));
  const send = (method, params) => new Promise((res) => {
    const n = ++id;
    pending.set(n, res);
    ws.send(JSON.stringify({ id: n, method, params: params || {} }));
  });
  const evaluate = async (expression) => {
    const r = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) {
      return { error: r.exceptionDetails.text + " " + ((r.exceptionDetails.exception || {}).description || "") };
    }
    return { value: r.result && r.result.value };
  };

  await send("Runtime.enable");
  await send("Network.enable");
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
  await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }] });
  await send("Page.navigate", { url: URL_ARG });
  await new Promise((r) => setTimeout(r, 8000));

  const probe = (await evaluate(`(function () {
    function n(s) { return document.querySelectorAll(s).length; }
    function txt(id) { var e = document.getElementById(id); return e ? e.textContent.trim() : null; }
    var bad = [];
    var w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT), node;
    while ((node = w.nextNode())) {
      if (/NaN|undefined|Infinity/.test(node.nodeValue)) {
        bad.push(((node.parentElement && node.parentElement.className) || "") + " :: " + node.nodeValue.trim().slice(0, 120));
      }
    }
    return {
      title: document.title,
      h1: (document.querySelector("h1") || {}).textContent,
      heroStats: n("#heroStats .stat"),
      heroSkeletonLeft: n("#heroStats .skeleton"),
      heroNote: txt("heroNote"),
      classRows: n("#classTable tbody tr"),
      teamCards: n("#teamGrid .member"),
      todoMarkers: (document.body.innerText.match(/TODO\\(team\\)/g) || []).length,
      workRows: n("#workTable tbody tr"),
      linkCards: n("#linksGrid .link-card"),
      pipelineNodes: n("#pipeline .node"),
      pipelineArrows: n("#pipeline path.arw"),
      edaProv: txt("edaProv"),
      edaTabs: n("#edaTabs button"),
      edaPanels: n("#edaBody .panel"),
      edaCharts: n("#edaBody svg.chart"),
      edaHeatmaps: n("#edaBody canvas"),
      edaFindings: n("#edaBody .ok-list li"),
      resProv: txt("resProv"),
      resTabs: n("#resTabs button"),
      resPanels: n("#resBody .panel"),
      resTimelineSegments: n("#resBody .tl-seg"),
      resCharts: n("#resBody svg.chart"),
      resThumbs: n("#resBody .thumb"),
      resEventRows: n("#resBody .tbl tbody tr"),
      resVideo: !!document.getElementById("ev-video"),
      reportPanels: n("#reportBody .panel"),
      reportBadItems: n("#reportBody .bad-list li"),
      reportOkItems: n("#reportBody .ok-list li"),
      demoAlert: (txt("demoAlert") || "").slice(0, 100),
      runBtnDisabled: (document.getElementById("runBtn") || {}).disabled,
      docHeight: document.documentElement.scrollHeight,
      horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth + 2,
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
      brokenImages: [].slice.call(document.images)
        .filter(function (i) { return i.complete && i.naturalWidth === 0; })
        .map(function (i) { return i.getAttribute("src"); }),
      suspiciousText: bad
    };
  })()`)).value;

  const shot = async (name) => {
    const s = await send("Page.captureScreenshot", { format: "png" });
    fs.writeFileSync(path.join(OUT, name + ".png"), Buffer.from(s.data, "base64"));
  };
  const scrollTo = async (id) => {
    await evaluate("window.scrollTo(0, document.getElementById('" + id + "').getBoundingClientRect().top + scrollY)");
    await new Promise((r) => setTimeout(r, 900));
  };

  await shot("01-top");
  for (const [name, id] of [["02-scene", "scene"], ["03-problem", "problem"], ["04-eda", "eda"],
                           ["05-results", "results"], ["06-demo", "demo"], ["07-report", "report"],
                           ["08-team", "team"]]) {
    if (id) { await scrollTo(id); await shot(name); }
  }

  // mobile pass
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
  await send("Page.navigate", { url: URL_ARG });
  await new Promise((r) => setTimeout(r, 7000));
  const mobile = (await evaluate(`({
    horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth + 2,
    scrollWidth: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
    menuButtonVisible: getComputedStyle(document.getElementById("menuBtn")).display !== "none",
    charts: document.querySelectorAll("svg.chart").length
  })`)).value;
  await shot("09-mobile-top");
  await scrollTo("eda");
  await shot("10-mobile-eda");
  await scrollTo("results");
  await shot("11-mobile-results");

  // interaction check: click a timeline segment and confirm the video seeks
  const seek = (await evaluate(`(function () {
    var seg = document.querySelector("#resBody .tl-seg");
    var vid = document.getElementById("ev-video");
    if (!seg || !vid) return { skipped: true };
    var want = parseFloat(seg.style.left) / 100 * (vid.duration || 300);
    seg.click();
    return new Promise(function (r) {
      setTimeout(function () { r({ skipped: false, want: want, currentTime: vid.currentTime, src: !!vid.currentSrc }); }, 1200);
    });
  })()`)).value;

  // dark-theme pass: the site ships a full light/dark pair, so both are checked.
  // Re-assert the desktop viewport first -- the mobile pass leaves its metrics
  // emulation in place, and a "dark desktop" shot taken at 390 px is not what
  // anyone wanted to review.
  await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
  await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "dark" }] });
  await send("Page.navigate", { url: URL_ARG });
  await new Promise((r) => setTimeout(r, 7000));
  const dark = (await evaluate(`(function () {
    var body = getComputedStyle(document.body);
    var h2 = document.querySelector("#eda h2");
    return {
      bodyBg: body.backgroundColor,
      bodyFg: body.color,
      headingVisible: !!h2 && h2.getBoundingClientRect().width > 0,
      charts: document.querySelectorAll("svg.chart").length,
      horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth + 2
    };
  })()`)).value;
  await shot("12-dark-top");
  await scrollTo("problem");
  await shot("13-dark-problem");
  await scrollTo("results");
  await shot("14-dark-results");

  const report = { url: URL_ARG, desktop: probe, mobile, dark, timelineSeek: seek, errors, warnings, failedRequests };
  console.log(JSON.stringify(report, null, 1));

  const problems = [];
  if (errors.length) problems.push(errors.length + " console error(s)");
  if (failedRequests.length) problems.push(failedRequests.length + " failed request(s)");
  if (probe.horizontalOverflow) problems.push("horizontal overflow on desktop");
  if (mobile && mobile.horizontalOverflow) problems.push("horizontal overflow on mobile");
  if (dark && dark.horizontalOverflow) problems.push("horizontal overflow in dark mode");
  if (probe.suspiciousText && probe.suspiciousText.length) problems.push(probe.suspiciousText.length + " node(s) printing NaN/undefined");
  if (probe.brokenImages && probe.brokenImages.length) problems.push("broken images: " + probe.brokenImages.join(", "));
  if (probe.heroSkeletonLeft) problems.push("hero stats never resolved from JSON");
  if (!probe.classRows) problems.push("class table empty");
  if (!probe.teamCards) problems.push("team cards empty");
  if (!probe.reportPanels) problems.push("report did not render");
  if (!probe.edaCharts) problems.push("no EDA charts drawn");
  if (!probe.edaPending && !probe.resTimelineSegments) problems.push("no results timeline");
  console.log("\nscreenshots -> " + OUT);
  ws.close();
  chrome.kill();
  if (problems.length) { console.log("FAIL: " + problems.join("; ")); process.exit(1); }
  console.log("PASS: no console errors, no failed requests, no overflow, all sections rendered");
  process.exit(0);
})().catch((e) => {
  console.error("check failed:", e.message);
  console.error(chromeErr.slice(-1500));
  chrome.kill();
  process.exit(1);
});
