/* =====================================================================
   site.js — shared helpers, nav, and the content that is genuinely static
   (the 14 class definitions, the team placeholders, the link list).
   Anything that is a measured number is NOT here: it comes from
   static/data/*.json at runtime.
   ===================================================================== */
(function () {
  "use strict";

  var DATA_BASE = "static/data/";
  var state = { eda: null, results: null, fetchErrors: {} };

  function $(id) { return document.getElementById(id); }
  function qs(sel, root) { return (root || document).querySelector(sel); }

  function loadJSON(name) {
    return fetch(DATA_BASE + name, { cache: "no-cache" })
      .then(function (r) {
        if (!r.ok) throw new Error(name + " -> HTTP " + r.status);
        return r.json();
      })
      .then(function (j) {
        state[name.replace(/\.json$/, "")] = j;
        document.dispatchEvent(new CustomEvent("data:" + name.replace(/\.json$/, "")));
        return j;
      })
      .catch(function (err) {
        state.fetchErrors[name] = err.message;
        console.warn("[site] could not load " + name + ": " + err.message);
        document.dispatchEvent(new CustomEvent("data:" + name.replace(/\.json$/, "")));
        return null;
      });
  }

  function esc(s) {
    return String(s === null || s === undefined ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function num(v, d) {
    if (v === null || v === undefined || (typeof v === "number" && !isFinite(v))) return "—";
    return Charts.fmt(v, d);
  }

  function bytes(n) {
    if (!n) return "—";
    var u = ["B", "KB", "MB", "GB"], i = 0, v = n;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return (v >= 10 || i === 0 ? Math.round(v) : v.toFixed(1)) + " " + u[i];
  }

  function seconds(s) {
    if (s === null || s === undefined) return "—";
    return s >= 60 ? Charts.clock(s) : s.toFixed(1) + " s";
  }

  function statBlock(items) {
    return items.map(function (it) {
      return '<div class="stat"><span class="v">' + it.v + '</span><span class="k">' + esc(it.k) + "</span></div>";
    }).join("");
  }

  function provenance(parts) {
    // Plain flex items with a gap: an explicit "·" separator ends up orphaned at
    // the start of a wrapped line on narrow screens.
    return parts.filter(Boolean).map(function (p) {
      return "<span>" + p + "</span>";
    }).join("");
  }

  /* --------------------------------------------------------------- nav */
  function initNav() {
    var btn = $("menuBtn"), nav = $("nav");
    if (btn && nav) {
      btn.addEventListener("click", function () {
        var open = nav.classList.toggle("open");
        btn.setAttribute("aria-expanded", open ? "true" : "false");
        btn.setAttribute("aria-label", open ? "Close menu" : "Open menu");
      });
      nav.addEventListener("click", function (e) {
        if (e.target.tagName === "A") { nav.classList.remove("open"); btn.setAttribute("aria-expanded", "false"); }
      });
    }
    var links = [].slice.call(document.querySelectorAll(".nav a"));
    var targets = links.map(function (a) { return document.querySelector(a.getAttribute("href")); });
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          links.forEach(function (a, i) { a.classList.toggle("active", targets[i] === e.target); });
        });
      }, { rootMargin: "-45% 0px -50% 0px" });
      targets.forEach(function (t) { if (t) io.observe(t); });
    }
  }

  /* ------------------------------------------------- the 14 class table
     Transcribed from src/rules/engine.py + configs/default.json. The
     thresholds are physical (metres, m/s, m/s²) because the rules measure
     through the scene homography, not in pixels — see the "scale-free"
     note below. */
  var CLASS_INFO = [
    ["accident", "contact between a vehicle (or rider) and any road user — centre gap ≤ <b>0.6 m</b> or bbox IoU &gt; <b>0.05</b> — <em>and</em> either a closing speed ≥ <b>1.5 m/s</b> or a deceleration ≥ <b>4 m/s²</b> at the moment of contact, held 3 s", false],
    ["near_miss", "an actual evasive manoeuvre: deceleration ≥ <b>2.5 m/s²</b> <em>and</em> closing speed ≥ <b>2 m/s</b> <em>and</em> (gap ≤ <b>3 m</b> or IoU &gt; 0.02) <em>and</em> a side-swipe or TTC ≤ 2 s, held 2 s", false],
    ["red_light", "bottom-centre trajectory crosses a stop line while heading <em>into</em> the intersection, and the nearest traffic-light ROI reads red in HSV, held 0.8 s", true],
    ["wrong_way", "moving ≥ <b>1.5 m/s</b> with velocity·lane-direction &lt; <b>−0.45</b> for 3 consecutive sampled frames", true],
    ["illegal_u_turn", "heading change ≥ <b>2.35 rad</b> over 4 s and the lane's <code>allowed_moves</code> does not permit a U-turn", "part"],
    ["stopped_vehicle", "vehicle on a road polygon at ≤ <b>0.6 m/s</b> for ≥ <b>10 s</b>, and <em>not</em> queued at a stop line", false],
    ["jaywalking", "person on a road polygon for ≥ 1 s whose centre is inside no crosswalk polygon", false],
    ["failure_to_yield", "person inside a crosswalk polygon while a moving vehicle occupies that same polygon", true],
    ["illegal_turn", "heading deviates ≥ <b>0.55 rad</b> from the lane direction and the movement (left/right) is not in that lane's <code>allowed_moves</code>", true],
    ["solid_line_crossing", "bottom-centre trajectory crosses a polyline listed under <code>solid_lines</code>, held 0.5 s", true],
    ["stop_line", "crossed — or standing within 35 px of — a stop line while that line's signal reads red", true],
    ["congestion", "≥ 4 vehicles in one lane/direction group with ≥ 4 of them crawling at ≤ <b>2.2 m/s</b>, sustained 5 s", false],
    ["road_obstacle", "a track whose class name contains obstacle / debris / animal / cone / barrier / fallen, on the road, for ≥ 0.5 s", false],
    ["fire_smoke", "a track whose class name contains fire / smoke / flame, on the road, for ≥ 0.5 s", false]
  ];

  function initClassTable() {
    var body = qs("#classTable tbody");
    if (!body) return;
    body.innerHTML = CLASS_INFO.map(function (c) {
      var need = c[2] === true
        ? '<span class="badge-yes">yes</span>'
        : (c[2] === "part"
          ? '<span class="badge-part">partly</span>'
          : '<span class="badge-no">no</span>');
      return "<tr><td>" + c[0] + "</td><td>" + c[1] + "</td><td>" + need + "</td></tr>";
    }).join("");
    var note = $("classTableNote");
    if (note) {
      note.innerHTML = "Transcribed from <code>src/rules/engine.py</code>; every threshold is settable " +
        "in <code>configs/default.json</code>. Two things are worth noticing. First, the thresholds are " +
        "<strong>physical</strong> — metres, m/s, m/s² — not pixels, because the rules measure through " +
        "the scene's road geometry; a threshold in pixels would mean something different on a different " +
        "camera. Second, the <em>needs geometry</em> column matters: 5 of the 14 classes are " +
        "<em>disabled</em> when the scene JSON has no lanes, stop lines or light ROIs for that camera. " +
        "The pipeline refuses to guess rather than emit false positives — and, as the results section " +
        "shows, that refusal is currently costing us most of the taxonomy.";
    }
  }

  /* -------------------------------------------------------------- team */
  var TEAM = [
    {
      slot: "U",
      name: "Umidjon Axmedov",
      role: "Team Lead & AI Engineer",
      focus: "Responsible for the computer-vision pipeline, model development and fine-tuning, AI experiments, and technical direction.",
      links: [["GitHub", "https://github.com/axumfa"], ["LinkedIn", "https://www.linkedin.com/in/umid-akhmedov-931418359/"]]
    },
    {
      slot: "A",
      name: "Asilbek Asqarov",
      role: "Software Developer",
      focus: "Responsible for the main application implementation and integration of its components.",
      links: [["GitHub", "https://github.com/AsilbekAsqarov17"], ["LinkedIn", "https://www.linkedin.com/in/asilbek-asqarov-63704035a/"]]
    },
    {
      slot: "A",
      name: "Asadbek Asrarkhanov",
      role: "DevOps & Technical Support",
      focus: "Responsible for deployment, technical support, and integration across the software and AI components.",
      links: [["GitHub", "https://github.com/locksley-w4"], ["LinkedIn", "https://www.linkedin.com/in/asadbek-asrarkhanov-b7b890326/"]]
    }
  ];

  var WORK = [
    ["Detector: YOLO11n → ONNX, CPU path, stride choice", "Umidjon Axmedov", "done — 10.7 MB ONNX, no PyTorch at inference"],
    ["ByteTrack-style tracker (dependency-free fallback)", "Umidjon Axmedov", "done — upstream ByteTrack is not installed, so the built-in two-stage IoU tracker runs"],
    ["RuleEngine: 12 of 14 classes", "Umidjon Axmedov", "done — thresholds in <code>configs/default.json</code>"],
    ["Scene calibration (lanes / stop lines / crosswalks / light ROIs)", "Umidjon Axmedov", "partial — only <code>data_video1</code> is calibrated; hidden test names resolve to the shipped default scene"],
    ["Part B causal risk scorer", "Umidjon Axmedov", "partial — causal, but not a calibrated positive-risk benchmark"],
    ["Detector fine-tune on pseudo-labels", "Umidjon Axmedov", "not used — pseudo-labels remain unreviewed and the shipped model is stock COCO YOLO11n"],
    ["Human event labels for the sample clips", "Umidjon Axmedov, Asilbek Asqarov", "done — development annotations recorded in <code>data/annotations/</code>"],
    ["This website + demo backend", "Asadbek Asrarkhanov, Asilbek Asqarov", "done"],
    ["Write-up and final predictions", "All members", "done — limitations and measured results are published on the site"]
  ];

  function initTeam() {
    var grid = $("teamGrid");
    if (grid) {
      grid.innerHTML = TEAM.map(function (m) {
        return '<article class="member">' +
          '<div class="avatar">' + esc(m.slot) + '</div>' +
          "<h3>" + m.name + "</h3>" +
          '<div class="role">' + m.role + "</div>" +
          "<p>" + m.focus + "</p>" +
          '<div class="links">' + m.links.map(function (l) {
            return '<a class="todo" href="' + esc(l[1]) + '" target="_blank" rel="noopener">' + l[0] + "</a>";
          }).join("") + "</div>" +
          "</article>";
      }).join("");
    }
    var wt = qs("#workTable tbody");
    if (wt) {
      wt.innerHTML = WORK.map(function (r) {
        var st = r[2].indexOf("done") === 0
          ? '<span class="badge-yes">done</span> ' + r[2].slice(5)
          : (r[2].indexOf("partial") === 0
            ? '<span class="badge-part">partial</span> ' + r[2].slice(8)
            : (r[2].indexOf("in progress") === 0
              ? '<span class="badge-part">in progress</span> ' + r[2].slice(12)
              : '<span class="badge-no">not started</span> ' + r[2]));
        return "<tr><td>" + r[0] + "</td><td>" + r[1] + "</td><td>" + st + "</td></tr>";
      }).join("");
    }
  }

  /* ------------------------------------------------------------- links */
  function buildLinks() {
    var repoGuess = "https://github.com/AsilbekAsqarov17/hackathon_cctv";
    var links = [
      { k: "source", t: "Source repository", d: "solution.py + src/ + configs/ + tools/", u: null, todo: repoGuess },
      { k: "weights", t: "Detector weights", d: "weights/yolo11n.onnx — 10.7 MB, COCO-pretrained YOLO11n exported to ONNX", u: "../weights/yolo11n.onnx" },
      { k: "pred", t: "predictions_samples.json", d: "The official submission file for the two sample clips.", u: null, pending: true,
        detail: "Not linked, deliberately: the repository <code>.gitignore</code> excludes " +
          "<code>predictions*.json</code>, so this file can never be committed and a link to it here " +
          "would be a dead link on every deployment. Produce it with " +
          "<code>python run_submission.py --videos data/samples_small --out predictions_samples.json " +
          "--team 798C27C9</code> and it lands at the repository root." },
      { k: "solution", t: "solution.py", d: "The only file a team has to implement — CLASSES, detect_events, RiskEstimator", u: "../solution.py" },
      { k: "arch-a", t: "docs/PART_A_ARCHITECTURE.md", d: "Runtime chain, canonical state, rules contract, scene configuration", u: "../docs/PART_A_ARCHITECTURE.md" },
      { k: "arch-b", t: "docs/PART_B_ARCHITECTURE.md", d: "Causality contract and the Part A → Part B feature cache", u: "../docs/PART_B_ARCHITECTURE.md" },
      { k: "research", t: "research/PART_B_NOTES.md", d: "Why DSTA / DoTA were not the runtime foundation", u: "../research/PART_B_NOTES.md" },
      { k: "metric", t: "evaluate.py", d: "The official metric: tIoU macro-F1 and the Part B AP / alarm / mTTA formula", u: "../evaluate.py" },
      { k: "labels", t: "data/annotations/my_labels.json", d: "Our own reviewed annotations. One clip is certified to contain no reportable event, which makes it a negative sample and turns Score_A into a direct false-positive measurement", u: "../data/annotations/my_labels.json" },
      { k: "eda-json", t: "static/data/eda.json", d: "Raw EDA measurements this page renders — resolution, counts, heatmaps, per-second series", u: "static/data/eda.json" },
      { k: "res-json", t: "static/data/results.json", d: "Raw pipeline output this page renders — events, risk curve, runtimes", u: "static/data/results.json" },
      { k: "data-readme", t: "static/data/README.md", d: "How both JSON files are generated, and how to read them without the page", u: "static/data/README.md" },
      { k: "eda-tool", t: "tools/analyze_samples.py", d: "The script that produced eda.json. Re-runnable.", u: "tools/analyze_samples.py" },
      { k: "res-tool", t: "tools/run_samples.py", d: "The script that produced results.json by calling solution.py directly.", u: "tools/run_samples.py" },
      { k: "check-site", t: "tools/check_site.js", d: "Headless-Chrome smoke test: console errors, failed requests, overflow, and a real click-to-seek assertion", u: "tools/check_site.js" },
      { k: "demo-src", t: "demo/app.py", d: "The live-demo backend: stdlib HTTP, job queue, byte-range video, graceful degradation", u: "demo/app.py" },
      { k: "site-readme", t: "website/README.md", d: "How to run it locally and deploy it", u: "README.md" }
    ];
    var grid = $("linksGrid");
    if (!grid) return;
    grid.innerHTML = links.map(function (l) {
      var cls = "link-card" + (l.pending ? " pending" : "");
      var inner = '<span class="lk">' + esc(l.k) + "</span><span class=\"lt\">" + esc(l.t) + '</span><span class="ld">' + esc(l.d) + "</span>";
      if (l.u) {
        return '<a class="' + cls + '" href="' + esc(l.u) + '">' + inner + "</a>";
      }
      return '<div class="' + cls + '">' + inner +
        (l.detail
          ? '<span class="ld" style="margin-top:.4rem">' + l.detail + "</span>"
          : '<span class="ld" style="margin-top:.4rem"><span class="todo">' + esc(l.todo) + "</span></span>") +
        "</div>";
    }).join("");
    var note = $("linksNote");
    if (note) {
      note.innerHTML = "Paths above are relative to the <code>website/</code> folder, so they resolve " +
        "correctly both from a checkout (where the repository sits one level up) and from a static host. " +
        "Every one of them is a real file in the repository — the one exception is called out on its own " +
        "card, with the reason.";
    }
  }

  /* --------------------------------------------------------- hero stats */
  function renderHero() {
    var host = $("heroStats"), note = $("heroNote");
    if (!host) return;

    // Always replace the skeletons on the first attempt, whether or not the
    // data arrived. An eternal shimmer reads as "still loading"; a plain
    // statement reads as honest. The fallback deliberately shows only facts
    // that come from the task specification, never a measured number — those
    // live in the JSON and nowhere else.
    function fallback(reason) {
      host.innerHTML = statBlock([
        { v: "14", k: "event classes required" },
        { v: "2", k: "outputs per video" },
        { v: "5 s", k: "anticipation horizon H" },
        { v: "10 s", k: "alarm window W" }
      ]);
      if (note) {
        note.innerHTML = "<strong>These four are constants from the task specification, not measurements.</strong> " +
          reason + " Every measured figure on this page is read at runtime from " +
          "<code>static/data/eda.json</code> and <code>static/data/results.json</code>, written by " +
          "<code>website/tools/analyze_samples.py</code> and <code>website/tools/run_samples.py</code>.";
      }
    }

    function paint() {
      var eda = state.eda, res = state.results;
      if (!eda && !res) {
        var missing = [];
        if (!eda) missing.push("eda.json");
        if (!res) missing.push("results.json");
        fallback("No generated data was found (<code>" + missing.join("</code>, <code>") +
          "</code>), so no measured figure is shown here yet.");
        return;
      }
      var items = [], srcs = [];
      if (eda && eda.videos && eda.videos.length) {
        var v = eda.videos[0], d = v.detection, c = v.container;
        items.push({ v: c.width + "×" + c.height, k: "camera resolution" });
        items.push({ v: c.frames_decoded.toLocaleString() + " fr", k: "frames per clip" });
        items.push({ v: num(d.objects_per_sampled_frame.mean, 1), k: "road users / sampled frame" });
        items.push({ v: Math.round(d.pedestrians.share_of_sampled_frames * 100) + "%", k: "frames with a pedestrian" });
        srcs.push("EDA");
      }
      if (res && res.videos && res.videos.length) {
        var rv = res.videos[0];
        items.push({ v: String((rv.events || []).length), k: "predicted segments, clip 1" });
        var rt = rv.runtime || {};
        items.push({
          v: num(rt.part_a_sec, 0) + " s",
          k: "Part A on CPU, clip 1 (" + num(rt.part_a_realtime_factor, 2) + "× real time)"
        });
        srcs.push("measured output");
      }
      if (!items.length) { fallback("The generated JSON is present but empty."); return; }
      host.innerHTML = statBlock(items);
      if (note) {
        note.innerHTML = "Sources: " + srcs.join(" + ") + " — generated " +
          (state.eda || state.results).generated_utc + " UTC from the two sample clips. " +
          "No number on this page was typed by hand.";
      }
    }
    paint();
    document.addEventListener("data:eda", paint);
    document.addEventListener("data:results", paint);
    document.addEventListener("data:ready", paint);
  }

  function initFootNote() {
    var n = $("footNote");
    if (!n) return;
    var bits = [];
    if (state.eda) bits.push("EDA generated " + state.eda.generated_utc + " UTC");
    if (state.results) bits.push("results generated " + state.results.generated_utc + " UTC");
    if (state.results && state.results.repo_commit) {
      bits.push("repo " + state.results.repo_commit + (state.results.repo_dirty ? " (working tree was dirty)" : ""));
    }
    n.textContent = bits.join(" · ");
  }

  /* ------------------------------------------- inline measured placeholders
     <span data-from="eda" data-stat="objects_mean_A"></span> in the prose is
     filled from the JSON, so a sentence can quote a real number without the
     surrounding HTML pretending to know it. */
  function fillInline() {
    var stats = buildStatTable();
    [].slice.call(document.querySelectorAll("[data-from]")).forEach(function (el) {
      var v = stats[el.getAttribute("data-stat")];
      el.textContent = (v === undefined || v === null) ? "—" : String(v);
      el.setAttribute("title", "measured — see the EDA / results section");
    });
  }

  function buildStatTable() {
    var t = {};
    var eda = state.eda, res = state.results;
    if (eda && eda.videos) {
      t.videos = eda.videos.length;
      eda.videos.forEach(function (v, i) {
        var s = i === 0 ? "A" : i === 1 ? "B" : String(i + 1);
        t["objects_mean_" + s] = num(v.detection.objects_per_sampled_frame.mean, 1);
        t["ped_share_" + s] = Math.round(v.detection.pedestrians.share_of_sampled_frames * 100) + "%";
        t["frames_" + s] = v.container.frames_decoded.toLocaleString();
        t["ped_mean_" + s] = num(v.detection.pedestrians.mean_per_sampled_frame, 1);
        t["max_ped_" + s] = v.detection.pedestrians.max_in_one_frame;
        t["bus_share_" + s] = Math.round((v.detection.per_class.bus || { share_of_sampled_frames: 0 }).share_of_sampled_frames * 100) + "%";
      });
    }
    if (res && res.videos) {
      t.res_videos = res.videos.length;
      res.videos.forEach(function (v, i) {
        var s = i === 0 ? "A" : i === 1 ? "B" : String(i + 1);
        t["events_" + s] = (v.events || []).length;
        t["classes_" + s] = Object.keys(v.event_counts_by_label || {}).length;
        t["overlaps_" + s] = (v.validation || {}).same_class_overlaps;
        t["rt_a_" + s] = num((v.runtime || {}).part_a_sec, 0) + " s";
        t["rt_factor_" + s] = num((v.runtime || {}).part_a_realtime_factor, 2) + "×";
        t["risk_mean_" + s] = num((v.risk || {}).mean, 3);
        t["risk_above_" + s] = Math.round(((v.risk || {}).share_at_or_above_threshold || 0) * 100) + "%";
        t["risk_alarms_" + s] = (v.risk || {}).alarm_count;
        t["scene_" + s] = (v.scene || {}).scene_id || "—";
      });
    }
    return t;
  }

  /* ---------------------------------------------------------------- go */
  document.addEventListener("DOMContentLoaded", function () {
    initNav();
    initClassTable();
    initTeam();
    buildLinks();
    renderHero();
    Promise.all([loadJSON("eda.json"), loadJSON("results.json")]).then(function () {
      initFootNote();
      fillInline();
      document.dispatchEvent(new CustomEvent("data:ready"));
    });
  });

  window.Site = {
    state: state, loadJSON: loadJSON, esc: esc, num: num, bytes: bytes, seconds: seconds,
    statBlock: statBlock, provenance: provenance, buildStatTable: buildStatTable, $: $
  };
})();
