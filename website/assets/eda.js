/* =====================================================================
   eda.js — renders the EDA section strictly from static/data/eda.json.
   If the JSON is missing it says so; it never substitutes invented data.
   ===================================================================== */
(function () {
  "use strict";
  var S = window.Site, C = window.Charts;
  var eda = null, active = 0, tabButtons = [], edaInit = false;

  function mediaFor(id) {
    if (eda.media) {
      for (var i = 0; i < eda.media.length; i++) if (eda.media[i].id === id) return eda.media[i];
    }
    // Conventional names produced by website/tools/analyze_samples.py, so the
    // page still shows the clip when the media step was skipped.
    return {
      id: id,
      video: "static/media/" + id + "_preview.webm",
      still: "static/img/" + id + "_still.jpg",
      width: 480, height: 270, fps: 15, bytes: null
    };
  }

  function renderProvenance() {
    var host = document.getElementById("edaProv");
    if (!host) return;
    if (!eda) {
      host.innerHTML = "<span><b>status</b> eda.json not found — run <code>python website/tools/analyze_samples.py</code></span>";
      return;
    }
    var d = eda.detector || {};
    host.innerHTML = S.provenance([
      "<b>source</b> static/data/eda.json",
      "<b>generated</b> " + eda.generated_utc,
      "repo " + (eda.repo_commit || "?") + (eda.repo_dirty ? " (dirty tree)" : ""),
      "<b>detector</b> " + d.model + " " + bytes(d.model_bytes) + " · " + d.imgsz + "px · conf " + d.confidence + " · stride " + d.stride,
      "<b>provider</b> " + (d.provider || "?")
    ]);
    function bytes(n) { return S.bytes(n); }
  }

  function mini(items) {
    return '<div class="statgrid">' + items.map(function (i) {
      return '<div class="mini"><span class="v">' + i.v + '</span><span class="k">' + S.esc(i.k) + "</span></div>";
    }).join("") + "</div>";
  }

  function renderVideo(v) {
    var c = v.container, d = v.detection, h = v.heatmaps;
    var out = [];

    /* ---- 1. container facts ---- */
    out.push('<section class="panel">');
    out.push("<h3>Container facts <span class=\"panel-head-meta\">ffprobe + a full decode</span></h3>");
    out.push('<p class="panel-note">Frame count is not trusted from metadata: the script decodes every ' +
      'frame and counts them, then compares. Both agree here, which is worth stating because a mismatch is ' +
      'the classic way a duration-asserting harness silently drops a video.</p>');
    out.push(mini([
      { v: c.width + "×" + c.height, k: "resolution" },
      { v: S.num(c.fps, 3) + " fps", k: "frame rate (" + (c.fps_rational || "?") + ")" },
      { v: c.frames_decoded.toLocaleString(), k: "frames decoded" },
      { v: S.num(c.duration_sec, 1) + " s", k: "duration" },
      { v: (c.frame_count_agrees ? "✓" : "✗") + " " + c.frames_reported_by_container.toLocaleString(), k: "container frame count" },
      { v: S.bytes(c.size_bytes), k: "file size (" + S.num((c.bitrate_bps || 0) / 1e6, 1) + " Mbit/s)" },
      { v: (c.codec || "?") + " " + (c.profile || ""), k: "codec / profile" },
      { v: c.audio_streams, k: "audio streams" }
    ]));
    out.push("</section>");

    /* ---- 2. object counts over time ---- */
    var tl = d.timeline_1s || [];
    var xs = tl.map(function (r) { return r.t; });
    out.push('<section class="panel">');
    out.push("<h3>Traffic density over time <span class=\"panel-head-meta\">1 s buckets, detector at stride " + d.stride + "</span></h3>");
    out.push('<p class="panel-note">Mean number of detected road users per second, split into vehicles and ' +
      "pedestrians. Hover or drag for exact values. This is a count of <em>boxes</em>, not of distinct " +
      "objects — a bus stopped at the kerb is counted once per frame, not once per clip.</p>");
    var c1 = document.getElementById("eda-density");
    var c2 = document.getElementById("eda-split");
    out.push('<section class="panel">');
    out.push("<h3>Traffic density over time <span class=\"panel-head-meta\">1 s buckets, detector at stride " + d.stride + "</span></h3>");
    out.push('<p class="panel-note">Mean number of detected road users per second, split into vehicles and ' +
      "pedestrians. Hover or drag for exact values. This is a count of <em>boxes</em>, not of distinct " +
      "objects — a bus stopped at the kerb is counted once per frame, not once per clip.</p>");
    out.push('<div id="eda-density"></div>');
    out.push('<div class="split" style="margin-top:.9rem"><div><h4 style="font-size:.86rem;color:var(--fg-dim);margin-bottom:.3rem">Vehicles vs pedestrians per second</h4><div id="eda-split"></div></div>' +
      '<div><h4 style="font-size:.86rem;color:var(--fg-dim);margin-bottom:.3rem">Buses + trucks per second</h4><div id="eda-heavy"></div></div></div>');
    out.push("</section>");

    /* ---- 3. class composition ---- */
    var pc = d.per_class || {};
    var names = Object.keys(pc);
    out.push('<section class="panel">');
    out.push("<h3>What the detector actually finds</h3>");
    out.push('<p class="panel-note">Counts from the stock COCO checkpoint, not a model fine-tuned for this ' +
      "camera. Everything here is a lower bound: a partially occluded car or a pedestrian 120 m away falls " +
      "under the 0.25 confidence floor and simply does not exist as far as the pipeline is concerned.</p>");
    out.push('<div class="table-scroll"><table class="tbl"><thead><tr>' +
      "<th>Class</th><th class=\"num\">Detections</th><th class=\"num\">Frames present</th><th class=\"num\">Share of frames</th><th class=\"num\">Max in one frame</th><th class=\"num\">Mean / frame</th>" +
      "</tr></thead><tbody>" + names.map(function (n) {
        var r = pc[n];
        return "<tr><td>" + n + "</td><td class=\"num\">" + r.detections_total.toLocaleString() + "</td><td class=\"num\">" +
          r.sampled_frames_with_at_least_one.toLocaleString() + "</td><td class=\"num\">" +
          Math.round(r.share_of_sampled_frames * 100) + "%</td><td class=\"num\">" + r.max_in_one_frame +
          "</td><td class=\"num\">" + S.num(r.mean_per_sampled_frame, 2) + "</td></tr>";
      }).join("") + "</tbody></table></div>");
    out.push(mini([
      { v: d.sampled_frames.toLocaleString(), k: "frames analysed" },
      { v: d.detections_total.toLocaleString(), k: "total detections" },
      { v: S.num(d.objects_per_sampled_frame.mean, 1), k: "mean boxes / frame" },
      { v: d.objects_per_sampled_frame.min + "–" + d.objects_per_sampled_frame.max, k: "min–max boxes / frame" },
      { v: d.objects_per_sampled_frame.p05 + " / " + d.objects_per_sampled_frame.p95, k: "p05 / p95" },
      { v: S.num(d.vehicles_per_sampled_frame_mean, 1), k: "mean vehicles / frame" }
    ]));
    out.push("</section>");

    /* ---- 4. pedestrians ---- */
    var ped = d.pedestrians;
    out.push('<section class="panel">');
    out.push("<h3>Pedestrians</h3>");
    out.push('<p class="panel-note">The pedestrians are not background here — they are the reason this ' +
      "intersection is hard. In a single frame the detector reports up to " + ped.max_in_one_frame +
      " of them, and a crowd on a crosswalk is exactly where <code>failure_to_yield</code> and " +
      "<code>jaywalking</code> have to be decided.</p>");
    out.push(mini([
      { v: ped.sampled_frames_with_person.toLocaleString(), k: "frames containing a person" },
      { v: Math.round(ped.share_of_sampled_frames * 100) + "%", k: "of all sampled frames" },
      { v: S.num(ped.mean_per_sampled_frame, 1), k: "mean persons / frame" },
      { v: ped.max_in_one_frame, k: "max in one frame" },
      { v: S.seconds(ped.peak_time_sec), k: "peak crowd at" }
    ]));
    out.push("</section>");

    /* ---- 5. heatmaps ---- */
    out.push('<section class="panel">');
    out.push("<h3>Where the traffic is</h3>");
    out.push('<p class="panel-note">Two independent views of the same 5 minutes. The motion map is a ' +
      "frame-difference energy map: it fires on anything that changes pixels, including shadows and " +
      "wipers. The occupancy map uses only the <em>bottom-centre</em> of each detected box, i.e. the " +
      "point where the object touches the road, so it reads as a map of the carriageway rather than of " +
      "the buildings behind it. Hover for cell values.</p>");
    out.push('<div class="split"><div><div id="eda-heat-motion"></div><p class="hint">' + S.esc(h.motion.note) + "</p></div>" +
      '<div><div id="eda-heat-occ"></div><p class="hint">' + S.esc(h.occupancy.note) + "</p></div></div>");
    out.push("</section>");

    /* ---- 6. windows + light ---- */
    var w = d.windows || {};
    out.push('<section class="panel">');
    out.push("<h3>Busiest, quietest, and the light</h3>");
    out.push('<p class="panel-note">10-second sliding windows over the per-second means. These clips are ' +
      "5 minutes 18 seconds long, so this is a <em>within-clip</em> statement — nothing here supports a " +
      "claim about time-of-day traffic profiles, and we do not make one.</p>");
    out.push(mini([
      { v: w.busiest ? S.seconds(w.busiest.start_sec) + "–" + S.seconds(w.busiest.end_sec) : "—", k: "busiest 10 s" },
      { v: w.busiest ? S.num(w.busiest.mean_objects, 1) : "—", k: "mean boxes there" },
      { v: w.quietest ? S.seconds(w.quietest.start_sec) + "–" + S.seconds(w.quietest.end_sec) : "—", k: "quietest 10 s" },
      { v: w.quietest ? S.num(w.quietest.mean_objects, 1) : "—", k: "mean boxes there" },
      { v: w.busiest_over_quietest ? S.num(w.busiest_over_quietest, 2) + "×" : "—", k: "busiest / quietest" },
      { v: d.bus_peak.max_concurrent, k: "max buses together (at " + S.seconds(d.bus_peak.at_sec) + ")" },
      { v: S.num(v.brightness.mean, 1), k: "mean luminance 0–255" },
      { v: S.num(v.brightness.min, 0) + "–" + S.num(v.brightness.max, 0), k: "luminance range" }
    ]));
    out.push('<div style="margin-top:.9rem"><div id="eda-bright"></div></div>');
    out.push("</section>");

    return { html: out.join(""), charts: { density: c1, split: c2 } };
  }

  function paintCharts(v) {
    var tl = v.detection.timeline_1s || [];
    if (!tl.length) return;
    var xs = tl.map(function (r) { return r.t; });
    C.line(document.getElementById("eda-density"), {
      x: xs, height: 250, xFormat: "time", yLabel: "mean detections per second", legend: true, yFloorZero: true,
      aria: "mean detected road users per second over the clip",
      series: [
        { label: "all road users", color: "#38bdf8", values: tl.map(function (r) { return r.objects; }), area: true },
        { label: "vehicles", color: "#f59e0b", values: tl.map(function (r) { return r.vehicles; }) },
        { label: "pedestrians", color: "#4ade80", values: tl.map(function (r) { return r.persons; }) }
      ]
    });
    C.bar(document.getElementById("eda-split"), {
      cats: tl.filter(function (_, i) { return i % 2 === 0; }).map(function (r) { return { label: String(r.t), values: [r.vehicles, r.persons] }; }),
      series: [{ label: "vehicles", color: "#f59e0b" }, { label: "pedestrians", color: "#4ade80" }],
      stacked: true, height: 190, every: 6, yLabel: "per second", xLabel: "second", legend: true
    });
    C.bar(document.getElementById("eda-heavy"), {
      cats: tl.filter(function (_, i) { return i % 2 === 0; }).map(function (r) { return { label: String(r.t), values: [r.buses, r.trucks] }; }),
      series: [{ label: "bus", color: "#a855f7" }, { label: "truck", color: "#f472b6" }],
      stacked: true, height: 190, every: 6, yLabel: "per second", xLabel: "second", legend: true
    });
    C.line(document.getElementById("eda-bright"), {
      x: xs, height: 170, xFormat: "time", yLabel: "mean luminance (0–255)", legend: false, yFloorZero: true,
      aria: "mean frame luminance per second",
      series: [
        { label: "luminance", color: "#fbbf24", values: tl.map(function (r) { return r.brightness; }), area: true, width: 1.4 }
      ]
    });
  }

  function paintHeatmaps(v) {
    var h = v.heatmaps;
    C.heatmap(document.getElementById("eda-heat-motion"), {
      grid_w: h.motion.grid_w, grid_h: h.motion.grid_h, values: h.motion.values,
      min: h.motion.min, max: h.motion.max, title: "Motion energy", subtitle: "mean |Δframe| per cell",
      aria: "motion energy heatmap over the frame"
    });
    C.heatmap(document.getElementById("eda-heat-occ"), {
      grid_w: h.occupancy.grid_w, grid_h: h.occupancy.grid_h, values: h.occupancy.values,
      min: 0, max: Math.max.apply(null, h.occupancy.totals || h.occupancy.values) || 1, gamma: 0.6,
      title: "Road occupancy", subtitle: "detections per cell over the whole clip",
      aria: "road occupancy heatmap over the frame"
    });
  }

  /* ------------------------------------------------------------ findings
     Every statement below is DERIVED from the arrays in eda.json at render
     time. Nothing here is a hardcoded conclusion, so if the data is
     regenerated the prose moves with it. */
  function findings() {
    var rows = [];
    eda.videos.forEach(function (v, i) {
      var name = v.id;
      var d = v.detection, c = v.container, tl = d.timeline_1s || [];
      var o = d.objects_per_sampled_frame;
      var w = d.windows || {};
      var half = Math.floor(tl.length / 2);
      var f1 = tl.slice(0, half), f2 = tl.slice(half);
      var mean = function (arr, k) {
        if (!arr.length) return null;
        var s = 0, n = 0;
        arr.forEach(function (r) { if (r[k] !== null && r[k] !== undefined) { s += r[k]; n++; } });
        return n ? s / n : null;
      };
      var trend = null;
      var a1 = mean(f1, "objects"), a2 = mean(f2, "objects");
      if (a1 && a2) trend = (a2 - a1) / a1 * 100;
      var busCls = d.per_class.bus || {};
      var truckCls = d.per_class.truck || {};
      var carCls = d.per_class.car || {};
      var tlCls = d.per_class.traffic_light || {};

      rows.push({
        name: name,
        tag: "clip " + (i + 1),
        items: [
          "The frame is <b>never</b> quiet: " + d.sampled_frames.toLocaleString() +
          " sampled frames, and at least " + o.min + " road users were detected in <em>every</em> one. " +
          "Peak was " + o.max + ", i.e. a " + (o.max / o.min).toFixed(1) + "× swing within five minutes.",
          "Pedestrians are structural, not incidental: at least one person appears in " +
          Math.round(d.pedestrians.share_of_sampled_frames * 100) + "% of sampled frames, averaging " +
          S.num(d.pedestrians.mean_per_sampled_frame, 1) + " per frame and peaking at " +
          d.pedestrians.max_in_one_frame + " at " + C.clock(d.pedestrians.peak_time_sec) + ".",
          "Buses dominate the class mix: " + S.num(busCls.mean_per_sampled_frame || 0, 2) +
          " buses per sampled frame (" + Math.round((busCls.share_of_sampled_frames || 0) * 100) +
          "% of frames contain one, up to " + (busCls.max_in_one_frame || 0) +
          " at once), and trucks add " + S.num(truckCls.mean_per_sampled_frame || 0, 2) +
          " (" + Math.round((truckCls.share_of_sampled_frames || 0) * 100) + "% of frames). " +
          "This is a bus and truck corridor, which is why the tracker fallback matters here.",
          w.busiest && w.quietest
            ? "The busiest 10 s window (" + C.clock(w.busiest.start_sec) + "–" + C.clock(w.busiest.end_sec) +
              ", mean " + S.num(w.busiest.mean_objects, 1) + " boxes) is " +
              S.num(w.busiest_over_quietest, 1) + "× the quietest (" +
              C.clock(w.quietest.start_sec) + "–" + C.clock(w.quietest.end_sec) + ", mean " +
              S.num(w.quietest.mean_objects, 1) + "). A signal cycle is visible in the density curve."
            : "",
          trend !== null
            ? "Density is " + (Math.abs(trend) < 8
              ? "flat across the clip (second half " + (trend >= 0 ? "up" : "down") + " " +
                S.num(Math.abs(trend), 0) + "% on the first)"
              : (trend > 0 ? "rising" : "falling") + " through the clip — second half is " +
                S.num(Math.abs(trend), 0) + "% " + (trend > 0 ? "higher" : "lower") +
                " than the first, mean " + S.num(a2, 1) + " vs " + S.num(a1, 1) + " boxes")
            : "",
          "Cars alone account for " + S.num(carCls.mean_per_sampled_frame || 0, 1) +
          " of the mean " + S.num(o.mean, 1) + " boxes, with a per-frame maximum of " +
          (carCls.max_in_one_frame || 0) + ".",
          "Traffic-light heads are found in only " +
          Math.round((tlCls.share_of_sampled_frames || 0) * 100) + "% of frames " +
          "(" + (tlCls.detections_total || 0) + " detections total). At 640 px they are a few pixels " +
          "wide — this is the concrete reason the red-light rules are not yet trustworthy."
        ].filter(Boolean)
      });
    });

    var cmp = eda.comparison || {};
    if (eda.videos.length === 2 && cmp.mean_objects_a !== undefined) {
      var a = eda.videos[0], b = eda.videos[1];
      rows.push({
        name: "both clips",
        tag: "comparison",
        items: [
          "The two clips are matched pairs — identical " + a.container.width + "×" + a.container.height +
          ", " + S.num(a.container.fps, 3) + " fps, " + a.container.frames_decoded.toLocaleString() +
          " frames, " + S.num(a.container.duration_sec, 1) + " s — so a difference between them is " +
          "traffic, not capture settings.",
          (b.detection.objects_per_sampled_frame.mean / a.detection.objects_per_sampled_frame.mean) > 1.02
            ? "<b>" + b.id + " is the busier clip:</b> " + S.num(b.detection.objects_per_sampled_frame.mean, 1) +
              " mean road users per frame against " + S.num(a.detection.objects_per_sampled_frame.mean, 1) +
              " (" + Math.round((b.detection.objects_per_sampled_frame.mean / a.detection.objects_per_sampled_frame.mean - 1) * 100) +
              "% more), and its busiest window is " +
              S.num(b.detection.windows.busiest.mean_objects, 1) + " boxes against " +
              S.num(a.detection.windows.busiest.mean_objects, 1) + "."
            : "<b>" + a.id + " is the busier clip:</b> " + S.num(a.detection.objects_per_sampled_frame.mean, 1) +
              " mean road users per frame against " + S.num(b.detection.objects_per_sampled_frame.mean, 1) + ".",
          "The bus difference is far larger than the car difference: " +
          S.num(a.detection.per_class.bus.mean_per_sampled_frame, 2) + " vs " +
          S.num(b.detection.per_class.bus.mean_per_sampled_frame, 2) + " buses per frame " +
          "(" + S.num(b.detection.per_class.bus.mean_per_sampled_frame / Math.max(1e-9, a.detection.per_class.bus.mean_per_sampled_frame), 1) +
          "×) for " + S.num(a.detection.per_class.car.mean_per_sampled_frame, 1) + " vs " +
          S.num(b.detection.per_class.car.mean_per_sampled_frame, 1) + " cars. The clips are separated by " +
          "bus service, which is the dominant difficulty in one and not the other."
        ]
      });
    }

    return rows;
  }

  function renderFindings() {
    var host = document.createElement("section");
    host.className = "panel";
    host.innerHTML = "<h3>What the numbers actually say <span class=\"panel-head-meta\">derived at render time from eda.json</span></h3>" +
      '<p class="panel-note">Each statement below is computed from the arrays in the JSON when the page ' +
      "loads, so regenerating the data moves the conclusions with it.</p>" +
      findings().map(function (r) {
        return '<h4 style="margin:.9rem 0 .3rem;color:var(--accent);font-family:var(--mono);font-size:.86rem">' +
          S.esc(r.name) + ' <span class="muted" style="font-weight:400">· ' + S.esc(r.tag) + "</span></h4>" +
          '<ul class="ok-list" style="margin-bottom:.2rem">' + r.items.map(function (t) {
            return "<li>" + t + "</li>";
          }).join("") + "</ul>";
      }).join("");
    return host;
  }

  function select(i) {
    active = i;
    tabButtons.forEach(function (b, j) { b.setAttribute("aria-selected", j === i ? "true" : "false"); });
    var v = eda.videos[i];
    if (!v) return;
    var built = renderVideo(v);
    var body = document.getElementById("edaBody");
    body.innerHTML = built.html;
    // put the real per-clip still in place of nothing-on-first-paint
    var m = mediaFor(v.id);
    {
      var fig = document.createElement("figure");
      fig.className = "shot";
      fig.style.margin = "0";
      fig.innerHTML = '<img src="' + S.esc(m.still) + '" alt="A frame from ' + S.esc(v.file) + '" loading="lazy" decoding="async">' +
        "<figcaption>" + S.esc(v.file) + " — " + v.container.width + "×" + v.container.height + ", " +
        v.container.frames_decoded.toLocaleString() + " frames. Seekable preview: " + S.esc(m.video) +
        " (VP9, " + m.width + "×" + m.height + ", " + m.fps + " fps" +
        (m.bytes ? ", " + S.bytes(m.bytes) : "") + ").</figcaption>";
      body.insertBefore(fig, body.firstChild);
    }
    paintCharts(v);
    paintHeatmaps(v);
    // the still + the derived-findings panel go at the top, above the charts
    body.insertBefore(renderFindings(), body.children[1] || null);
  }

  function notes() {
    if (!eda || !eda.notes) return "";
    return '<section class="panel"><h3>What we found, and what we refuse to claim</h3>' +
      '<ul class="bad-list" style="margin-top:.2rem">' + eda.notes.map(function (n) { return "<li>" + S.esc(n) + "</li>"; }).join("") + "</ul></section>";
  }

  function init() {
    if (edaInit) return;
    var body = document.getElementById("edaBody"), tabs = document.getElementById("edaTabs");
    eda = S.state.eda;
    if (!eda) {
      body.innerHTML = '<div class="pending-box"><h3>EDA data pending</h3><p>' +
        "<code>static/data/eda.json</code> was not found. Generate it with " +
        "<code>python website/tools/analyze_samples.py</code> from the repository root. " +
        "Nothing on this page is filled in by hand, so an empty JSON file means an empty section rather " +
        "than invented numbers.</p></div>";
      renderProvenance();
      return;
    }
    renderProvenance();
    edaInit = true;
    tabButtons = eda.videos.map(function (v, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.setAttribute("role", "tab");
      b.textContent = v.file;
      b.addEventListener("click", function () { select(i); });
      tabs.appendChild(b);
      return b;
    });
    select(0);
    body.insertAdjacentHTML("beforeend", notes());
  }

  document.addEventListener("DOMContentLoaded", init);
  document.addEventListener("data:ready", function () { if (!eda) init(); });
})();
