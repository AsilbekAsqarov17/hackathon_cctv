/* =====================================================================
   results.js — renders static/data/results.json.

   The interactive bit: each event segment on the timeline is a button.
   Clicking it seeks the preview <video> to the segment start, so a judge
   can check a claim against the footage in two clicks. The same timeline
   is reused for demo results, where the video is the uploaded blob URL.
   ===================================================================== */
(function () {
  "use strict";
  var S = window.Site, C = window.Charts;
  var res = null, idx = 0;

  var LABEL_COLORS = {
    accident: "#f87171", near_miss: "#fb923c", red_light: "#e879f9", wrong_way: "#fbbf24",
    illegal_u_turn: "#2dd4bf", stopped_vehicle: "#a3e635", jaywalking: "#4ade80",
    failure_to_yield: "#c084fc", illegal_turn: "#fdba74", solid_line_crossing: "#93c5fd",
    stop_line: "#f472b6", congestion: "#818cf8", road_obstacle: "#22d3ee", fire_smoke: "#f87171"
  };
  var evId = 0;

  /* -------------------------------------------------- interactive timeline */
  function buildTimeline(host, events, duration, onSeek) {
    var labels = [];
    events.forEach(function (e) { if (labels.indexOf(e.label) < 0) labels.push(e.label); });
    labels.sort();
    var wrap = document.createElement("div");
    wrap.className = "tl-wrap";
    if (!events.length) {
      wrap.innerHTML = '<div class="tl-empty">No segments were produced for this clip. That is a real ' +
        "result, not a rendering failure — see the note under the table.</div>";
      host.appendChild(wrap);
      return;
    }
    var rows = document.createElement("div");
    rows.className = "tl-rows";
    labels.forEach(function (lab) {
      var row = document.createElement("div");
      row.className = "tl-row";
      var name = document.createElement("div");
      name.className = "tl-name";
      name.textContent = lab;
      name.title = lab;
      var track = document.createElement("div");
      track.className = "tl-track";
      events.filter(function (e) { return e.label === lab; }).forEach(function (e) {
        var seg = document.createElement("button");
        seg.type = "button";
        seg.className = "tl-seg";
        seg.style.left = (100 * e.start / duration) + "%";
        seg.style.width = Math.max(0.35, 100 * (e.end - e.start) / duration) + "%";
        seg.style.setProperty("--c", LABEL_COLORS[e.label] || "#38bdf8");
        seg.title = e.label + "  " + e.start.toFixed(2) + "–" + e.end.toFixed(2) + " s  (" + (e.end - e.start).toFixed(2) + " s)";
        seg.setAttribute("aria-label", seg.title);
        seg.addEventListener("click", function () { onSeek(e, seg); });
        track.appendChild(seg);
      });
      row.appendChild(name); row.appendChild(track);
      rows.appendChild(row);
    });
    wrap.appendChild(rows);
    var ax = document.createElement("div");
    ax.className = "tl-axis";
    for (var t = 0; t <= duration; t += Math.max(1, Math.round(duration / 8))) {
      var s = document.createElement("span");
      s.textContent = C.clock(t);
      ax.appendChild(s);
    }
    wrap.appendChild(ax);
    host.appendChild(wrap);
  }

  function eventTable(events, duration) {
    if (!events.length) return "";
    var rows = events.map(function (e, i) {
      return "<tr><td class=\"num\">" + (i + 1) + "</td><td>" + e.label + "</td><td class=\"num\">" +
        e.start.toFixed(2) + "</td><td class=\"num\">" + e.end.toFixed(2) + "</td><td class=\"num\">" +
        (e.end - e.start).toFixed(2) + "</td><td class=\"num\">" +
        (100 * (e.end - e.start) / duration).toFixed(1) + "%</td></tr>";
    }).join("");
    return '<div class="table-scroll" style="margin-top:.9rem"><table class="tbl"><thead><tr>' +
      "<th class=\"num\">#</th><th>Label</th><th class=\"num\">Start s</th><th class=\"num\">End s</th>" +
      "<th class=\"num\">Duration s</th><th class=\"num\">Share of clip</th></tr></thead><tbody>" + rows + "</tbody></table></div>";
  }

  function riskChart(host, risk, events, duration) {
    if (!risk || !risk.curve || !risk.curve.length) {
      host.innerHTML = '<div class="alert alert-warn">No risk curve in results.json — run the script with Part B enabled.</div>';
      return;
    }
    var pts = risk.curve;
    C.line(host, {
      x: pts.map(function (p) { return p[0]; }),
      height: 260, xFormat: "time", yLabel: "risk score (0–1)", yFloorZero: true, min: 0, max: 1,
      aria: "per-frame accident risk score over the clip",
      rules: [{ value: risk.threshold || 0.5, label: "alarm θ = " + (risk.threshold || 0.5), color: "#f59e0b" }],
      markers: events.filter(function (e) { return e.label === "accident"; }).map(function (e) {
        return { x: e.start, label: "accident", color: "#f87171" };
      }),
      series: [{ label: "risk score", color: "#a855f7", area: true, width: 1.5, values: pts.map(function (p) { return p[1]; }) }]
    });
  }

  /* ------------------------------------------------------------ one video */
  function renderVideo(v, mediaUrl) {
    var c = v.container, rt = v.runtime || {}, risk = v.risk || {};
    var frames = c.frames !== undefined ? c.frames : c.frames_decoded;
    var wrap = document.createElement("div");
    wrap.className = "report";

    /* video + timeline */
    var media = document.createElement("section");
    media.className = "panel";
    media.innerHTML = "<h3>" + S.esc(v.file) + " <span class=\"panel-head-meta\">" + c.width + "×" + c.height +
      " · " + (frames === undefined ? "?" : frames.toLocaleString()) + " frames · " + S.num(c.duration_sec, 1) + " s</span></h3>";
    var vw = document.createElement("div");
    vw.className = "split";
    var vcell = document.createElement("div");
    if (mediaUrl) {
      vcell.innerHTML = '<div class="video-shell"><video id="ev-video" controls preload="metadata" playsinline ' +
        'src="' + S.esc(mediaUrl) + '"></video></div>' +
        '<p class="hint">Low-bitrate preview (' + S.esc(mediaUrl.split("/").pop()) + '). Click any coloured ' +
        "segment below to jump straight to it.</p>";
    } else {
      vcell.innerHTML = '<div class="alert alert-warn">No preview clip for this video. Encode one with ' +
        "<code>ffmpeg -i &lt;clip&gt; -an -vf scale=480:-2,fps=15 -c:v libvpx-vp9 -b:v 300k out.webm</code>.</div>";
    }
    var tcell = document.createElement("div");
    tcell.innerHTML = '<p class="panel-note" style="margin-top:0">Event timeline — one row per class, ' +
      'time on the horizontal axis. <strong>Click a segment to seek the video.</strong></p><div id="tl-host"></div>' +
      '<ul class="ev-list" id="ev-list"></ul>';
    vw.appendChild(vcell); vw.appendChild(tcell);
    media.appendChild(vw);
    if (v.event_counts_by_label && Object.keys(v.event_counts_by_label).length) {
      var cbox = document.createElement("div");
      cbox.style.marginTop = ".9rem";
      cbox.innerHTML = '<h4 style="font-size:.86rem;color:var(--fg-dim);margin-bottom:.3rem">Segments per class ' +
        "(raw predictions — there is no ground truth for these clips, so this is a count, not a score)</h4>" +
        '<div id="res-counts"></div>';
      media.appendChild(cbox);
    }
    wrap.appendChild(media);

    /* risk */
    var rp = document.createElement("section");
    rp.className = "panel";
    rp.innerHTML = "<h3>Causal accident-risk curve <span class=\"panel-head-meta\">" +
      (risk.points_total ? risk.points_total.toLocaleString() + " points, one per frame" : "no curve") +
      "</span></h3>" +
      '<p class="panel-note">Every point is what <code>RiskEstimator.step()</code> returned for that frame, ' +
      "replaying only feature samples whose timestamp is ≤ the current frame. The dashed line is the " +
      "official alarm threshold; alarms are runs at or above it, merged when closer than " +
      (risk.merge_gap_sec || 2) + "&nbsp;s apart — the same function the metric itself uses.</p>" +
      '<div id="risk-chart"></div>' +
      '<div class="statgrid" style="margin-top:.8rem">' +
      mini({ v: S.num(risk.mean, 3), k: "mean score" }) +
      mini({ v: S.num(risk.max, 3), k: "max score" }) +
      mini({ v: S.num(risk.median, 3), k: "median score" }) +
      mini({ v: Math.round((risk.share_at_or_above_threshold || 0) * 100) + "%", k: "frames ≥ θ" }) +
      mini({ v: String(risk.alarm_count === undefined ? "—" : risk.alarm_count), k: "alarms" }) +
      mini({ v: S.seconds(risk.longest_run_above_threshold_sec), k: "longest alarm run" }) +
      "</div>";
    wrap.appendChild(rp);

    /* facts */
    var sc = v.scene || {};
    var others = sc.other_scene_files_with_geometry || [];
    var hasLines = (sc.lanes || 0) > 0 && (sc.stop_lines || 0) > 0;
    var withLanes = others.filter(function (o) { return (o.lanes || 0) > 0; });
    var dur = c.duration_sec || 0;
    var total = (rt.part_a_sec || 0) + (rt.part_b_sec || 0);
    var budget = 3 * dur;
    var overBudget = total > budget;
    var facts = document.createElement("section");
    facts.className = "panel";
    facts.innerHTML = "<h3>Run facts</h3>" +
      '<div class="statgrid">' +
      mini({ v: S.num(rt.part_a_sec, 0) + " s", k: "Part A wall clock" }) +
      mini({ v: S.num(rt.part_b_sec, 0) + " s", k: "Part B wall clock" }) +
      mini({ v: S.num(total, 0) + " s", k: "combined" }) +
      mini({ v: dur ? S.num(total / dur, 2) + "×" : "—", k: "combined / clip length" }) +
      mini({ v: dur ? S.num(budget, 0) + " s" : "—", k: "harness budget (3×)" }) +
      mini({ v: String((v.events || []).length), k: "segments produced" }) +
      "</div>" +
      '<p class="panel-note" style="margin-top:.8rem"><strong>Part B is not free.</strong> It builds its ' +
      "own detector, tracker and rule stack rather than reusing Part A's state, so the detector runs " +
      "<em>twice</em> per clip. On this clip Part B costs " +
      (rt.part_a_sec && rt.part_b_sec
        ? Math.round((rt.part_b_sec / rt.part_a_sec) * 100) + "% of what Part A costs"
        : "a comparable amount") +
      ", sampling the detector every " + 5 + " frames and repeating the previous score in between. " +
      (overBudget
        ? '<span class="badge-no">over the 3× budget</span> On this machine the combined run would be ' +
          "cut off by the harness. On the evaluation GPU it would not, but we are not going to pretend the " +
          "CPU number is comfortable."
        : '<span class="badge-yes">inside the 3× harness budget</span> with ' +
          (budget && total ? Math.round((1 - total / budget) * 100) + "% headroom" : "headroom") +
          " — but the margin is CPU-only and a larger detector would erase it.") +
      "</p>" +
      '<p class="panel-note" style="margin-top:.6rem">Scene config used: <code>' + S.esc(sc.path || "none") + "</code>" +
      " — scene_id <code>" + S.esc(sc.scene_id || "—") + "</code>, " + (sc.lanes || 0) + " lanes, " +
      (sc.stop_lines || 0) + " stop lines, " + (sc.crosswalks || 0) + " crosswalks, " +
      (sc.traffic_lights || 0) + " traffic-light ROIs, " + (sc.road_polygons || 0) + " road polygons" +
      (sc.has_homography ? ", with homography" : ", no homography") +
      (sc.selected_via ? ", selected via <code>" + S.esc(sc.selected_via) + "</code>" : "") + ". Lookup order: " +
      (sc.lookup_order || []).map(function (p) { return "<code>" + S.esc(p) + "</code>"; }).join(" → ") + "." +
      (sc.per_camera_calibrated
        ? " This clip has its own calibrated geometry file."
        : (hasLines
          ? " The selected file carries lane and stop-line geometry, so the line- and signal-based " +
            "classes are live for this clip."
          : " <strong>No lane or stop-line geometry is selected for this clip</strong>, so " +
            "<code>red_light</code>, <code>stop_line</code>, <code>wrong_way</code>, " +
            "<code>illegal_turn</code> and <code>solid_line_crossing</code> cannot fire. The pipeline " +
            "refuses to guess rather than emit false positives.")) +
      (withLanes.length
        ? " <br><strong>Lane geometry that exists but is not selected:</strong> " +
          withLanes.map(function (o) {
            return "<code>" + S.esc(o.path) + "</code> (" + (o.lanes || 0) + " lanes, " +
              (o.stop_lines || 0) + " stop lines, " + (o.crosswalks || 0) + " crosswalks)";
          }).join(", ") + " — hand-calibrated against a different development video, so transferring it " +
          "here is calibration work rather than a config change."
        : "") + "</p>" +
      '<p class="panel-note">Format check against the official rules: ' +
      (v.validation && v.validation.same_class_overlaps === 0 ? "<span class=\"badge-yes\">0 same-class overlaps</span>" :
        '<span class="badge-no">' + ((v.validation || {}).same_class_overlaps || "?") + " same-class overlaps</span>") +
      " " + (v.validation && v.validation.all_intervals_valid ? "<span class=\"badge-yes\">all intervals valid</span>" :
        '<span class="badge-no">interval error</span>') + " " +
      (v.validation && !(v.validation.labels_outside_official_list || []).length
        ? "<span class=\"badge-yes\">all labels official</span>"
        : '<span class="badge-no">non-official label present</span>') +
      " · alarms extracted with <code>evaluate.alarm_starts</code>, the metric's own function" +
      (v.code_revision ? " · code revision <code>" + S.esc(v.code_revision) + "</code>" : "") +
      (v.config_sha ? ", config sha <code>" + S.esc(v.config_sha) + "</code>" : "") + ".</p>";
    if (v.warnings && v.warnings.length) {
      facts.innerHTML += '<p class="panel-note"><span class="badge-part">runtime warning</span> ' +
        v.warnings.map(S.esc).join(" · ") + "</p>";
    }
    wrap.appendChild(facts);

    /* annotated frames */
    if (v.annotated_frames && v.annotated_frames.length) {
      var fp = document.createElement("section");
      fp.className = "panel";
      fp.innerHTML = "<h3>Annotated frames <span class=\"panel-head-meta\">" +
        v.annotated_frames.length + " frames, drawn by the same detector</span></h3>" +
        '<p class="panel-note">Each thumbnail is a real frame from the clip with every detection drawn. ' +
        "Click one to seek the video to that timestamp. The banner carries the timestamp, the risk score and " +
        "the detection count at that instant.</p>" +
        '<div class="thumbs" id="thumbs"></div>';
      wrap.appendChild(fp);
      var thumbs = fp.querySelector("#thumbs");
      v.annotated_frames.forEach(function (f) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "thumb";
        b.innerHTML = '<img src="' + S.esc(f.src) + '" alt="Annotated frame at ' + S.num(f.t, 2) +
          ' seconds" loading="lazy" decoding="async"><span class="cap">t=' + S.num(f.t, 2) + " s · risk " +
          S.num(f.risk, 2) + " · " + f.detections + " det" +
          (f.active_labels && f.active_labels.length ? "<br>" + f.active_labels.join(", ") : "") + "</span>";
        b.addEventListener("click", function () { seekVideo(f.t); });
        thumbs.appendChild(b);
      });
    }

    function seekVideo(t) {
      var el = document.getElementById("ev-video");
      if (!el) return;
      try {
        el.currentTime = Math.max(0, t);
        el.play().catch(function () { /* autoplay policy — the seek still happened */ });
      } catch (e) { /* ignore */ }
    }

    /* now paint the timeline + risk chart (needs the DOM in place) */
    return {
      node: wrap,
      paint: function () {
        var dur = c.duration_sec || 1;
        buildTimeline(document.getElementById("tl-host"), v.events || [], dur, function (e) { seekVideo(e.start); });
        var list = document.getElementById("ev-list");
        if (list) {
          (v.events || []).slice(0, 400).forEach(function (e) {
            var li = document.createElement("li");
            var b = document.createElement("button");
            b.type = "button";
            b.innerHTML = '<span class="ev-l">' + e.label + "</span><span class=\"ev-t\">" +
              e.start.toFixed(2) + "–" + e.end.toFixed(2) + " s</span><span class=\"ev-d\">" +
              (e.end - e.start).toFixed(2) + " s long · click to seek</span>";
            b.addEventListener("click", function () { seekVideo(e.start); });
            li.appendChild(b); list.appendChild(li);
          });
        }
        riskChart(document.getElementById("risk-chart"), risk, v.events || [], dur);
        var countsHost = document.getElementById("res-counts");
        if (countsHost && v.event_counts_by_label) {
          var labels = Object.keys(v.event_counts_by_label).sort(function (a, b) {
            return v.event_counts_by_label[b] - v.event_counts_by_label[a];
          });
          C.bar(countsHost, {
            cats: labels.map(function (l) { return { label: l, values: [v.event_counts_by_label[l]] }; }),
            series: [{ label: "segments", color: "#38bdf8" }],
            height: 210, every: 1, yLabel: "segments", xLabel: "class",
            aria: "number of predicted segments per class"
          });
        }
        var tl = document.getElementById("tl-host");
        if (tl) tl.insertAdjacentHTML("beforeend", eventTable(v.events || [], dur));
      }
    };
  }

  /* ------------------------------------- scored against the team's own labels
     Rendered only when results.json carries a "scored" block, i.e. a labels
     file was found. The wording is deliberately unflattering: these are our
     own annotations, they cover one clip, and on the clip certified to
     contain no events every prediction is, by construction, a false positive. */
  function scoredPanel(res) {
    var s = res.scored;
    if (!s || s.error) return null;
    var host = document.createElement("section");
    host.className = "panel";
    var rows = (s.per_class || []).map(function (r) {
      var c5 = r["tp_fp_fn@0.5"] || [0, 0, 0];
      return "<tr><td>" + r.label + '</td><td class="num">' + S.num(r["f1@0.3"], 3) +
        '</td><td class="num">' + S.num(r["f1@0.5"], 3) + '</td><td class="num">' +
        S.num(r["f1@0.7"], 3) + '</td><td class="num">' + S.num(r.f1_mean, 3) +
        '</td><td class="num">' + c5.join(" / ") + "</td></tr>";
    }).join("");
    host.innerHTML = "<h3>Scored against our own annotations <span class=\"panel-head-meta\">" +
      S.esc(s.computed_by || "") + "</span></h3>" +
      '<div class="statgrid">' +
      mini({ v: S.num(s.score_a, 3), k: "Score_A on the labelled clip" }) +
      mini({ v: S.num(s.micro["f1@0.5"], 3), k: "micro F1 @ tIoU 0.5" }) +
      mini({ v: S.num(s.micro["f1@0.7"], 3), k: "micro F1 @ tIoU 0.7" }) +
      mini({ v: s.part_b ? S.num(s.part_b.score_b, 3) : "n/a", k: "Score_B" }) +
      mini({ v: s.part_b ? S.num(s.part_b.ap, 3) : "n/a", k: "chance-normalised AP" }) +
      mini({ v: S.num(s.model_score, 3), k: "model score M" }) +
      "</div>" +
      '<p class="panel-note" style="margin-top:.8rem"><strong>Read this before anything else.</strong> ' +
      "These are <em>our own</em> labels, not the organizers&rsquo;. They cover " +
      (s.labelled_videos || []).length + " of " + (res.videos || []).length + " clips" +
      ((s.unlabelled_videos || []).length
        ? " &mdash; " + s.labelled_videos.join(", ") + " labelled, " + s.unlabelled_videos.join(", ") + " not"
        : "") +
      ". The labelled clip was reviewed and certified to contain <strong>no reportable event of any " +
      "class</strong>, so it is a negative sample: under macro-F1 a false positive costs exactly as much " +
      "as a missed event, and every segment the pipeline predicts on it is, by construction, a false " +
      "positive. <strong>Score_A on it is therefore a direct false-positive measurement, and it is " +
      S.num(s.score_a, 3) + ".</strong> That is the most informative number we have, and it is not flattering.</p>" +
      (rows ? '<div class="table-scroll" style="margin-top:.7rem"><table class="tbl"><thead><tr>' +
        '<th>Class</th><th class="num">F1@0.3</th><th class="num">F1@0.5</th><th class="num">F1@0.7</th>' +
        '<th class="num">mean</th><th class="num">TP/FP/FN@0.5</th></tr></thead><tbody>' +
        rows + "</tbody></table></div>" : "") +
      (s.caveats && s.caveats.length
        ? '<p class="panel-note" style="margin-top:.8rem"><strong>Caveats, quoted from the labels file:</strong></p>' +
          '<ul class="bad-list" style="margin-top:.2rem">' + s.caveats.map(function (c) {
            return "<li>" + S.esc(c) + "</li>";
          }).join("") + "</ul>"
        : "") +
      '<p class="panel-note" style="margin-top:.6rem">Labels live in <code>' + S.esc(s.labels_file) +
      "</code>. A class that appears in our predictions but not in the ground truth is scored as pure " +
      "false positive by the official matcher &mdash; which is the intended behaviour, and a large part " +
      "of why the score is low." +
      (s.part_b
        ? ""
        : " <strong>Part B is reported as <code>n/a</code> because the labels contain no " +
          "<code>accident</code> events</strong> &mdash; and the official metric returns " +
          "<code>M = Score_A</code> outright when a test set has no accidents. The risk curve is still " +
          "produced and shown below, but on this clip there is nothing to score it against.") +
      "</p>";
    return host;
  }

  function mini(o) {
    return '<div class="mini"><span class="v">' + o.v + '</span><span class="k">' + S.esc(o.k) + "</span></div>";
  }

  /* ------------------------------------------------------------- render */
  function mediaUrlFor(id) {
    if (res.media) {
      for (var i = 0; i < res.media.length; i++) if (res.media[i].id === id) return res.media[i].video;
    }
    /* results.json does not carry a media manifest, so fall back to the
       conventional name written by website/tools/analyze_samples.py */
    return "static/media/" + id + "_preview.webm";
  }

  function render() {
    var body = document.getElementById("resBody"), prov = document.getElementById("resProv");
    res = S.state.results;
    if (!res) {
      prov.innerHTML = "<span><b>status</b> results.json not found</span>";
      body.innerHTML = '<div class="pending-box"><h3><span class="badge-pending">pending</span> No measured output yet</h3>' +
        "<p><code>static/data/results.json</code> has not been generated. It is written by " +
        "<code>python website/tools/run_samples.py</code>, which calls <code>solution.detect_events()</code> " +
        "and <code>solution.RiskEstimator</code> on the sample clips. Part A runs at roughly " +
        "<span data-from=\"results\" data-stat=\"rt_factor_A\">1.5×</span> real time on CPU, so budget " +
        "about five minutes per five-minute clip.</p>" +
        "<p>Until that file exists this section stays empty on purpose. Everything below it — the timeline " +
        "renderer, the risk chart, the click-to-seek behaviour — is already implemented and will populate " +
        "the moment the file is dropped in. Re-running the script is the only step needed; no HTML changes.</p>" +
        "<p class=\"hint\">To see it working right now, use the <a href=\"#demo\">live demo</a> above with a " +
        "short clip.</p></div>";
      return;
    }
    var d = res.detector || {}, ar = res.alarm_rule || {};
    var revs = (res.videos || []).map(function (v) { return v.code_revision; })
      .filter(function (r, i, a) { return r && a.indexOf(r) === i; });
    var fps = (res.videos || []).map(function (v) { return v.code_fingerprint; })
      .filter(function (r, i, a) { return r && a.indexOf(r) === i; });
    prov.innerHTML = S.provenance([
      "<b>source</b> static/data/results.json",
      "<b>generated</b> " + res.generated_utc,
      "repo " + (res.repo_commit || "?") + (res.repo_dirty ? " (dirty tree)" : ""),
      "<b>code fingerprint</b> " + (res.code_fingerprint || "?") +
        (fps.length > 1 ? " (clips: " + fps.join(", ") + ")" : ""),
      "<b>per-clip revision</b> " + (revs.length ? revs.join(", ") : "?"),
      "<b>entry points</b> " + (res.entry_points || []).join(", "),
      "<b>detector</b> " + (d.model || "?") + (d.model_present ? "" : " <b>MISSING</b>") + " · " + (d.provider || "?"),
      "<b>config</b> " + (res.config || "?"),
      "<b>alarm rule</b> θ=" + (ar.threshold === undefined ? "?" : ar.threshold) + ", merge " + (ar.merge_gap_sec === undefined ? "?" : ar.merge_gap_sec) + " s"
    ]) + (res.partial
      ? '<div class="alert alert-warn" style="width:100%;margin-top:.4rem">This file is <b>partial</b> — ' +
        (res.videos || []).length + " of " + (res.expected_videos || (res.videos || []).length) +
        " clips finished. Re-run <code>python website/tools/run_samples.py</code> (add " +
        "<code>--resume</code> to keep what is already here).</div>"
      : "") + (res.metadata_stale
        ? '<div style="width:100%;margin-top:.4rem;font-size:.78rem;color:var(--fg-mute)">'
          + "<b>Provenance note.</b> The events and the risk curve below are the unedited output of "
          + "one run, stamped with fingerprint <code>" + (((res.videos || [])[0] || {}).code_fingerprint)
          + "</code>. The scene selection, detector availability and metric score beside them were "
          + "refreshed later from a newer tree (" + ((res.tree_at_refresh || {}).code_fingerprint)
          + "), so re-running <code>python website/tools/run_samples.py</code> against the current code "
          + "would give a slightly different event list. The numbers themselves are real.</div>"
        : (res.warning
          ? '<div class="alert alert-warn" style="width:100%;margin-top:.4rem">' + S.esc(res.warning) + "</div>"
          : ""));

    var tabs = document.getElementById("resTabs");
    if (!tabs) {
      tabs = document.createElement("div");
      tabs.className = "tabs";
      tabs.id = "resTabs";
      tabs.setAttribute("role", "tablist");
      tabs.setAttribute("aria-label", "Sample clip");
      body.parentNode.insertBefore(tabs, body);
    }
    tabs.innerHTML = "";
    body.innerHTML = "";
    res.videos.forEach(function (v, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.setAttribute("role", "tab");
      b.textContent = v.file;
      b.setAttribute("aria-selected", i === 0 ? "true" : "false");
      b.addEventListener("click", function () { select(i); });
      tabs.appendChild(b);
    });
    (res.notes || []).forEach(function (n, i) {
      body.insertAdjacentHTML("beforeend", '<p class="fineprint">' + (i + 1) + ". " + S.esc(n) + "</p>");
    });
    select(0);
  }

  function select(i) {
    idx = i;
    var tabs = document.getElementById("resTabs");
    if (tabs) [].slice.call(tabs.children).forEach(function (b, j) { b.setAttribute("aria-selected", j === i ? "true" : "false"); });
    var v = res.videos[i];
    var body = document.getElementById("resBody");
    var notes = [].slice.call(body.querySelectorAll("p.fineprint"));
    body.innerHTML = "";
    var built = renderVideo(v, mediaUrlFor(v.id));
    body.appendChild(built.node);
    built.paint();
    notes.forEach(function (n) { body.appendChild(n); });
    var sp = scoredPanel(res);
    if (sp) body.insertBefore(sp, body.firstChild);
  }

  document.addEventListener("DOMContentLoaded", render);
  document.addEventListener("data:ready", function () { if (!res) render(); });
})();
