/* =====================================================================
   demo.js — the live-demo client.

   Protocol (all stdlib-HTTP, no framework on either side):
     GET  /api/health            -> detector + limit status
     POST /api/jobs              -> {job_id}   (raw request body = the file)
     GET  /api/jobs/<id>         -> {state, stage, progress, result?, error?}
   The page degrades instead of breaking: if the API is not reachable
   (static hosting) the dropzone explains why and stays disabled.
   ===================================================================== */
(function () {
  "use strict";
  var S = window.Site, C = window.Charts;

  var LIMITS = { seconds: 120, bytes: 150 * 1024 * 1024 };
  var file = null, jobId = null, pollTimer = null, objectUrl = null, apiOk = null;

  var $ = function (id) { return document.getElementById(id); };
  var dz = $("dropzone"), input = $("fileInput"), runBtn = $("runBtn"),
    cancelBtn = $("cancelBtn"), progress = $("progress"), bar = $("progressBar"),
    stage = $("progressStage"), pct = $("progressPct"), hint = $("progressHint"),
    alertBox = $("demoAlert"), out = $("demoOut");

  function setAlert(kind, html) {
    if (!html) { alertBox.hidden = true; alertBox.className = "alert"; alertBox.innerHTML = ""; return; }
    alertBox.hidden = false;
    alertBox.className = "alert alert-" + kind;
    alertBox.innerHTML = html;
  }

  function setProgress(on, label, frac, note) {
    progress.hidden = !on;
    if (!on) return;
    stage.textContent = label;
    var pctVal = Math.max(0, Math.min(100, Math.round(frac * 100)));
    bar.style.width = pctVal + "%";
    pct.textContent = pctVal + " %";
    if (note !== undefined) hint.textContent = note;
  }

  /* ------------------------------------------------------------- health */
  function checkHealth() {
    return fetch("api/health", { cache: "no-cache" })
      .then(function (r) { return r.json(); })
      .then(function (h) {
        apiOk = true;
        var bits = [];
        if (h.limits) {
          if (h.limits.max_seconds) { LIMITS.seconds = h.limits.max_seconds; $("limitDur").textContent = h.limits.max_seconds + " s"; }
          if (h.limits.max_bytes) { LIMITS.bytes = h.limits.max_bytes; $("limitSize").textContent = Math.round(h.limits.max_bytes / 1048576) + " MB"; }
        }
        if (h.detector) {
          bits.push("detector <b>" + S.esc(h.detector.model || "?") + "</b> on <b>" +
            S.esc(h.detector.provider || "no provider") + "</b>");
          if (h.detector.error) {
            bits.push('<span class="badge-no">unavailable</span>');
            setAlert("warn", "<b>The detector could not be loaded.</b> " + S.esc(h.detector.error) +
              " The backend will still run and still report the real video metadata, but with no " +
              "detector there are no tracks, so the event list will be empty and the risk curve flat. " +
              "That is a degraded run, reported as such — not a crash.");
          } else if (h.detector.present === false) {
            setAlert("warn",
              "<b>Detector weights are not present.</b> The backend will still run and still report the " +
              "real video metadata, but with no detector there are no tracks, so the event list will be " +
              "empty and the risk curve flat. That is a degraded run, reported as such — not a crash. " +
              "Place <code>weights/yolo11n.onnx</code> in the repository and restart.");
          }
        }
        if (h.recent_part_a_realtime_factor) {
          $("rtEstimate").textContent = String(h.recent_part_a_realtime_factor);
          $("rtUnit").textContent = "real time on this host";
        } else {
          $("rtEstimate").textContent = "1.5";
          $("rtUnit").textContent = "real time on a CPU";
        }
        hint.innerHTML = "Backend: " + bits.join(" · ") +
          (h.python ? " · Python " + S.esc(h.python) : "") +
          (h.repo_root ? "" : " · repo not found");
        runBtn.disabled = !file;
        return h;
      })
      .catch(function (err) {
        apiOk = false;
        dz.classList.remove("hot");
        setAlert("warn",
          "<b>No demo backend on this host.</b> This page is a static site and the upload needs the small " +
          "Python server that lives next to it. To enable the demo, run " +
          "<code>python website/demo/app.py</code> and open <code>http://127.0.0.1:8000</code>. " +
          "Everything else on this page works as-is when hosted statically " +
          "(GitHub&nbsp;Pages, Netlify, Vercel). <span class=\"tiny muted\">[" + S.esc(err.message) + "]</span>");
        runBtn.disabled = true;
        return null;
      });
  }

  /* -------------------------------------------------------------- input */
  function humanSize(b) { return S.bytes(b); }

  function accept(f) {
    file = f;
    setAlert(null);
    if (!f) { runBtn.disabled = true; return; }
    var type = (f.type || "").toLowerCase();
    var extOK = /\.(mp4|webm|mov|m4v|avi)$/i.test(f.name);
    if (!extOK && type.indexOf("video/") !== 0) {
      setAlert("err", "<b>That is not a video.</b> Accepted: <code>.mp4</code>, <code>.webm</code>, " +
        "<code>.mov</code>, <code>.m4v</code>, <code>.avi</code> (the browser must be able to decode it).");
      file = null; runBtn.disabled = true; return;
    }
    if (f.size > LIMITS.bytes) {
      setAlert("err", "<b>" + humanSize(f.size) + " is over the " + humanSize(LIMITS.bytes) +
        " limit.</b> Trim the clip first — for example " +
        "<code>ffmpeg -i in.mp4 -t 120 -c copy out.mp4</code> keeps the stream copy and takes seconds.");
      file = null; runBtn.disabled = true; return;
    }
    var warn = "";
    // duration check is advisory before upload; the server is authoritative
    var probe = document.createElement("video");
    probe.preload = "metadata";
    var url = URL.createObjectURL(f);
    probe.onloadedmetadata = function () {
      if (probe.duration && probe.duration > LIMITS.seconds + 0.5) {
        setAlert("err", "<b>" + probe.duration.toFixed(1) + " s is over the " + LIMITS.seconds +
          " s limit.</b> Trim it to " + LIMITS.seconds + " s: <code>ffmpeg -i in.mp4 -t " + LIMITS.seconds +
          " -c copy out.mp4</code>.");
        file = null; runBtn.disabled = true;
        URL.revokeObjectURL(url);
        return;
      }
      showFileInfo(f, probe, warn);
      URL.revokeObjectURL(url);
    };
    probe.onerror = function () {
      showFileInfo(f, null, warn);
      URL.revokeObjectURL(url);
    };
    probe.src = url;
    runBtn.disabled = apiOk === false;
  }

  function showFileInfo(f, probe, warn) {
    var dur = probe ? probe.duration : null;
    var w = probe ? probe.videoWidth : null, h = probe ? probe.videoHeight : null;
    out.innerHTML = '<div class="statgrid">' +
      mini(humanSize(f.size), "file size") +
      mini(dur ? C.clock(dur) : "—", "duration" + (dur && dur > LIMITS.seconds ? " (over limit)" : "")) +
      mini(w ? w + "×" + h : "—", "resolution") +
      mini("ready", "status") + "</div>" +
      '<p class="hint">' + S.esc(f.name) + (warn ? " " + warn : "") + "</p>" +
      '<p class="hint">Press <b>Run analysis</b>. The file is sent to this host\'s own Python process — ' +
      "it is not uploaded anywhere else.</p>";
  }

  function mini(v, k) {
    return '<div class="mini"><span class="v">' + v + '</span><span class="k">' + S.esc(k) + "</span></div>";
  }

  /* ---------------------------------------------------------------- job */
  function start() {
    if (!file || !apiOk) return;
    setAlert(null);
    runBtn.disabled = true;
    cancelBtn.hidden = false;
    out.innerHTML = '<div class="loading">Uploading…</div>';
    setProgress(true, "uploading", 0.02, "0 of " + humanSize(file.size));
    var fd = new FormData();
    fd.append("file", file, file.name);
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "api/jobs");
    xhr.upload.onprogress = function (e) {
      if (e.lengthComputable) {
        setProgress(true, "uploading", 0.02 + 0.13 * (e.loaded / e.total),
          humanSize(e.loaded) + " of " + humanSize(e.total));
      }
    };
    xhr.onload = function () {
      cancelBtn.hidden = true;
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          var j = JSON.parse(xhr.responseText);
          jobId = j.job_id;
          setProgress(true, "queued", 0.15, "job " + jobId);
          poll();
        } catch (e) { fail("Backend sent a reply this page cannot parse."); }
      } else {
        var msg = xhr.responseText;
        try { msg = JSON.parse(xhr.responseText).error || msg; } catch (e) { }
        fail("<b>HTTP " + xhr.status + "</b> " + S.esc(String(msg).slice(0, 300)));
      }
    };
    xhr.onerror = function () { cancelBtn.hidden = true; fail("Network error while uploading."); };
    xhr.onabort = function () { cancelBtn.hidden = true; setProgress(false); out.innerHTML = '<div class="loading">Cancelled.</div>'; };
    xhr.send(fd);
  }

  function fail(msg) {
    runBtn.disabled = false;
    setProgress(false);
    setAlert("err", msg);
  }

  function poll() {
    if (!jobId) return;
    clearTimeout(pollTimer);
    pollTimer = setTimeout(function () {
      fetch("api/jobs/" + encodeURIComponent(jobId), { cache: "no-cache" })
        .then(function (r) { return r.json(); })
        .then(function (j) {
          var p = 0.15 + 0.8 * (typeof j.progress === "number" ? j.progress : 0);
          setProgress(true, j.stage || j.state || "working", p, j.detail || "");
          if (j.state === "done") {
            setProgress(false);
            runBtn.disabled = false;
            renderResult(j.result || {});
          } else if (j.state === "error") {
            setProgress(false);
            runBtn.disabled = false;
            setAlert("err", "<b>Job failed.</b> " + S.esc(j.error || "no message") +
              (j.degraded ? " The run continued in degraded mode." : ""));
            if (j.result) renderResult(j.result);
          } else if (j.state === "cancelled") {
            setProgress(false);
            runBtn.disabled = false;
            out.innerHTML = '<div class="loading">Cancelled.</div>';
          } else {
            poll();
          }
        })
        .catch(function (err) { fail("Lost contact with the backend: " + S.esc(err.message)); });
    }, 700);
  }

  function cancel() {
    if (jobId) {
      fetch("api/jobs/" + encodeURIComponent(jobId) + "/cancel", { method: "POST" }).catch(function () { });
    }
    clearTimeout(pollTimer);
    setProgress(false);
    cancelBtn.hidden = true;
    runBtn.disabled = false;
  }

  /* ------------------------------------------------------------- result */
  function renderResult(r) {
    var events = r.events || [];
    var risk = r.risk || {};
    var frames = r.annotated_frames || [];
    var cont = r.container || {};
    var dur = cont.duration_sec || (risk.points ? risk.points[risk.points.length - 1][0] : 1);

    var h = [];
    if (r.degraded) {
      h.push('<div class="alert alert-warn"><b>Degraded run.</b> ' + S.esc(r.degraded_reason || "part of the pipeline was unavailable") + "</div>");
    }
    h.push('<div class="statgrid">' +
      mini(S.esc(cont.width || "?") + "×" + S.esc(cont.height || "?"), "resolution") +
      mini((cont.frames || "?").toLocaleString ? (cont.frames || 0).toLocaleString() : cont.frames, "frames") +
      mini(S.num(cont.duration_sec, 1) + " s", "duration") +
      mini(String(events.length), "event segments") +
      mini(S.num((r.runtime || {}).part_a_sec, 1) + " s", "Part A") +
      mini(S.num((r.runtime || {}).part_b_sec, 1) + " s", "Part B") +
      "</div>");

    if (r.scene) {
      h.push('<p class="hint">Scene config: <code>' + S.esc(r.scene.path || "none") + "</code>" +
        (r.scene.via ? " (via <code>" + S.esc(r.scene.via) + "</code>)" : "") +
        " — <code>" + S.esc(r.scene.scene_id || "—") + "</code>, " + (r.scene.lanes || 0) + " lanes, " +
        (r.scene.stop_lines || 0) + " stop lines, " + (r.scene.crosswalks || 0) + " crosswalks, " +
        (r.scene.traffic_lights || 0) + " light ROIs" +
        ((r.scene.lanes || 0) === 0 && (r.scene.stop_lines || 0) === 0
          ? " — <b>no lane or stop-line geometry, so red_light, stop_line, wrong_way, " +
            "illegal_turn and solid_line_crossing cannot fire on any clip</b>"
          : (r.scene.per_camera_calibrated ? " — <b>per-camera calibrated</b>" : "")) +
        (r.scene.lookup_order
          ? "<br>lookup order: " + r.scene.lookup_order.map(function (p) {
              return "<code>" + S.esc(p) + "</code>";
            }).join(" → ")
          : "") +
        "</p>");
    }
    if ((r.counts || {}) && Object.keys(r.counts).length) {
      h.push('<ul class="filelist">' + Object.keys(r.counts).sort().map(function (k) {
        return "<li><b>" + k + "</b><span>" + r.counts[k] + "</span></li>";
      }).join("") + "</ul>");
    }

    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    h.push('<div class="video-shell" style="margin-top:.8rem"><video id="demo-video" controls playsinline preload="metadata" src="' + objectUrl + '"></video></div>');

    h.push('<p class="panel-note" style="margin-top:1rem">Event timeline — <strong>click a segment to seek the video.</strong></p>');
    h.push('<div id="demo-tl"></div>');
    h.push('<div id="demo-risk" style="margin-top:1rem"></div>');
    if (frames.length) {
      h.push('<p class="panel-note" style="margin-top:1rem">Annotated frames from your clip — click to seek.</p><div class="thumbs" id="demo-thumbs"></div>');
    }
    if ((r.warnings || []).length) {
      h.push('<p class="panel-note" style="margin-top:.8rem"><span class="badge-part">runtime warning</span> ' +
        r.warnings.map(S.esc).join(" · ") + "</p>");
    }
    out.innerHTML = h.join("");

    /* timeline */
    buildTL(events, dur);
    /* risk */
    if (risk.points && risk.points.length) {
      C.line($("demo-risk"), {
        x: risk.points.map(function (p) { return p[0]; }),
        y: risk.points.map(function (p) { return p[1]; }),
        height: 210, xFormat: "time", yLabel: "risk score (0–1)", yFloorZero: true, min: 0, max: 1,
        aria: "per-frame risk score for the uploaded clip",
        rules: [{ value: risk.threshold || 0.5, label: "alarm θ = " + (risk.threshold || 0.5), color: "#f59e0b" }],
        markers: events.filter(function (e) { return e.label === "accident"; })
          .map(function (e) { return { x: e.start, label: "accident", color: "#f87171" }; }),
        series: [{ label: "risk", color: "#a855f7", area: true, width: 1.5 }]
      });
      var rm = $("demo-risk").parentNode;
      var rs = document.createElement("div");
      rs.className = "statgrid";
      rs.style.marginTop = ".7rem";
      rs.innerHTML =
        mini(S.num(risk.mean, 3), "mean") + mini(S.num(risk.max, 3), "max") +
        mini(Math.round((risk.share_at_or_above_threshold || 0) * 100) + "%", "frames ≥ θ") +
        mini(String(risk.alarm_count === undefined ? "—" : risk.alarm_count), "alarms");
      if (rm) rm.appendChild(rs);
    }
    /* thumbs */
    var th = $("demo-thumbs");
    if (th) {
      frames.forEach(function (f) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "thumb";
        b.innerHTML = '<img src="' + f.src + '" alt="Annotated frame at ' + S.num(f.t, 2) + ' s">' +
          '<span class="cap">t=' + S.num(f.t, 2) + " s · " + f.detections + " det</span>";
        b.addEventListener("click", function () { seek(f.t); });
        th.appendChild(b);
      });
    }
    function seek(t) {
      var v = $("demo-video");
      if (!v) return;
      try { v.currentTime = Math.max(0, t); v.play().catch(function () { }); } catch (e) { }
    }
  }

  var LABEL_COLORS = {
    accident: "#f87171", near_miss: "#fb923c", red_light: "#e879f9", wrong_way: "#fbbf24",
    illegal_u_turn: "#2dd4bf", stopped_vehicle: "#a3e635", jaywalking: "#4ade80",
    failure_to_yield: "#c084fc", illegal_turn: "#fdba74", solid_line_crossing: "#93c5fd",
    stop_line: "#f472b6", congestion: "#818cf8", road_obstacle: "#22d3ee", fire_smoke: "#f87171"
  };

  function buildTL(events, duration) {
    var host = $("demo-tl");
    if (!host) return;
    if (!events.length) {
      host.innerHTML = '<div class="tl-wrap"><div class="tl-empty">No segments. With an uncalibrated scene ' +
        "only the road-fallback rules can fire, and on a quiet clip they may not fire at all. " +
        "An empty list is a valid answer.</div></div>";
      return;
    }
    var labels = [];
    events.forEach(function (e) { if (labels.indexOf(e.label) < 0) labels.push(e.label); });
    labels.sort();
    var rows = '<div class="tl-rows">';
    labels.forEach(function (lab) {
      rows += '<div class="tl-row"><div class="tl-name">' + S.esc(lab) + '</div><div class="tl-track">';
      events.filter(function (e) { return e.label === lab; }).forEach(function (e) {
        rows += '<button type="button" class="tl-seg" data-t="' + e.start + '" title="' + S.esc(lab) + " " +
          e.start.toFixed(2) + "–" + e.end.toFixed(2) + ' s" style="left:' +
          (100 * e.start / duration) + "%;width:" + Math.max(0.35, 100 * (e.end - e.start) / duration) +
          '%;--c:' + (LABEL_COLORS[lab] || "#38bdf8") + '"></button>';
      });
      rows += "</div></div>";
    });
    rows += '</div><div class="tl-axis">';
    for (var t = 0; t <= duration; t += Math.max(1, Math.round(duration / 6))) {
      rows += "<span>" + C.clock(t) + "</span>";
    }
    rows += "</div>";
    host.innerHTML = '<div class="tl-wrap">' + rows + "</div>";
    [].slice.call(host.querySelectorAll(".tl-seg")).forEach(function (b) {
      b.addEventListener("click", function () {
        var v = $("demo-video");
        if (!v) return;
        try { v.currentTime = parseFloat(b.getAttribute("data-t")); v.play().catch(function () { }); } catch (e) { }
      });
    });
  }

  /* --------------------------------------------------------------- wire */
  function init() {
    if (!dz) return;
    ["dragenter", "dragover"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add("hot"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove("hot"); });
    });
    dz.addEventListener("drop", function (e) {
      var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) accept(f);
    });
    dz.addEventListener("click", function () { input.click(); });
    dz.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
    });
    input.addEventListener("change", function () { accept(input.files && input.files[0]); });
    runBtn.addEventListener("click", start);
    cancelBtn.addEventListener("click", cancel);
    checkHealth();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
