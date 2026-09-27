/* =====================================================================
   report.js — the write-up. Prose is ours; every number inside it is
   pulled from eda.json / results.json so the report cannot drift away
   from the measurements.
   ===================================================================== */
(function () {
  "use strict";
  var S = window.Site, C = window.Charts;

  function li(list) {
    // Accepts either varargs of strings or a single array. Call sites build a
    // list with push() and then hand it over, so both shapes must work -- and
    // both must yield one <li> per entry, not one <li> holding the lot.
    var items = [];
    for (var i = 1; i < arguments.length; i++) {
      var a = arguments[i];
      if (Object.prototype.toString.call(a) === "[object Array]") items = items.concat(a);
      else items.push(a);
    }
    return '<ul class="' + list + '">' + items.map(function (s) {
      return "<li>" + s + "</li>";
    }).join("") + "</ul>";
  }

  function panel(title, body, meta) {
    return '<section class="panel"><h3>' + title +
      (meta ? ' <span class="panel-head-meta">' + meta + "</span>" : "") + "</h3>" + body + "</section>";
  }

  function build(eda, res) {
    var out = [];
    var v = eda && eda.videos && eda.videos[0];
    var rv = res && res.videos && res.videos[0];
    var det = (res && res.detector) || {};

    /* ---------------- what we built ---------------- */
    out.push(panel("What we built", li("ok-list",
      "<b>Two causal passes over the video, two outputs.</b> <code>solution.detect_events()</code> walks " +
      "the clip at a detector stride of " + (det.stride || "?") + " and emits " +
      "<code>[start, end, label]</code> segments; <code>solution.RiskEstimator</code> then walks the clip " +
      "again at its own, coarser stride and emits a risk score per frame. The two passes are independent, " +
      "so Part B's score cannot be contaminated by the event output it is supposed to anticipate.",
      "<b>A detector that fits the offline host.</b> YOLO11n exported to ONNX (" +
      (det.model_bytes ? S.bytes(det.model_bytes) : "?") + ", " + (det.imgsz || "?") + " px, conf " +
      (det.confidence || "?") + "), run through " + (det.provider || "?") +
      ". No PyTorch, no CUDA, no network access at inference time.",
      "<b>Fourteen rules over one canonical state object.</b> Every rule consumes the same " +
      "<code>FrameState</code> and returns a <code>RuleSignal</code>; none of them opens the video or runs " +
      "a model, which is what makes them individually testable and individually disable-able. The " +
      "<code>TrackManager</code> is the only owner of track state, so there is no second, divergent " +
      "history to disagree with.",
      "<b>A scene-calibration seam.</b> Lanes, stop lines, solid markings, crosswalks and traffic-light ROIs " +
      "live in a JSON file per camera. When a camera has no file, the geometry rules are <em>off</em> " +
      "instead of guessing — a deliberate precision/recall trade we would make again.",
      "<b>A static website whose numbers are generated, not typed.</b> " +
      "Two Python scripts produce <code>eda.json</code> and <code>results.json</code>; this page renders " +
      "them and has no hard-coded measurements to drift out of date."
    ), "design + engineering effort, not a score"));

    /* ---------------- what worked ---------------- */
    var worked = [];
    if (rv) {
      worked.push("<b>The format is valid.</b> On " + S.esc(rv.file) + " the pipeline returned " +
        (rv.events || []).length + " segments with " + ((rv.validation || {}).same_class_overlaps || 0) +
        " same-class overlaps and " + (((rv.validation || {}).all_intervals_valid) ? "no" : "some") +
        " out-of-range intervals. " +
        "<code>python evaluate.py --pred … --validate-only</code> accepts this shape.");
    }
    if (v) {
      worked.push("<b>Detection quality is good enough to build on.</b> Mean " +
        S.num(v.detection.objects_per_sampled_frame.mean, 1) + " road users per sampled frame, " +
        (v.detection.per_class.car ? v.detection.per_class.car.detections_total.toLocaleString() + " cars and " : "") +
        (v.detection.per_class.person ? v.detection.per_class.person.detections_total.toLocaleString() + " people" : "") +
        " over the clip at stride " + v.detection.stride + ", on a 640 px input covering a 1080p frame. " +
        "That is enough for trajectory-level rules.");
    }
    if (rv && rv.risk && rv.risk.points_total) {
      worked.push("<b>Causality holds.</b> The risk curve has " + rv.risk.points_total.toLocaleString() +
        " points — one per frame, each computed only from that frame and its own causal runtime state. " +
        "Between detector samples the previous score is repeated verbatim, which the task explicitly " +
        "permits; it is a hold, not a lookahead.");
    }
    worked.push("<b>The combined run fits the harness budget, but only just.</b> The harness allows " +
      "3× the clip duration for Part A + Part B together. Part B builds its own detector/tracker/rule " +
      "stack rather than reusing Part A's state — deliberately, so Part B is provably independent of the " +
      "event output — which means the detector runs twice. On CPU the combined figure is " +
      (rv ? S.num((((rv.runtime || {}).part_a_sec || 0) + ((rv.runtime || {}).part_b_sec || 0)) /
        Math.max(1e-6, (rv.container || {}).duration_sec || 1), 2) + "× real time" : "—") +
      ". That is inside the budget here, but it is a CPU number on 1080p dev copies; on the real 4K " +
      "inputs with a GPU it is the part of the system we would profile first.");
    worked.push("<b>Rule thresholds are scale-free.</b> Every distance, speed and acceleration threshold " +
      "is expressed in metres, m/s and m/s² rather than pixels, so the same configuration means the same " +
      "thing on a different camera or a different resolution. A pixel threshold on a 4K feed is a hidden " +
      "hyper-parameter of the resolution, and this way it is not.");
    worked.push("<b>It fails soft, on purpose.</b> <code>detect_events</code> catches per-frame detector and " +
      "tracker failures and keeps going, and the rule engine isolates each rule in its own " +
      "<code>try</code> — a rule that throws loses its own signal for that frame, not the whole run. We " +
      "exercised that path for real while building this site: during a transient upstream fault the " +
      "<a href=\"#demo\">live demo</a> returned a complete, honest result with the warning text attached " +
      "instead of a 500.");
    out.push(panel("What worked", li("ok-list", worked)));

    /* ---------------- what did not ---------------- */
    var failed = [];
    if (rv && rv.risk && typeof rv.risk.share_at_or_above_threshold === "number") {
      var sat = (rv.risk.share_at_or_above_threshold * 100).toFixed(1);
      failed.push("<b>The risk score saturates, and it is the single worst thing about the system.</b> " +
        "On " + S.esc(rv.file) + " the mean score is " + S.num(rv.risk.mean, 3) + " and <b>" + sat +
        "% of all frames sit at or above the 0.5 alarm threshold</b>, for a longest unbroken alarm run of " +
        S.seconds(rv.risk.longest_run_above_threshold_sec) + " — effectively the whole clip. " +
        (res.videos[1] && res.videos[1].risk
          ? "The other clip tells the same story: " + S.num(res.videos[1].risk.mean, 3) + " mean, " +
            Math.round(res.videos[1].risk.share_at_or_above_threshold * 100) + "% above threshold, " +
            res.videos[1].risk.alarm_count + " alarm. "
          : "") +
        "The metric is built to punish exactly this: chance-normalised AP gives a constant score 0, and an " +
        "alarm that is always on matches nothing usefully. As configured, Part B would score close to " +
        "<b>0</b> no matter how good Part A is. The cause is visible in <code>src/part_b.py</code>: the " +
        "score is a <code>max()</code> over several weak terms, smoothed with a 0.30 EMA whose decay is " +
        "0.85 — so nearly any frame with a nearby vehicle parks the score at 1.0 and the EMA then refuses " +
        "to let it fall. In a scene averaging " +
        (v ? S.num(v.detection.objects_per_sampled_frame.mean, 0) : "20+") +
        " road users per frame there is <em>always</em> a nearby vehicle. The fix is calibration, not a new " +
        "model: fit a logistic head on the existing <code>RiskFeatures</code> contract against a labelled " +
        "split, and gate the pedestrian-conflict term so a person walking 5 m from a car on the pavement " +
        "is not a 0.65.");
    }
    var acc = rv && (rv.events || []).filter(function (e) { return e.label === "accident"; }).length;
    var sc = (res && res.scored) || null;
    if (sc && !sc.error) {
      failed.push("<b>Scored against our own annotations, Part A is currently " +
        S.num(sc.score_a, 3) + ".</b> We reviewed a clip and certified it to contain <em>no reportable " +
        "event of any class</em>, which makes it a negative sample and turns the score into a direct " +
        "false-positive measurement. The pipeline still puts " +
        (res.videos.length
          ? res.videos.reduce(function (a, v2) { return a + (v2.events || []).length; }, 0) +
            " segments across the pair"
          : "segments") +
        ", so the false-positive count is high. These numbers come from the organizers' own " +
        "<code>evaluate.evaluate()</code>, run unmodified — we did not write a scorer to make this look " +
        "better or worse than it is. The honest reading is that the rule set is tuned for recall and has " +
        "no precision discipline yet; the per-class table in the results section shows which rules are " +
        "responsible.");
    } else {
      failed.push("<b>No accuracy claim is possible, because there is no ground truth.</b> Nothing is " +
        "labelled for either sample clip. We have pseudo-labels for a <em>different</em> video " +
        "(<code>data_video1</code>), and the repository's own README is explicit that they must be " +
        "reviewed before being used for a reported result. So the " + (acc || 0) +
        " <code>accident</code> segments on " + (rv ? S.esc(rv.file) : "the clip") +
        " are <em>predictions we do not endorse</em>. They are shown because the timeline component " +
        "needs real data to be real, and labelled raw precisely so nobody mistakes them for an F1.");
    }
    var other = (rv && rv.scene && rv.scene.other_scene_files_with_geometry) || [];
    var scene = (rv && rv.scene) || {};
    var hasLines = (scene.lanes || 0) > 0 && (scene.stop_lines || 0) > 0;
    if (rv && rv.scene && !hasLines) {
      var withLanes = other.filter(function (o) { return (o.lanes || 0) > 0; });
      failed.push("<b>Five of the fourteen classes are inert on the sample clips.</b> " +
        "<code>red_light</code>, <code>stop_line</code>, <code>wrong_way</code>, " +
        "<code>illegal_turn</code> and <code>solid_line_crossing</code> all need lane and stop-line " +
        "geometry. The file the pipeline actually selects is <code>" + S.esc(scene.path || "none") + "</code>" +
        (scene.selected_via ? " (via <code>" + S.esc(scene.selected_via) + "</code>)" : "") +
        ", scene_id <code>" + S.esc(scene.scene_id || "?") + "</code>, with " + (scene.lanes || 0) +
        " lanes, " + (scene.stop_lines || 0) + " stop lines, " + (scene.solid_lines || 0) + " solid lines" +
        (withLanes.length
          ? ". A fuller geometry does exist — <code>" +
            withLanes.map(function (o) { return S.esc(o.path); }).join("</code>, <code>") +
            "</code> — but it was hand-calibrated against a <em>different</em> development video, so " +
            "carrying it over to these two clips is a calibration task, not a config flip."
          : ", and no lane geometry exists for these camera ids yet.") +
        " Nothing is silently guessed: the rules disable themselves. But <em>correctly silent</em> still " +
        "scores zero on all five.");
    }
    failed.push("<b>Three classes are stubs that cannot work with a COCO detector.</b> " +
      "<code>road_obstacle</code>, <code>fire_smoke</code> and <code>illegal_u_turn</code> are substring " +
      "matches on the track's class name (<code>&quot;fire&quot;</code>, <code>&quot;debris&quot;</code>, …). " +
      "COCO has no such categories, so they can never activate; <code>illegal_u_turn</code> is in fact " +
      "<code>enabled: false</code> in <code>configs/default.json</code>. This is a deliberate trade, not an " +
      "oversight: the metric averages F1 over the classes appearing in the ground truth <em>or</em> in our " +
      "predictions, so enabling a class that can never fire would add a hard zero to the denominator. They " +
      "are code, not capability, and we will not count them as solved.");
    failed.push("<b>The upstream ByteTrack is not installed.</b> <code>third_party/byte_track</code> is " +
      "absent and its dependencies (<code>lap</code>, <code>cython_bbox</code>, <code>scipy</code>) have no " +
      "manylinux wheels, so <code>build_tracker</code> falls back to the built-in two-stage IoU tracker. " +
      "That tracker matches on IoU alone, with no motion model, so it cannot re-acquire a vehicle briefly " +
      "occluded by a bus — and bus occlusion is this scene's dominant failure mode. Both runs logged the " +
      "fallback as a warning, visible in the per-video output rather than buried in a log nobody reads.");
    var rois = (scene.traffic_light_rois) || [];
    var roiSmall = scene.smallest_roi_px_at_640;
    failed.push("<b>Reading a red light is the weakest link in the chain.</b> " +
      "<code>TrafficLightReader</code> classifies each signal-head ROI in HSV, and every " +
      "<code>red_light</code> and <code>stop_line</code> decision rests on that. The configured ROIs are " +
      (rois.length
        ? rois.map(function (r) {
            return "<code>" + S.esc(r.id) + "</code> " + S.num(r.w_px_1080p, 0) + "&times;" +
              S.num(r.h_px_1080p, 0) + " px at 1080p";
          }).join(", ") + ", which is only " + S.num(roiSmall, 0) + " px on the short side once the frame " +
          "is scaled to the detector's 640 px input"
        : "not exposed in the scene block we emit") +
      ". One lamp inside that is a handful of pixels, so a brake light in the same box reads as amber and " +
      "an amber phase reads as red. It is the first thing we would validate properly, and the only way to " +
      "do it is to check the reader against a hand-read signal phase — which is why the labels file " +
      "carries one.");
    var svSec = (rv && rv.event_seconds_by_label && rv.event_seconds_by_label.stopped_vehicle) || 0;
    var clipSec = (rv && rv.container && rv.container.duration_sec) || 0;
    var svShare = clipSec ? svSec / clipSec : 0;
    var svCount = (rv && rv.event_counts_by_label && rv.event_counts_by_label.stopped_vehicle) || 0;
    if (svShare > 0.5) {
      failed.push("<b><code>stopped_vehicle</code> covers " + Math.round(svShare * 100) +
        "% of the clip.</b> On " + (rv ? S.esc(rv.file) : "one clip") + " the " + svCount +
        " <code>stopped_vehicle</code> segments span " + S.num(svSec, 0) + " s of a " + S.num(clipSec, 0) +
        " s clip. The rule is meant to exclude vehicles queued at a stop line, and at this intersection " +
        "that is most of what stands still. It is a direct, measurable consequence of the geometry " +
        "situation above, and the clearest single argument for finishing the calibration.");
    } else if (svSec > 0) {
      worked.push("<b>The queue exclusion works.</b> <code>stopped_vehicle</code> covers " +
        Math.round(svShare * 100) + "% of " + (rv ? S.esc(rv.file) : "the clip") + " (" +
        S.num(svSec, 0) + " s of " + S.num(clipSec, 0) + " s) across " + svCount +
        "segments — vehicles genuinely standing still are separated from vehicles queued at a signal.");
    } else {
      worked.push("<b>The queue exclusion works, and it changed the answer.</b> " +
        "<code>stopped_vehicle</code> now fires <em>zero times</em> on " +
        (rv ? S.esc(rv.file) : "the clip") + ". Before the queue logic was added it covered the " +
        "entire clip on both samples: with no stop-line geometry, every car waiting at the lights " +
        "counted as an illegal stop. It is the clearest example on this page of why geometry is " +
        "not a detail — and of why you need labels to notice.");
    }
    failed.push("<b>The temporal segmenter is unvalidated against the metric.</b> We have never seen a tIoU " +
      "curve, so the minimum durations, the per-class maximum durations and the 1 s merge gap in " +
      "<code>configs/default.json</code> are reasoned guesses. F1 at tIoU 0.7 is unforgiving about " +
      "boundaries and we currently have no way to tune them.");
    failed.push("<b>Detector fine-tuning is stalled on review.</b> The pseudo-label dataset exists but is " +
      "unreviewed, and we will not report a number from unreviewed labels. The EDA shows why a fine-tune " +
      "is worth doing at all: traffic-light heads appear in only " +
      (v && v.detection.per_class.traffic_light
        ? Math.round(v.detection.per_class.traffic_light.share_of_sampled_frames * 100) + "%"
        : "a few percent") +
      " of sampled frames, and bicycles and motorcycles are effectively absent — that is the confidence " +
      "floor talking, not the scene.");
    out.push(panel("What did <p style=\"color:var(--bad)\">not</p> work", li("bad-list", failed)));

    /* ---------------- next ---------------- */
    out.push(panel("What we would do next, in order", li("ok-list",
      "<b>1. Label 20 minutes of one clip by hand, twice, and measure inter-annotator agreement.</b> " +
      "Everything else is downstream of this. The README's own advice is that without a dev set you are " +
      "guessing, and that is currently true of us.",
      "<b>2. Calibrate the risk scorer against those labels.</b> Keep the causal <code>RiskFeatures</code> " +
      "contract exactly as it is and replace <code>_score_features</code> with a logistic regression on " +
      "the same nine features, fitted to P(collision within 5 s). The feature boundary is already right; " +
      "only the decision function is wrong.",
      "<b>3. Calibrate the two sample cameras.</b> Lane polygons from tracked trajectories, then stop lines " +
      "only after the approach direction is confirmed, then signal ROIs from zoomed crops with the HSV " +
      "state checked against the clip. This is what unlocks five classes and is the highest-value hour in " +
      "the project.",
      "<b>4. Make Part B cheaper.</b> It re-detects the whole clip. Re-introducing a causal feature " +
      "cache between the passes would roughly halve the combined runtime, at the cost of coupling Part B " +
      "to Part A's track history. That is a real trade and not obviously worth taking — but the runtime " +
      "should be profiled before anyone concludes it cannot be.",
      "<b>5. Replace the tracker fallback with a real ByteTrack</b>, or add a constant-velocity Kalman " +
      "predictor to the existing one so tracks survive bus occlusion.",
      "<b>6. Then, and only then, fine-tune the detector</b> on reviewed pseudo-labels and measure " +
      "the delta on the same fixed split.",
      "<b>7. Then consider a learned temporal head</b> (DSTA/DoTA-style) for Part B. The notes in " +
      "<code>research/PART_B_NOTES.md</code> are right that they are dashcam-domain and would need " +
      "adaptation, and that this should replace the scorer rather than the causal feature contract."
    )));

    /* ---------------- how the score works ---------------- */
    out.push('<section class="panel"><h3>How we are scored, and what that implies for the design</h3>' +
      '<div class="two-col"><div>' +
      "<p><strong>Part A</strong> — for each class and each tIoU threshold in {0.3, 0.5, 0.7}, greedy " +
      "one-to-one matching by descending temporal IoU. TP/FP/FN are pooled over <em>all</em> videos, then " +
      "<code>F1_c(τ)</code>. <code>Score_A</code> is the mean over classes of the mean over thresholds. " +
      "The class list is whatever appears in the ground truth <em>or</em> in our predictions — so a class " +
      "we predict that never occurs scores a hard 0, and a class we never predict costs nothing. That " +
      "asymmetry is why we would rather not enable the three substring-match classes: they can only ever " +
      "add a zero to the denominator.</p>" +
      "<p>Practical consequence: <strong>segment boundaries matter as much as segment existence</strong>. " +
      "A correct 9-second event labelled 8.5–17.0 when the truth is 9.0–18.0 has tIoU ≈ 0.65 — a true " +
      "positive at 0.5 and 0.3, a false positive <em>and</em> a false negative at 0.7. Tightening the " +
      "edges is worth as much as finding more events.</p>" +
      "</div><div>" +
      "<p><strong>Part B</strong> — <code>accident</code> only, H = 5 s, W = 10 s, θ = 0.5. Frames in " +
      "[s−5, s) before an accident are positive; frames inside an accident or inside [s−5, e] of a " +
      "near-miss are <em>ignored</em>; everything else is negative.</p>" +
      "<ul style=\"font-size:.89rem;color:var(--fg-dim);padding-left:1.1rem;margin:.4rem 0\">" +
      "<li><b>AP</b> — average precision over frames, chance-normalised to " +
      "<code>max(0, (AP_raw − r)/(1 − r))</code> where r is the positive rate. A constant or random score " +
      "scores exactly 0. This is the term our saturated scorer fails.</li>" +
      "<li><b>F1_alarm</b> — alarms are maximal runs at or above 0.5, runs closer than 2 s apart merged, " +
      "alarm time = run start. An alarm in [s−10, s) of a still-unmatched accident matches it, earliest " +
      "alarm first. Alarms that begin on ignored frames are thrown away.</li>" +
      "<li><b>mTTA</b> — mean of (accident start − matched alarm start), 0 when unmatched.</li>" +
      "</ul>" +
      "<p><code>Score_B = 0.4·AP + 0.4·F1_alarm + 0.2·mTTA/10</code> and " +
      "<code>M = 0.7·Score_A + 0.3·Score_B</code>. Note what the ignored window does: a near-miss is a " +
      "penalty-free zone. A scorer that fires hard on near-misses does not merely fail to gain there, it " +
      "loses the chance to gain on the accident five seconds later. That is an argument for suppressing " +
      "short-lived peaks, which EMA smoothing already does — the problem is that our EMA is far too " +
      "sticky in the other direction.</p>" +
      "</div></div></section>");

    /* ---------------- metric honesty panel ---------------- */
    if (rv && rv.risk) {
      var r2 = res.videos[1] && res.videos[1].risk;
      var rows = res.videos.map(function (v2) {
        var k = v2.risk || {};
        return "<tr><td>" + S.esc(v2.file) + "</td><td class=\"num\">" + S.num(k.mean, 3) + "</td><td class=\"num\">" +
          S.num(k.median, 3) + "</td><td class=\"num\">" + S.num(k.max, 3) + "</td><td class=\"num\">" +
          (k.share_at_or_above_threshold === undefined ? "—" : Math.round(k.share_at_or_above_threshold * 100) + "%") +
          "</td><td class=\"num\">" + (k.alarm_count === undefined ? "—" : k.alarm_count) + "</td><td class=\"num\">" +
          S.seconds(k.longest_run_above_threshold_sec) + "</td></tr>";
      }).join("");
      out.push('<section class="panel"><h3>Part B measured, side by side <span class="panel-head-meta">' +
        "one row per clip; the labels contain no accident, so these curves are unscored</span></h3>" +
        '<div class="table-scroll"><table class="tbl"><thead><tr><th>Clip</th><th class="num">mean</th>' +
        "<th class=\"num\">median</th><th class=\"num\">max</th><th class=\"num\">frames ≥ 0.5</th>" +
        "<th class=\"num\">alarms</th><th class=\"num\">longest run</th></tr></thead><tbody>" + rows +
        "</tbody></table></div>" +
        '<p class="panel-note">A calibrated Part B would show a mean near the base rate of accidents (close to ' +
        "0 on most frames) and a short list of isolated spikes. This shows a curve that is off almost the " +
        "whole time, which is the signature of a scorer that has not been calibrated rather than one that " +
        "has learned something.</p></section>");
      void r2;
    }

    return out.join("");
  }

  function init() {
    var host = document.getElementById("reportBody");
    if (!host) return;
    var eda = S.state.eda, res = S.state.results;
    if (!eda && !res) {
      // nothing measured yet: show the pending explanation but stay unclaimed,
      // so a later data:eda / data:results event can still fill it in
      if (host.getAttribute("data-state") !== "pending") {
        host.setAttribute("data-state", "pending");
        host.innerHTML = '<div class="pending-box"><h3>Report pending</h3><p>Neither ' +
          "<code>eda.json</code> nor <code>results.json</code> was found, and this report quotes measured " +
          "numbers throughout. Generate them with <code>python website/tools/analyze_samples.py</code> and " +
          "<code>python website/tools/run_samples.py</code>.</p></div>";
      }
      return;
    }
    host.setAttribute("data-state", "built");
    host.innerHTML = build(eda, res);
  }

  // The two JSON files arrive independently and out of order, so rebuild on
  // every one of them rather than latching onto whichever landed first.
  document.addEventListener("DOMContentLoaded", init);
  document.addEventListener("data:eda", init);
  document.addEventListener("data:results", init);
  document.addEventListener("data:ready", init);
})();
