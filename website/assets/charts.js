/* =====================================================================
   charts.js — tiny hand-rolled chart kit. No dependencies, no build.
   Everything renders into an <svg> sized by viewBox, so it is fluid and
   crisp on a phone. All charts expose a `src` (data path) for the
   "where did this number come from" line under each figure.
   ===================================================================== */
(function (global) {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";

  function el(name, attrs, text) {
    var n = document.createElementNS(NS, name);
    if (attrs) for (var k in attrs) if (attrs[k] !== null && attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  function nice(max, min) {
    if (max === min) { max = min + 1; }
    var span = max - min;
    var step = Math.pow(10, Math.floor(Math.log(span / 5) / Math.LN10));
    var err = (span / 5) / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    return { step: step, lo: Math.floor(min / step) * step, hi: Math.ceil(max / step) * step };
  }

  function fmt(v, d) {
    if (v === null || v === undefined || isNaN(v)) return "—";
    if (d === undefined) d = Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2;
    return Number(v).toFixed(d);
  }

  function clock(sec) {
    sec = Math.max(0, Math.round(sec));
    var m = Math.floor(sec / 60), s = sec % 60;
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  /* A reusable time-axis line chart with optional threshold rules.
     series: [{key,label,color,values:[y],area:bool,dash:bool}] over shared xs. */
  function lineChart(host, opts) {
    var xs = opts.x, series = opts.series || [];
    var W = 1000, H = opts.height || 300;
    var m = { t: 16, r: 14, b: 34, l: 46 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;

    var lo = 0, hi = 0, all = [];
    series.forEach(function (s) { all = all.concat(s.values.filter(function (v) { return v !== null && !isNaN(v); })); });
    if (opts.min !== undefined) lo = opts.min;
    if (opts.max !== undefined) hi = opts.max;
    if (all.length) { lo = Math.min(lo, Math.min.apply(null, all)); hi = Math.max(hi, Math.max.apply(null, all)); }
    if (opts.yFloorZero) lo = 0;
    var n = nice(hi, lo); hi = n.hi; lo = opts.yFloorZero ? 0 : n.lo;

    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, class: "chart", preserveAspectRatio: "none", role: "img" });
    svg.setAttribute("aria-label", opts.aria || "chart");
    var xmax = xs.length ? xs[xs.length - 1] : 1;
    var X = function (v) { return m.l + (xmax ? (v / xmax) * iw : 0); };
    var Y = function (v) { return m.t + ih - ((v - lo) / (hi - lo || 1)) * ih; };

    // y grid + labels
    var ticks = 5, i;
    for (i = 0; i <= ticks; i++) {
      var v = lo + ((hi - lo) * i) / ticks, y = Y(v);
      svg.appendChild(el("line", { class: "grid-line", x1: m.l, x2: m.l + iw, y1: y, y2: y }));
      svg.appendChild(el("text", { class: "lbl", x: m.l - 7, y: y + 3.5, "text-anchor": "end" }, fmt(v, hi - lo < 6 ? 2 : 0)));
    }
    // x ticks
    var nx = Math.min(8, xs.length || 1);
    for (i = 0; i <= nx; i++) {
      var xv = (xmax * i) / nx, x = X(xv);
      svg.appendChild(el("text", { class: "lbl", x: x, y: H - 14, "text-anchor": i === 0 ? "start" : (i === nx ? "end" : "middle") },
        opts.xFormat === "time" ? clock(xv) : String(Math.round(xv)) + (opts.xUnit || "")));
    }
    svg.appendChild(el("line", { class: "axis", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }));
    if (opts.yLabel) svg.appendChild(el("text", { class: "lbl", x: m.l, y: 10 }, opts.yLabel));

    // threshold rules
    (opts.rules || []).forEach(function (r) {
      var y = Y(r.value);
      svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: y, y2: y, stroke: r.color || "#f59e0b", "stroke-width": 1.4, "stroke-dasharray": "6 4", opacity: .9 }));
      svg.appendChild(el("text", { class: "val", x: m.l + iw - 4, y: y - 5, "text-anchor": "end", fill: r.color || "#f59e0b" }, r.label));
    });
    // vertical markers
    (opts.markers || []).forEach(function (mk) {
      var x = X(mk.x);
      svg.appendChild(el("line", { x1: x, x2: x, y1: m.t, y2: m.t + ih, stroke: mk.color || "#f87171", "stroke-width": 1.3, opacity: .75 }));
      if (mk.label) svg.appendChild(el("text", { class: "val", x: x + 4, y: m.t + 11, fill: mk.color || "#f87171" }, mk.label));
    });

    // series
    var self = this;
    series.forEach(function (s) {
      var d = "", started = false, areaD = "";
      s.values.forEach(function (v, i) {
        if (v === null || isNaN(v)) { started = false; return; }
        var x = X(xs[i]), y = Y(v);
        d += (started ? "L" : "M") + x.toFixed(2) + " " + y.toFixed(2) + " ";
        started = true;
      });
      if (s.area) {
        areaD = d + "L" + X(xs[xs.length - 1]).toFixed(2) + " " + Y(lo).toFixed(2) + " L" + X(xs[0]).toFixed(2) + " " + Y(lo).toFixed(2) + " Z";
        svg.appendChild(el("path", { d: areaD, fill: s.color, opacity: .14 }));
      }
      svg.appendChild(el("path", {
        d: d, fill: "none", stroke: s.color, "stroke-width": s.width || 1.8,
        "stroke-linejoin": "round", "stroke-linecap": "round",
        "stroke-dasharray": s.dash ? "5 4" : null
      }));
    });

    // crosshair + readout
    var cross = el("line", { class: "cross", x1: 0, x2: 0, y1: m.t, y2: m.t + ih, opacity: 0 });
    svg.appendChild(cross);
    var tip = el("div", { class: "tl-tooltip", style: "display:none" });
    var hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent", style: "cursor:crosshair" });
    svg.appendChild(hit);
    function move(ev) {
      var r = svg.getBoundingClientRect();
      var px = ((ev.clientX - r.left) / r.width) * W;
      var idx = Math.round(((px - m.l) / iw) * (xs.length - 1));
      idx = Math.max(0, Math.min(xs.length - 1, idx));
      cross.setAttribute("x1", X(xs[idx])); cross.setAttribute("x2", X(xs[idx])); cross.setAttribute("opacity", 1);
      tip.style.display = "block";
      tip.style.left = Math.min(window.innerWidth - 300, ev.clientX + 12) + "px";
      tip.style.top = (ev.clientY + 12) + "px";
      var html = "<b>" + (opts.xFormat === "time" ? clock(xs[idx]) + " (" + fmt(xs[idx], 1) + " s)" : (opts.xLabel || "x") + " " + fmt(xs[idx], 1)) + "</b>";
      series.forEach(function (s) {
        var v = s.values[idx];
        if (v === null || v === undefined || isNaN(v)) return;
        html += '<div style="color:' + s.color + '">' + s.label + ": " + fmt(v, 3) + "</div>";
      });
      tip.innerHTML = html;
    }
    function leave() { cross.setAttribute("opacity", 0); tip.style.display = "none"; }
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerdown", move);
    hit.addEventListener("pointerleave", leave);
    document.addEventListener("scroll", leave, { passive: true });

    host.innerHTML = "";
    host.appendChild(svg);
    host.appendChild(tip);
    if (opts.legend) {
      var ul = document.createElement("ul");
      ul.className = "chart-legend";
      series.forEach(function (s) {
        var li = document.createElement("li");
        li.innerHTML = '<i style="background:' + s.color + (s.dash ? ";opacity:.7" : "") + '"></i>' + s.label;
        ul.appendChild(li);
      });
      host.appendChild(ul);
    }
    return svg;
  }

  /* Grouped/stacked bar chart. cats: [{label, values:[..]}] */
  function barChart(host, opts) {
    var cats = opts.cats, series = opts.series || [];
    var W = 1000, H = opts.height || 260;
    var m = { t: 16, r: 14, b: 42, l: 46 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var stacked = !!opts.stacked;
    var tops = cats.map(function (c) {
      return stacked ? c.values.reduce(function (a, b) { return a + (b || 0); }, 0) : Math.max.apply(null, c.values);
    });
    var n = nice(Math.max.apply(null, tops.concat([1])), 0);
    var hi = n.hi;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, class: "chart", preserveAspectRatio: "none", role: "img" });
    svg.setAttribute("aria-label", opts.aria || "bar chart");
    var Y = function (v) { return m.t + ih - (v / (hi || 1)) * ih; };
    var i;
    for (i = 0; i <= 4; i++) {
      var v = (hi * i) / 4, y = Y(v);
      svg.appendChild(el("line", { class: "grid-line", x1: m.l, x2: m.l + iw, y1: y, y2: y }));
      svg.appendChild(el("text", { class: "lbl", x: m.l - 7, y: y + 3.5, "text-anchor": "end" }, fmt(v, 0)));
    }
    var bw = iw / Math.max(1, cats.length);
    var inner = Math.max(2, Math.min(34, bw * (stacked ? 0.6 : 0.8) / series.length));
    cats.forEach(function (c, ci) {
      var cx = m.l + bw * ci + bw / 2;
      var acc = 0;
      series.forEach(function (s, si) {
        var v = c.values[si] || 0;
        var w = stacked ? inner : inner;
        var x = stacked ? cx - w / 2 : cx - (inner * series.length) / 2 + si * inner;
        var y0 = Y(stacked ? acc : 0), y1 = Y(stacked ? acc + v : v);
        svg.appendChild(el("rect", {
          x: x.toFixed(2), y: y1.toFixed(2), width: w.toFixed(2), height: Math.max(0, y0 - y1).toFixed(2),
          fill: s.color, opacity: stacked ? .92 : .88, rx: Math.min(2, w / 3)
        }));
        acc += v;
      });
      if (ci % (opts.every || 1) === 0) {
        svg.appendChild(el("text", { class: "lbl", x: cx, y: H - 22, "text-anchor": "middle" }, c.label));
      }
    });
    svg.appendChild(el("line", { class: "axis", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }));
    if (opts.xLabel) svg.appendChild(el("text", { class: "lbl", x: m.l + iw / 2, y: H - 6, "text-anchor": "middle" }, opts.xLabel));
    if (opts.yLabel) svg.appendChild(el("text", { class: "lbl", x: m.l, y: 10 }, opts.yLabel));
    host.innerHTML = "";
    host.appendChild(svg);
    if (opts.legend) {
      var ul = document.createElement("ul"); ul.className = "chart-legend";
      series.forEach(function (s) {
        var li = document.createElement("li");
        li.innerHTML = '<i class="dot" style="background:' + s.color + '"></i>' + s.label;
        ul.appendChild(li);
      });
      host.appendChild(ul);
    }
    return svg;
  }

  /* Heatmap on a canvas from a flat grid + value array. Hover readout. */
  function heatmap(host, spec) {
    var w = spec.grid_w, h = spec.grid_h, values = spec.values;
    var vmax = spec.max, vmin = spec.min;
    if (vmax === vmin) vmax = vmin + 1;
    var cw = 7, ch = 4; // css size, canvas is scaled by dpr
    var canvas = document.createElement("canvas");
    var dpr = Math.min(2, global.devicePixelRatio || 1);
    canvas.width = w * cw * dpr; canvas.height = h * ch * dpr;
    canvas.style.width = (w * cw) + "px"; canvas.style.height = (h * ch) + "px";
    canvas.style.width = "100%";
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", spec.aria || "heatmap");
    var ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var v = values[y * w + x];
        var t = (v - vmin) / (vmax - vmin);
        t = Math.max(0, Math.min(1, Math.pow(t, spec.gamma || 0.72)));
        ctx.fillStyle = ramp(t);
        ctx.fillRect(x * cw * dpr, y * ch * dpr, cw * dpr + 0.5, ch * dpr + 0.5);
      }
    }
    // readout
    var side = document.createElement("div");
    side.className = "heat-side";
    side.innerHTML =
      '<div><b style="color:var(--fg)">' + spec.title + "</b><br>" + (spec.subtitle || "") + "</div>" +
      '<div style="display:flex;align-items:center;gap:.4rem"><span>' + fmt(vmin, 2) + '</span><i class="cbar" style="background:linear-gradient(180deg,' +
      ramp(1) + "," + ramp(.75) + "," + ramp(.5) + "," + ramp(.25) + "," + ramp(0) + ')"></i><span>' + fmt(vmax, 2) + "</span></div>" +
      '<div class="heat-readout" id="hm-read">hover the map</div>';

    var box = document.createElement("div");
    box.className = "heat";
    box.appendChild(canvas);
    box.appendChild(side);
    var read = box.querySelector("#hm-read");
    canvas.addEventListener("pointermove", function (ev) {
      var r = canvas.getBoundingClientRect();
      var fx = (ev.clientX - r.left) / r.width, fy = (ev.clientY - r.top) / r.height;
      var gx = Math.floor(fx * w), gy = Math.floor(fy * h);
      if (gx < 0 || gy < 0 || gx >= w || gy >= h) return;
      read.textContent = "cell " + gx + "," + gy + " → " + fmt(values[gy * w + gx], 3);
    });
    canvas.addEventListener("pointerleave", function () { read.textContent = "hover the map"; });
    host.innerHTML = "";
    host.appendChild(box);
    return canvas;
  }

  /* Perceptually-ish dark ramp; returns css rgb. */
  function ramp(t) {
    t = Math.max(0, Math.min(1, t));
    var stops = [
      [0.00, [8, 14, 24]], [0.18, [22, 45, 84]], [0.36, [26, 104, 138]],
      [0.55, [46, 160, 116]], [0.74, [186, 186, 68]], [0.88, [226, 130, 44]], [1.0, [240, 66, 66]]
    ];
    for (var i = 1; i < stops.length; i++) {
      if (t <= stops[i][0]) {
        var a = stops[i - 1], b = stops[i];
        var f = (t - a[0]) / (b[0] - a[0] || 1);
        return "rgb(" + Math.round(a[1][0] + (b[1][0] - a[1][0]) * f) + "," +
          Math.round(a[1][1] + (b[1][1] - a[1][1]) * f) + "," +
          Math.round(a[1][2] + (b[1][2] - a[1][2]) * f) + ")";
      }
    }
    return "rgb(240,66,66)";
  }

  global.Charts = { line: lineChart, bar: barChart, heatmap: heatmap, fmt: fmt, clock: clock, ramp: ramp };
})(window);
