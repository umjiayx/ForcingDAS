/* Frame-synchronised comparison viewers for the ForcingDAS project page.
 * Sprites are vertical strips (one tile per frame) written by tools/make_animations.py;
 * window.ANIM (static/anim/data.js) carries the per-frame NRMSE and colour-bar metadata. */
(function () {
  "use strict";
  const A = window.ANIM;
  const BASE = "static/anim/";
  const COLORS = {
    enkf: "#8a8f99", "3d_var": "#8a8f99", flowdas: "#8b5cf6", sda: "#2b7bd6",
    tensor_var: "#14a3b8", forcingdas_ar: "#ea580c", forcingdas_fs: "#c81e1e",
  };
  const DOMAINS = {
    ns: {
      refs: ["gt", "obs"],
      rows: [["Filtering", ["enkf", "flowdas", "forcingdas_ar"]],
             ["Full-sequence smoothing", ["sda", "forcingdas_fs"]]],
      time: (k) => `k = ${k}`,
    },
    sevir: {
      refs: ["gt", "obs"],
      rows: [["Filtering", ["flowdas", "forcingdas_ar"]],
             ["Full-sequence smoothing", ["sda", "forcingdas_fs"]]],
      time: (k) => `k = ${k}  ·  t = ${(k - A.sevir.ctx + 1) * 5 >= 0 ? "+" : ""}${(k - A.sevir.ctx + 1) * 5} min`,
    },
    era5: {
      refs: ["gt", "obs"],
      rows: [["Filtering", ["3d_var", "flowdas", "forcingdas_ar"]],
             ["Full-sequence smoothing", ["tensor_var", "forcingdas_fs"]]],
      time: (k) => {
        const h = (k - A.era5.ctx + 1) * 6;
        return `k = ${k}  ·  t = ${h >= 0 ? "+" : "−"}${Math.floor(Math.abs(h) / 24)} d ${Math.abs(h) % 24} h`;
      },
    },
  };
  const ICON_PLAY = '<svg viewBox="0 0 16 16"><path d="M4 2.5v11l9-5.5z"/></svg>';
  const ICON_PAUSE = '<svg viewBox="0 0 16 16"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z"/></svg>';
  const el = (tag, cls, html) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  };
  const gradient = (cols, discrete) => discrete
    ? `linear-gradient(90deg, ${cols.map((c, i) => `${c} ${(i / cols.length) * 100}% ${((i + 1) / cols.length) * 100}%`).join(", ")})`
    : `linear-gradient(90deg, ${cols.join(", ")})`;
  const fmt = (v) => (v == null ? "—" : v < 0.1 ? v.toFixed(3) : v.toFixed(3));

  /* ---------------------------------------------------------------- the engine */
  function Player(T, onFrame) {
    this.T = T; this.k = 0; this.fps = 8; this.playing = false; this.visible = false;
    this.onFrame = onFrame; this._last = 0; this._acc = 0;
    const tick = (ts) => {
      if (this.playing && this.visible) {
        const dt = this._last ? ts - this._last : 0;
        this._acc += dt;
        const step = 1000 / this.fps;
        if (this._acc >= step) {
          this._acc %= step;
          this.set((this.k + 1) % this.T);
        }
      }
      this._last = ts;
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  Player.prototype.set = function (k) { this.k = k; this.onFrame(k); };

  function observe(node, player, onFirstVisible) {
    let first = true;
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      player.visible = e.isIntersecting;
      if (e.isIntersecting && first) { first = false; onFirstVisible && onFirstVisible(); }
    }), { rootMargin: "200px" });
    io.observe(node);
  }

  function tileNode(label, opts) {
    const t = el("div", "tile" + (opts.ours ? " ours" : ""));
    const fig = el("figure");
    const img = el("div", "img");
    img.style.aspectRatio = `${opts.W} / ${opts.H}`;
    img.style.backgroundSize = `100% ${opts.T * 100}%`;
    if (opts.coast) {
      const c = el("img", "coast"); c.src = BASE + "coast.svg"; c.alt = ""; img.appendChild(c);
    }
    const cap = el("figcaption");
    const n = el("span", "n", label);
    const v = el("span", "v", "");
    cap.append(n, v);
    fig.append(img, cap);
    t.appendChild(fig);
    return { node: t, img, val: v };
  }
  function setFrame(img, k, T) {
    img.style.backgroundPosition = `0 ${T > 1 ? (k / (T - 1)) * 100 : 0}%`;
  }

  /* ---------------------------------------------------------------- NRMSE chart */
  function Chart(host, T, ctx, onSeek) {
    this.T = T; this.ctx = ctx;
    this.W = 900; this.H = 190; this.m = { l: 46, r: 12, t: 10, b: 30 };
    const ns = "http://www.w3.org/2000/svg";
    this.ns = ns;
    this.svg = document.createElementNS(ns, "svg");
    this.svg.setAttribute("viewBox", `0 0 ${this.W} ${this.H}`);
    this.svg.setAttribute("role", "img");
    host.appendChild(this.svg);
    const seek = (ev) => {
      const r = this.svg.getBoundingClientRect();
      const x = ((ev.clientX - r.left) / r.width) * this.W;
      const k = Math.round(((x - this.m.l) / (this.W - this.m.l - this.m.r)) * (T - 1));
      onSeek(Math.max(0, Math.min(T - 1, k)));
    };
    let down = false;
    this.svg.addEventListener("pointerdown", (e) => { down = true; seek(e); });
    window.addEventListener("pointerup", () => (down = false));
    this.svg.addEventListener("pointermove", (e) => down && seek(e));
  }
  Chart.prototype.x = function (k) { return this.m.l + (k / (this.T - 1)) * (this.W - this.m.l - this.m.r); };
  Chart.prototype.y = function (v) { return this.m.t + (1 - v / this.ymax) * (this.H - this.m.t - this.m.b); };
  Chart.prototype.draw = function (series, ylabel) {
    const ns = this.ns, s = this.svg;
    s.innerHTML = "";
    let mx = 0;
    series.forEach((d) => d.v.forEach((v) => { if (v != null && v > mx) mx = v; }));
    const nice = [0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2];
    let stepY = nice.find((n) => mx / n <= 5) || 1;
    this.ymax = Math.ceil(mx / stepY) * stepY || 1;
    const add = (tag, attrs, parent) => {
      const e = document.createElementNS(ns, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      (parent || s).appendChild(e);
      return e;
    };
    add("rect", { x: this.x(0), y: this.m.t, width: this.x(this.ctx - 0.5) - this.x(0),
                  height: this.H - this.m.t - this.m.b, fill: "var(--ctx)", "fill-opacity": 0.09 });
    const g = add("g", { class: "grid" });
    for (let v = 0; v <= this.ymax + 1e-9; v += stepY) {
      add("line", { x1: this.m.l, x2: this.W - this.m.r, y1: this.y(v), y2: this.y(v) }, g);
      const t = add("text", { x: this.m.l - 6, y: this.y(v) + 4, "text-anchor": "end" });
      t.textContent = +v.toFixed(3);
    }
    const xstep = this.T > 40 ? 10 : 5;
    for (let k = 0; k < this.T; k += xstep) {
      const t = add("text", { x: this.x(k), y: this.H - this.m.b + 16, "text-anchor": "middle" });
      t.textContent = k;
    }
    const xl = add("text", { x: this.W - this.m.r, y: this.H - 2, "text-anchor": "end" });
    xl.textContent = "frame k";
    const yl = add("text", { x: 12, y: this.m.t + (this.H - this.m.t - this.m.b) / 2,
                             transform: `rotate(-90 12 ${this.m.t + (this.H - this.m.t - this.m.b) / 2})`,
                             "text-anchor": "middle" });
    yl.textContent = ylabel;
    const ct = add("text", { x: this.x((this.ctx - 1) / 2), y: this.m.t + 13, "text-anchor": "middle",
                             style: "fill: var(--ctx); font-weight: 700" });
    ct.textContent = "context";
    series.forEach((d) => {
      let p = "", pen = false;
      d.v.forEach((v, k) => {
        if (v == null) { pen = false; return; }
        p += (pen ? "L" : "M") + this.x(k).toFixed(1) + " " + this.y(v).toFixed(1);
        pen = true;
      });
      add("path", { d: p, fill: "none", stroke: d.color, "stroke-width": d.ours ? 2.6 : 1.7,
                    "stroke-dasharray": d.ours ? "" : "5 3", "stroke-linejoin": "round" });
    });
    this.head = add("line", { y1: this.m.t, y2: this.H - this.m.b, stroke: "var(--text)",
                              "stroke-width": 1.2, "stroke-opacity": 0.7 });
  };
  Chart.prototype.at = function (k) {
    if (this.head) { this.head.setAttribute("x1", this.x(k)); this.head.setAttribute("x2", this.x(k)); }
  };

  /* ---------------------------------------------------------------- full viewer */
  function buildViewer(root, dom) {
    const cfg = DOMAINS[dom], M = A[dom];
    const T = M.T, ctx = M.ctx, isEra = dom === "era5";
    const label = {}; const ours = {};
    M.methods.forEach((m) => { label[m.key] = m.label; ours[m.key] = !!m.ours; });
    label.obs = M.obs_label.replace(/ \(.*/, "");
    const state = { mode: "field", v: isEra ? M.variables[0].key : null };

    const head = el("div", "vhead");
    head.append(el("div", "setting", `${M.setting}. Held-out trajectory #${M.traj}; every method assimilates the identical <i>y</i>.`));
    root.appendChild(head);

    const grid = el("div", "grid");
    const refcol = el("div", "refcol");
    const methods = el("div", "methods");
    grid.append(refcol, methods);
    root.appendChild(grid);
    const tiles = {};
    const mk = (key, parent) => {
      const tl = tileNode(key === "obs" ? M.obs_label : label[key],
                          { W: M.W, H: M.H, T, ours: ours[key], coast: isEra });
      tiles[key] = tl; parent.appendChild(tl.node);
    };
    cfg.refs.forEach((k) => mk(k, refcol));
    cfg.rows.forEach(([name, keys]) => {
      methods.appendChild(el("div", "mrow-label", name));
      const r = el("div", "mrow");
      keys.forEach((k) => mk(k, r));
      methods.appendChild(r);
    });

    const controls = el("div", "controls");
    const play = el("button", "play", ICON_PLAY);
    play.setAttribute("aria-label", "Play");
    const slider = el("input"); slider.type = "range"; slider.min = 0; slider.max = T - 1; slider.value = 0;
    slider.setAttribute("aria-label", "Frame");
    const fl = el("div", "frame-label");
    const modeSeg = el("div", "seg");
    [["field", "Estimate"], ["err", "|Error|"]].forEach(([m, txt]) => {
      const b = el("button", null, txt); b.setAttribute("aria-pressed", m === state.mode);
      b.onclick = () => { state.mode = m; [...modeSeg.children].forEach((c) => c.setAttribute("aria-pressed", c === b)); refresh(); };
      modeSeg.appendChild(b);
    });
    const speed = el("div", "seg");
    [["0.5×", 4], ["1×", 8], ["2×", 16]].forEach(([txt, f]) => {
      const b = el("button", null, txt); b.setAttribute("aria-pressed", f === 8);
      b.onclick = () => { player.fps = f; [...speed.children].forEach((c) => c.setAttribute("aria-pressed", c === b)); };
      speed.appendChild(b);
    });
    controls.append(play, slider, fl, modeSeg, speed);
    let varSeg = null;
    if (isEra) {
      varSeg = el("div", "seg");
      M.variables.forEach((v) => {
        const b = el("button", null, v.key.toUpperCase()); b.setAttribute("aria-pressed", v.key === state.v);
        b.onclick = () => { state.v = v.key; [...varSeg.children].forEach((c) => c.setAttribute("aria-pressed", c === b)); refresh(); };
        varSeg.appendChild(b);
      });
      controls.appendChild(varSeg);
    }
    const cbar = el("div", "cbar");
    controls.appendChild(cbar);
    root.appendChild(controls);

    const chartHost = el("div", "chart");
    const legend = el("div", "legend");
    root.append(chartHost, legend);
    const chart = new Chart(chartHost, T, ctx, (k) => { player.set(k); });

    let loaded = false;
    const src = (key) => {
      const mode = key === "obs" || key === "gt" ? "field" : state.mode;
      return BASE + dom + "/" + (isEra ? state.v + "_" : "") + `${key}_${mode}.webp`;
    };
    const curNrmse = () => (isEra ? M.variables.find((v) => v.key === state.v).nrmse : M.nrmse);
    function refresh() {
      if (loaded) Object.keys(tiles).forEach((k) => { tiles[k].img.style.backgroundImage = `url(${src(k)})`; });
      const vm = isEra ? M.variables.find((v) => v.key === state.v) : M;
      const range = state.mode === "err" ? vm.err_range : vm.field_range;
      const cols = state.mode === "err" ? M.err_cmap : vm.field_cmap;
      const unit = isEra ? vm.label : (state.mode === "err" ? M.field_label.replace(/ \(.*/, "") : M.field_label);
      cbar.innerHTML = "";
      const bar = el("span", "bar"); bar.style.background = gradient(cols, state.mode !== "err" && M.field_cmap_discrete);
      cbar.append(el("span", null, (+range[0].toFixed(2)).toString()), bar,
                  el("span", null, (+range[1].toFixed(2)).toString()),
                  el("span", null, (state.mode === "err" ? "|error| " : "") + unit));
      const nr = curNrmse();
      const series = [];
      cfg.rows.forEach(([, keys]) => keys.forEach((k) => series.push({ key: k, v: nr[k], color: COLORS[k], ours: ours[k] })));
      chart.draw(series, isEra ? "lat-weighted RMSE (z-score)" : "NRMSE");
      legend.innerHTML = "";
      series.forEach((d) => {
        const sp = el("span");
        const sw = el("i"); sw.style.background = d.color;
        if (!d.ours) sw.style.background = `repeating-linear-gradient(90deg, ${d.color} 0 5px, transparent 5px 8px)`;
        sp.append(sw, document.createTextNode(label[d.key]));
        legend.appendChild(sp);
      });
      Object.keys(tiles).forEach((k) => {
        const c = COLORS[k];
        if (c) tiles[k].node.querySelector(".n").style.borderBottom = `2px ${ours[k] ? "solid" : "dashed"} ${c}`;
      });
      onFrame(player ? player.k : 0);
    }
    function onFrame(k) {
      Object.keys(tiles).forEach((key) => {
        setFrame(tiles[key].img, k, T);
        tiles[key].img.classList.toggle("ctx", k < ctx);
        const nr = curNrmse()[key];
        tiles[key].val.textContent = nr ? (k < ctx ? "context" : fmt(nr[k])) : "";
      });
      slider.value = k;
      fl.innerHTML = cfg.time(k) + (k < ctx ? '  <span class="ctxbadge">· clean context</span>' : "");
      chart.at(k);
    }
    const player = new Player(T, onFrame);
    slider.oninput = () => player.set(+slider.value);
    const setPlaying = (p) => {
      player.playing = p; play.innerHTML = p ? ICON_PAUSE : ICON_PLAY;
      play.setAttribute("aria-label", p ? "Pause" : "Play");
    };
    play.onclick = () => setPlaying(!player.playing);
    slider.addEventListener("pointerdown", () => setPlaying(false));
    observe(root, player, () => { loaded = true; refresh(); });
    setPlaying(true);
    refresh();
  }

  /* ---------------------------------------------------------------- teaser */
  function buildTeaser(root) {
    const M = A.sevir, T = M.T, ctx = M.ctx;
    const keys = ["obs", "gt", "flowdas", "forcingdas_ar"];
    const names = { obs: "Observation y (10% of pixels)", gt: "Ground truth",
                    flowdas: "FlowDAS (filter)", forcingdas_ar: "ForcingDAS (filter)" };
    const row = el("div", "mrow four");
    const tiles = keys.map((k) => {
      const tl = tileNode(names[k], { W: M.W, H: M.H, T, ours: k === "forcingdas_ar" });
      row.appendChild(tl.node);
      return [k, tl];
    });
    root.appendChild(row);
    const clock = el("div", "teaser-clock");
    root.appendChild(clock);
    const player = new Player(T, (k) => {
      tiles.forEach(([key, tl]) => {
        setFrame(tl.img, k, T);
        tl.img.classList.toggle("ctx", k < ctx);
        const nr = M.nrmse[key];
        tl.val.textContent = nr && k >= ctx ? "NRMSE " + nr[k].toFixed(2) : "";
      });
      clock.innerHTML = k < ctx
        ? `<span class="ctxbadge">clean context</span> · frame ${k + 1} of ${ctx}`
        : `assimilating · +${(k - ctx + 1) * 5} min (frame ${k - ctx + 1} of ${T - ctx})`;
    });
    player.fps = 6; player.playing = true;
    observe(root, player, () => tiles.forEach(([key, tl]) => {
      tl.img.style.backgroundImage = `url(${BASE}sevir/${key}_field.webp)`;
    }));
    player.set(0);
  }

  document.addEventListener("DOMContentLoaded", () => {
    const teaser = document.getElementById("teaser-anim");
    if (teaser) buildTeaser(teaser);
    const tabs = [...document.querySelectorAll(".tab")];
    const views = {};
    tabs.forEach((t) => {
      const dom = t.dataset.dom;
      const v = document.getElementById("viewer-" + dom);
      views[dom] = v;
      buildViewer(v, dom);
      t.onclick = () => {
        tabs.forEach((x) => x.setAttribute("aria-selected", x === t));
        Object.entries(views).forEach(([d, node]) => (node.hidden = d !== dom));
      };
    });
    // deep links: #ns / #sevir / #era5 open that tab
    const h = location.hash.slice(1);
    const hit = tabs.find((t) => t.dataset.dom === h);
    if (hit) { hit.click(); document.getElementById("results").scrollIntoView(); }
    document.querySelectorAll(".copy").forEach((b) => (b.onclick = () => {
      const txt = document.getElementById(b.dataset.target).innerText;
      navigator.clipboard && navigator.clipboard.writeText(txt).then(() => {
        b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1400);
      });
    }));
  });
})();
