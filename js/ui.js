/*
 * ui.js — renders every section of the page from a single context object:
 *   ctx = { problem, dataset, analysis, labels, view: { method, k }, inputs }
 * and wires the control events back to app.js through callbacks.
 */
(function (root) {
  'use strict';
  const R = root.Regression;
  const $ = (id) => document.getElementById(id);
  const charts = {};

  /* ---------------- formatting ---------------- */
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const num = (x, d = 2) => (isFinite(x) ? Number(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }) : (isNaN(x) ? '–' : '∞'));
  const money = (ctx, x, d = 2) => {
    if (!isFinite(x)) return num(x, d);
    const s = num(Math.abs(x), d);
    const sign = x < 0 ? '-' : '';
    return ctx.problem.money ? `${sign}$${s}` : `${sign}${s}`;
  };
  const signedMoney = (ctx, x, d = 2) => (x >= 0 ? '+' : '') + money(ctx, x, d);
  const coefStr = (b, big = 4) => {
    const a = Math.abs(b);
    if (a !== 0 && a < 1e-4) return b.toExponential(3);
    return num(b, a >= 100 ? 2 : big);
  };
  const tableCoef = (b) => (Math.abs(b) !== 0 && Math.abs(b) < 1e-4 ? b.toExponential(3) : num(b, 4));
  const pStr = (p) => (p < 1e-4 ? '< 0.0001' : num(p, 4));

  /* ---------------- helpers on the context ---------------- */
  function activeMethodKey(ctx) {
    const { analysis, view } = ctx;
    if (view.method !== 'compare') return view.method;
    let best = analysis.methodKeys[0];
    for (const key of analysis.methodKeys) {
      if (analysis.methods[key].results[view.k - 1].test.rmse < analysis.methods[best].results[view.k - 1].test.rmse - 1e-9) best = key;
    }
    return best;
  }
  function activeResult(ctx) {
    return ctx.analysis.methods[activeMethodKey(ctx)].results[ctx.view.k - 1];
  }
  /** Best k of the current view (method sweet spot, or global in Compare All). */
  function sweetK(ctx) {
    return ctx.view.method === 'compare' ? ctx.analysis.globalBest.k : ctx.analysis.methods[ctx.view.method].bestK;
  }
  function curve(ctx) {
    const key = ctx.view.method === 'compare' ? ctx.analysis.globalBest.method : ctx.view.method;
    return ctx.analysis.methods[key].results;
  }
  /** smallest k whose test RMSE is within 1% of the curve minimum */
  function parsimoniousK(results) {
    const min = Math.min(...results.map((r) => r.test.rmse));
    return results.find((r) => r.test.rmse <= min * 1.01).k;
  }

  /* ---------------- hero & header ---------------- */
  function renderHeader(ctx) {
    const { problem, analysis, dataset } = ctx;
    $('page-title').textContent = `${problem.title} Multiple Linear Regression Explorer`;
    $('crumb-title').textContent = `${problem.title} Feature Progression`;
    document.title = `${problem.title} · MLR Explorer`;
    $('footer-title').textContent = `${problem.title} CRISP-DM Machine Learning Project`;
    $('footer-dataset').innerHTML = problem.sourceUrl
      ? `<a href="${esc(problem.sourceUrl)}" target="_blank" rel="noopener">${esc(problem.sourceName)}</a>`
      : esc(problem.sourceName);

    const gb = analysis.globalBest, p = analysis.p, full = analysis.full;
    const names = analysis.methodKeys.map((k) => analysis.methods[k].name).join(', ');
    const kp = parsimoniousK(analysis.methods[gb.method].results);
    const tail = full.overfit
      ? `pinpointing the <strong>${gb.k}-feature optimal sweet spot</strong> and the <strong>${p}-feature overfitting cliff</strong>.`
      : `pinpointing the <strong>${gb.k}-feature optimal sweet spot</strong> (the full ${p}-feature model does not overfit here; the most parsimonious model within 1% of the best RMSE uses <strong>k=${kp}</strong>).`;
    $('hero-text').innerHTML =
      `${esc(problem.description)} Compare ${analysis.methodKeys.length} Machine Learning Feature Selection Algorithms ` +
      `(<strong>${esc(names)}</strong>) against Baseline OLS for predicting <strong>${esc(dataset.target)}</strong>. ` +
      `Explore the bias-variance tradeoff across feature counts k = 1 … ${p} (80/20 split, ${analysis.split.train.length} train / ${analysis.split.test.length} test), ${tail}`;

    const badges = [
      `<span class="hero-badge"><i class="fa-solid fa-database"></i> ${esc(problem.badge)} · ${analysis.n} rows</span>`,
      `<span class="hero-badge"><i class="fa-solid fa-bolt"></i> ${analysis.methodKeys.length} Selection Methods</span>`,
      `<span class="hero-badge"><i class="fa-solid fa-star"></i> Global Sweet Spot: k=${gb.k} (${money(ctx, gb.rmse)} RMSE)</span>`,
      full.overfit
        ? `<span class="hero-badge"><i class="fa-solid fa-triangle-exclamation"></i> Overfitting Warning: k=${p} (${money(ctx, full.rmse)} RMSE)</span>`
        : `<span class="hero-badge"><i class="fa-solid fa-circle-check"></i> No Overfitting Cliff: k=${p} full model ${gb.k === p ? 'is optimal' : `within 1% of best (${money(ctx, full.rmse)})`}</span>`,
    ];
    $('hero-badges').innerHTML = badges.join('');
  }

  /* ---------------- controls ---------------- */
  function renderControls(ctx, cb) {
    const { analysis, view } = ctx;
    const all = ['lasso', 'forward', 'rfe', 'elasticnet', 'exhaustive'];
    const meta = root.FeatureSelection.METHODS;
    const icons = { lasso: 'fa-filter', forward: 'fa-arrow-trend-up', rfe: 'fa-scissors', elasticnet: 'fa-diagram-project', exhaustive: 'fa-magnifying-glass' };
    let html = all.map((key) => {
      const enabled = analysis.methodKeys.includes(key);
      const title = enabled ? '' : ` title="Disabled: ${analysis.p} predictors > ${root.FeatureSelection.EXHAUSTIVE_MAX_P} (2^${analysis.p} subsets)"`;
      return `<button class="tbtn${view.method === key ? ' active' : ''}" data-method="${key}"${enabled ? '' : ' disabled'}${title}>` +
        `<i class="fa-solid ${icons[key]}"></i>${meta[key].name}${key === 'lasso' ? ' (L1 Ranking)' : ''} <span class="tag">${enabled ? (key === 'lasso' ? 'Benchmark' : meta[key].tag) : `p > ${root.FeatureSelection.EXHAUSTIVE_MAX_P}`}</span></button>`;
    }).join('');
    html += `<button class="tbtn${view.method === 'compare' ? ' active' : ''}" data-method="compare"><i class="fa-solid fa-layer-group"></i> Compare All ${analysis.methodKeys.length} Methods</button>`;
    $('method-buttons').innerHTML = html;

    $('k-label').innerHTML = `<i class="fa-solid fa-list-ol"></i> 2. Number of Features (k = 1 to ${analysis.p})`;
    const sk = sweetK(ctx);
    const results = curve(ctx);
    const nestedMethod = view.method === 'lasso' || view.method === 'elasticnet' || view.method === 'forward' || view.method === 'rfe';
    let kHtml = '';
    for (let k = 1; k <= analysis.p; k++) {
      const r = results[k - 1];
      let cls = 'tbtn', label;
      if (k === sk) { cls += ' k-sweet'; label = `k = ${k} ★ Optimal Sweet Spot <span class="sub">(Min RMSE)</span>`; }
      else if (k === analysis.p) {
        if (analysis.full.overfit) { cls += ' k-cliff'; label = `k = ${k} ⚠ Full Model <span class="sub">(Overfitting Cliff)</span>`; }
        else label = `k = ${k} <span class="sub">(Full Model)</span>`;
      } else if (nestedMethod) {
        const added = r.added.map((j) => ctx.labels[j]).join(', ');
        label = `k = ${k} <span class="sub">(${k === 1 ? '' : '+ '}${esc(added)})</span>`;
      } else label = `k = ${k} <span class="sub">(${k} predictors)</span>`;
      if (k === view.k) cls += ' active';
      kHtml += `<button class="${cls}" data-k="${k}">${label}</button>`;
    }
    $('k-buttons').innerHTML = kHtml;

    $('method-buttons').onclick = (e) => { const b = e.target.closest('button[data-method]'); if (b && !b.disabled) cb.onMethod(b.dataset.method); };
    $('k-buttons').onclick = (e) => { const b = e.target.closest('button[data-k]'); if (b) cb.onK(Number(b.dataset.k)); };
  }

  function renderNotes(ctx) {
    const notes = [...(ctx.dataset.notes || []), ...(ctx.analysis.notes || [])];
    $('notes').innerHTML = notes.map((n) => `<div class="alert alert-warn"><i class="fa-solid fa-circle-info"></i><div>${esc(n)}</div></div>`).join('');
  }

  /* ---------------- metrics ---------------- */
  function renderMetrics(ctx) {
    const r = activeResult(ctx), m = r.model, t = r.test;
    const gb = ctx.analysis.globalBest;
    const isSweet = ctx.view.k === sweetK(ctx);
    const isGlobal = Math.abs(t.rmse - gb.rmse) < 1e-6;
    const fSig = m.Fp < 0.05;
    const nTest = ctx.analysis.split.test.length;
    const methodNote = ctx.view.method === 'compare' ? ` · best at k=${ctx.view.k}: ${ctx.analysis.methods[activeMethodKey(ctx)].name}` : '';
    const cards = [
      {
        cls: isSweet ? 'good' : '', title: 'Test RMSE',
        badge: isSweet ? '<span class="badge badge-green"><i class="fa-solid fa-star"></i> Optimal Sweet Spot</span>' : '',
        value: money(ctx, t.rmse),
        sub: (isGlobal ? 'Lower is better (Best across methods)' : `Lower is better · ${signedMoney(ctx, t.rmse - gb.rmse)} vs global best`) + methodNote,
      },
      {
        title: 'R² Score', badge: `<span class="badge badge-indigo"><i class="fa-solid fa-circle-check"></i> ${num(t.r2 * 100, 1)}% FIT</span>`,
        value: num(t.r2, 4), sub: `Explains ${num(t.r2 * 100, 2)}% of ${esc(ctx.dataset.target)} variance (test set)`,
      },
      {
        title: 'Adjusted R²', badge: '<span class="badge badge-gray"><i class="fa-solid fa-scale-balanced"></i></span>',
        value: num(t.adjR2, 4), sub: `Penalizes redundant parameters (test n=${nTest}, k=${m.k})`,
      },
      {
        title: 'Mean Absolute Error', badge: '<span class="badge badge-gray"><i class="fa-solid fa-calculator"></i></span>',
        value: money(ctx, t.mae), sub: `Average prediction error${ctx.problem.money ? ' ($)' : ''}`,
      },
      {
        title: 'F-Statistic',
        badge: `<span class="badge ${fSig ? 'badge-green' : 'badge-amber'}">P ${m.Fp < 1e-4 ? '< 0.0001' : '= ' + num(m.Fp, 4)}</span>`,
        value: num(m.F, 1),
        sub: `${fSig ? 'Model is highly statistically significant' : 'Model is not statistically significant'} (df = ${m.k}, ${m.dfResid}; training set)`,
      },
    ];
    $('metrics').innerHTML = cards.map((c) =>
      `<div class="metric ${c.cls || ''}"><div class="metric-head"><span>${c.title}</span>${c.badge}</div>` +
      `<div class="metric-value">${c.value}</div><div class="metric-sub">${c.sub}</div></div>`).join('');

    // Active predictors + Occam's Razor
    const results = curve(ctx);
    const sk = sweetK(ctx), k = ctx.view.k;
    const best = results[sk - 1].test;
    const kp = parsimoniousK(results);
    let occam, ok = true;
    if (k === sk) {
      occam = kp < sk && sk === ctx.analysis.p
        ? `★ Lowest test RMSE at k=${sk} (${money(ctx, best.rmse)}), but Occam's Razor favours k=${kp}: within 1% (${money(ctx, results[kp - 1].test.rmse)}) using ${sk - kp} fewer feature${sk - kp > 1 ? 's' : ''}.`
        : `★ Occam's Razor: ${sk} feature${sk > 1 ? 's' : ''} yield the minimum test RMSE (${money(ctx, best.rmse)}) and peak R² (${num(best.r2, 4)}).`;
    } else if (k > sk) {
      ok = false;
      const d = t.rmse - best.rmse;
      occam = `⚠ ${k - sk} extra feature${k - sk > 1 ? 's' : ''} over the k=${sk} sweet spot raise test RMSE by ${money(ctx, d)} (+${num(100 * d / best.rmse, 1)}%) — added variance, not signal.`;
    } else if (k >= kp) {
      occam = `✓ Parsimonious: k=${k} is within 1% of the best test RMSE (k=${sk}) with ${sk - k} fewer feature${sk - k > 1 ? 's' : ''}.`;
    } else {
      ok = false;
      occam = `⚠ Under-fit: adding predictors up to k=${sk} still lowers test RMSE (${money(ctx, t.rmse)} → ${money(ctx, best.rmse)}).`;
    }
    $('predictors-strip').innerHTML =
      `<div class="left"><i class="fa-solid fa-list-check"></i> Active Predictors (${m.k}): ${m.cols.map((j) => `<span class="feature-tag">${esc(ctx.labels[j])}</span>`).join('')}</div>` +
      `<div class="occam ${ok ? 'ok' : 'warn'}">${occam}</div>`;
  }

  /* ---------------- charts ---------------- */
  function renderCharts(ctx, cb) {
    const { analysis, view } = ctx;
    const ks = Array.from({ length: analysis.p }, (_, i) => i + 1);
    const compare = view.method === 'compare';
    const keys = compare ? analysis.methodKeys : [view.method];
    const dashes = { lasso: [], forward: [7, 4], rfe: [2, 3], elasticnet: [12, 4], exhaustive: [4, 2, 1, 2] };
    const SWEET = '#f4845f';
    const sets = (metric) => keys.map((key) => {
      const m = analysis.methods[key];
      const color = compare ? m.color : '#1e3a5f';
      return {
        label: m.name,
        data: m.results.map((r) => r.test[metric]),
        borderColor: color,
        backgroundColor: color,
        borderDash: compare ? dashes[key] : [],
        borderWidth: compare ? 2 : 2.5,
        tension: 0,
        pointRadius: ks.map((k) => (k === m.bestK ? 7 : k === view.k ? 5 : 3.5)),
        pointHoverRadius: 8,
        pointBackgroundColor: ks.map((k) => (k === m.bestK ? SWEET : color)),
        pointBorderColor: ks.map((k) => (k === view.k ? '#16243a' : '#fff')),
        pointBorderWidth: ks.map((k) => (k === view.k ? 2 : 1.5)),
      };
    });

    const res = curve(ctx), sk = sweetK(ctx), p = analysis.p;
    $('rmse-badge').textContent = compare ? `${keys.length} methods` : analysis.methods[view.method].name;
    $('r2-badge').textContent = `Peak at k=${sk}`;
    $('rmse-sub').textContent = `Lower is better. Minimum at k=${sk} (${money(ctx, res[sk - 1].test.rmse)}); full model k=${p}: ${money(ctx, res[p - 1].test.rmse)}.`;
    $('r2-sub').textContent = `Higher is better. Peaks at k=${sk} (${num(res[sk - 1].test.r2, 4)}); k=${p}: ${num(res[p - 1].test.r2, 4)}.`;

    if (typeof root.Chart === 'undefined') {
      document.querySelectorAll('.chart-box').forEach((b) => { b.innerHTML = '<p class="chart-fallback">Chart.js could not be loaded (offline?). All other sections still work.</p>'; });
      return;
    }
    const opts = (yTitle, fmt) => ({
      responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
      interaction: { mode: 'nearest', intersect: false },
      plugins: {
        legend: { display: compare, position: 'bottom', labels: { boxWidth: 18, font: { size: 11 } } },
        tooltip: { callbacks: { title: (it) => `k = ${it[0].label}`, label: (it) => `${it.dataset.label}: ${fmt(it.parsed.y)}` } },
      },
      scales: {
        x: { title: { display: true, text: 'Number of Features (k)', font: { size: 11 } }, grid: { display: false } },
        y: { title: { display: true, text: yTitle, font: { size: 11 } }, ticks: { callback: (v) => fmt(v), font: { size: 10 } }, grid: { color: '#e9eef5' } },
      },
      onClick: (e, els) => { if (els.length) cb.onK(els[0].index + 1); },
    });
    root.Chart.defaults.color = '#7b8798';
    root.Chart.defaults.font.family = "'Plus Jakarta Sans', 'Inter', system-ui, sans-serif";
    const draw = (id, metric, yTitle, fmt) => {
      if (charts[id]) charts[id].destroy();
      charts[id] = new root.Chart($(id), { type: 'line', data: { labels: ks, datasets: sets(metric) }, options: opts(yTitle, fmt) });
    };
    draw('rmse-chart', 'rmse', `Test RMSE${ctx.problem.money ? ' ($)' : ''} (Lower is Better)`, (v) => money(ctx, v, 0));
    draw('r2-chart', 'r2', 'Test R-squared (Higher is Better)', (v) => num(v, 3));
  }

  /* ---------------- equation & coefficients ---------------- */
  function renderEquation(ctx) {
    const m = activeResult(ctx).model;
    let html = `<span class="y">${esc(ctx.dataset.target)}</span> = <span class="b0">${coefStr(m.beta[0])}</span>`;
    m.cols.forEach((j, i) => {
      const b = m.beta[i + 1];
      html += ` ${b < 0 ? '−' : '+'} ${coefStr(Math.abs(b))} × (<span class="var">${esc(ctx.labels[j])}</span>)`;
    });
    $('equation').innerHTML = html;
    const name = ctx.analysis.methods[activeMethodKey(ctx)].name;
    $('equation-label').innerHTML = `<i class="fa-solid fa-square-root-variable"></i> Fitted OLS Regression Equation · k=${m.k} · ${esc(name)} selection`;
  }

  function renderCoefficients(ctx) {
    const { analysis } = ctx;
    const m = activeResult(ctx).model;
    const vifs = R.vif(analysis.X, m.cols, analysis.split.train);
    const vifBadge = (v) => {
      if (!isFinite(v)) return '<span class="badge badge-red">∞ (Singular)</span>';
      if (v < 5) return `<span class="badge badge-green">${num(v, 2)} (Safe)</span>`;
      if (v < 10) return `<span class="badge badge-amber">${num(v, 2)} (Moderate)</span>`;
      return `<span class="badge badge-red">${num(v, 2)} (High)</span>`;
    };
    const pBadge = (p) => (p < 0.05
      ? `<span class="badge badge-green">${pStr(p)} <i class="fa-solid fa-check"></i></span>`
      : `<span class="badge badge-amber">${pStr(p)} (n.s.)</span>`);
    const ciStr = (lo, hi) => {
      const d = Math.max(Math.abs(lo), Math.abs(hi)) < 0.1 ? 4 : 2;
      return `[${num(lo, d)},<wbr> ${num(hi, d)}]`;
    };
    const rows = [];
    for (let i = 0; i <= m.k; i++) {
      const name = i === 0 ? 'Intercept (β₀)' : ctx.labels[m.cols[i - 1]];
      rows.push(`<tr><td class="strong">${esc(name)}</td><td class="num strong">${tableCoef(m.beta[i])}</td><td class="num">${tableCoef(m.se[i])}</td>` +
        `<td class="num">${num(m.t[i], 3)}</td><td>${pBadge(m.p[i])}</td><td class="num ci">${ciStr(m.ci[i][0], m.ci[i][1])}</td>` +
        `<td>${i === 0 ? '–' : vifBadge(vifs[i - 1])}</td></tr>`);
    }
    $('coef-body').innerHTML = rows.join('');
    $('coef-sub').textContent = `OLS on the training set (n=${m.n}, df=${m.dfResid}). Train R² = ${num(m.r2, 4)}, Adj. R² = ${num(m.adjR2, 4)}. VIF < 5 safe, 5–10 moderate, > 10 high multicollinearity.`;
  }

  /* ---------------- live predictor ---------------- */
  const niceRound = (x) => {
    if (x === 0 || !isFinite(x)) return 0;
    const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(x))) - 1);
    return Math.round(x / mag) * mag;
  };
  function defaultInput(ctx, f) {
    const d = ctx.problem.defaults && ctx.problem.defaults[f.name];
    if (d !== undefined) return d;
    if (f.type === 'categorical') return f.reference;
    return f.integer ? Math.round(f.median) : niceRound(f.median);
  }

  function renderPredictor(ctx) {
    const m = activeResult(ctx).model;
    const active = new Set(m.cols);
    const features = ctx.features;
    $('predictor-title').textContent = `Live ${ctx.problem.predictTitle || ctx.dataset.target + ' Predictor'}`;
    let html = '';
    features.forEach((f, idx) => {
      if (!(f.name in ctx.inputs)) ctx.inputs[f.name] = defaultInput(ctx, f);
      const isActive = f.design.some((j) => active.has(j));
      const unit = (ctx.problem.units && ctx.problem.units[f.name]) || '';
      const id = `in-${idx}`;
      const inactive = isActive ? '' : ' <span class="inactive">(not in model)</span>';
      if (f.type === 'numeric') {
        const span = f.max - f.min;
        const lo = f.min >= 0 ? Math.max(0, f.min - 0.1 * span) : f.min - 0.1 * span;
        const hi = f.max + 0.1 * span;
        const step = f.integer ? 1 : Math.pow(10, Math.floor(Math.log10(span / 1000 || 1)));
        const v = ctx.inputs[f.name];
        html += `<div class="field${isActive ? '' : ' dim'}"><div class="field-head"><label for="${id}">${esc(f.name)}${unit ? ` (${esc(unit)})` : ''}${inactive}</label>` +
          `<span class="val" id="${id}-val">${valueLabel(ctx, f, v)}</span></div>` +
          `<input type="number" id="${id}" data-f="${idx}" value="${v}" step="any">` +
          `<input type="range" id="${id}-r" data-f="${idx}" min="${lo}" max="${hi}" step="${step}" value="${v}" aria-label="${esc(f.name)} slider"></div>`;
      } else {
        const opts = f.levels.map((lv) => `<option value="${esc(lv)}"${lv === ctx.inputs[f.name] ? ' selected' : ''}>${esc(lv)}${lv === f.reference ? ' (Reference)' : ''}</option>`).join('');
        html += `<div class="field${isActive ? '' : ' dim'}"><div class="field-head"><label for="${id}">${esc(f.name)}${inactive}</label></div>` +
          `<select id="${id}" data-f="${idx}">${opts}</select></div>`;
      }
    });
    const box = $('predictor-fields');
    box.innerHTML = html;
    box.oninput = (e) => {
      const idx = e.target.dataset.f;
      if (idx === undefined) return;
      const f = features[idx];
      if (f.type === 'numeric') {
        const v = Number(e.target.value);
        if (e.target.value === '' || !isFinite(v)) return;
        ctx.inputs[f.name] = v;
        const other = e.target.type === 'range' ? $(`in-${idx}`) : $(`in-${idx}-r`);
        other.value = v;
        $(`in-${idx}-val`).textContent = valueLabel(ctx, f, v);
      } else ctx.inputs[f.name] = e.target.value;
      updatePrediction(ctx);
    };
    box.onchange = box.oninput;
    updatePrediction(ctx);
  }

  function valueLabel(ctx, f, v) {
    const unit = ctx.problem.units && ctx.problem.units[f.name];
    const d = f.integer ? 0 : 2;
    return unit === '$' ? `$${num(v, d)}` : num(v, d);
  }

  function updatePrediction(ctx) {
    const m = activeResult(ctx).model;
    // design row from the current inputs
    const x = new Array(ctx.analysis.p).fill(0);
    ctx.features.forEach((f) => {
      const v = ctx.inputs[f.name];
      if (f.type === 'numeric') x[f.design[0]] = Number(v);
      else f.dummyLevels.forEach((lv, i) => { if (f.design[i] !== undefined && f.design[i] >= 0) x[f.design[i]] = v === lv ? 1 : 0; });
    });
    const pred = R.predictRow(m, x);
    $('prediction-label').textContent = ctx.problem.predictLabel || `Predicted ${ctx.dataset.target}`;
    $('prediction-value').textContent = money(ctx, pred);
    const parts = m.cols.map((j, i) => `${esc(ctx.labels[j])}: ${signedMoney(ctx, m.beta[i + 1] * x[j], 0)}`);
    $('prediction-contrib').innerHTML = `Active predictors: ${m.cols.map((j) => esc(ctx.labels[j])).join(', ')}<br>` +
      `(Intercept: ${money(ctx, m.beta[0], 0)} | ${parts.join(' | ')})`;
  }

  /* ---------------- residuals ---------------- */
  function renderResiduals(ctx) {
    const r = activeResult(ctx);
    const { analysis, dataset } = ctx;
    const thr = 1.25 * r.test.rmse;
    const rows = analysis.split.test.map((row, i) => {
      const actual = analysis.y[row], pred = r.test.preds[i], res = actual - pred;
      return { id: dataset.rowIds[row], actual, pred, res };
    }).sort((a, b) => Math.abs(b.res) - Math.abs(a.res)).slice(0, 10);
    $('resid-body').innerHTML = rows.map((o, i) => {
      const high = Math.abs(o.res) > thr;
      const pct = o.actual !== 0 ? `${num(100 * Math.abs(o.res) / Math.abs(o.actual), 2)}%` : '–';
      return `<tr><td class="strong">#${i + 1}</td><td>${esc(ctx.problem.entity)} #${o.id}</td><td class="num">${money(ctx, o.actual)}</td>` +
        `<td class="num">${money(ctx, o.pred)}</td><td class="${o.res >= 0 ? 'pos' : 'neg'}">${signedMoney(ctx, o.res)}</td>` +
        `<td class="num">${money(ctx, Math.abs(o.res))}</td><td class="num">${pct}</td>` +
        `<td>${high ? '<span class="badge badge-amber"><i class="fa-solid fa-triangle-exclamation"></i> High Residual</span>' : '<span class="badge badge-green"><i class="fa-solid fa-circle"></i> In Range</span>'}</td></tr>`;
    }).join('');
    const nTest = analysis.split.test.length;
    $('resid-sub').textContent = `Top ${rows.length} absolute residuals on the out-of-sample test set (${nTest} ${ctx.problem.entityPlural || 'rows'}). ` +
      `Residual = Actual − Predicted; High Residual when |residual| > 1.25 × RMSE (${money(ctx, thr)}). Sample ID = 0-based CSV row index.`;
    $('th-actual').textContent = `Actual ${dataset.target}`;
    $('th-pred').textContent = `Predicted ${dataset.target}`;
  }

  /* ---------------- public API ---------------- */
  function renderAll(ctx, cb) {
    $('app').classList.remove('hidden');
    renderHeader(ctx);
    renderNotes(ctx);
    renderView(ctx, cb);
  }
  /** Re-render everything that depends on the method / k selection. */
  function renderView(ctx, cb) {
    renderControls(ctx, cb);
    renderMetrics(ctx);
    renderCharts(ctx, cb);
    renderEquation(ctx);
    renderCoefficients(ctx);
    renderPredictor(ctx);
    renderResiduals(ctx);
  }

  function showOverlay(text) {
    $('overlay-text').textContent = text || 'Loading…';
    $('overlay').classList.remove('hidden');
  }
  function hideOverlay() { $('overlay').classList.add('hidden'); }
  function setOverlayText(text) { $('overlay-text').textContent = text; }

  function showStatus(kind, message, hint) {
    const icon = { error: 'fa-circle-exclamation', ok: 'fa-circle-check', warn: 'fa-triangle-exclamation' }[kind];
    $('custom-status').innerHTML = message
      ? `<div class="alert alert-${kind}"><i class="fa-solid ${icon}"></i><div>${esc(message)}${hint ? `<span class="hint">${esc(hint)}</span>` : ''}</div></div>`
      : '';
  }

  root.UI = { renderAll, renderView, showOverlay, hideOverlay, setOverlayText, showStatus, esc, num };
})(this);
