/*
 * app.js — main flow: problem switching (Problem 1 / Problem 2 / Custom),
 * dataset loading with local fallback, custom CSV URL handling and errors.
 */
(function (root) {
  'use strict';
  const { CSV, FeatureSelection, UI } = root;
  const $ = (id) => document.getElementById(id);

  // Style reference link (StatsUnlock). Change here if the URL moves.
  const STYLE_REF_URL = 'https://www.statsunlock.com/';

  const PROBLEMS = {
    p1: {
      id: 'p1',
      label: 'Problem 1 · 50 Startups → Profit',
      title: '50 Startups',
      badge: '50 Startups Dataset',
      url: 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv',
      fallback: 'data/50_Startups.csv',
      target: 'Profit',
      description: 'Predict the annual Profit of 50 US startups from R&D Spend, Administration, Marketing Spend and State (one-hot, California = reference).',
      sourceName: '50 Startups Dataset (Avik Jain 100-Days-Of-ML-Code)',
      sourceUrl: 'https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/datasets/50_Startups.csv',
      money: true,
      entity: 'Startup', entityPlural: 'startups',
      predictTitle: 'Profit Predictor', predictLabel: 'Predicted Startup Profit',
      units: { 'R&D Spend': '$', Administration: '$', 'Marketing Spend': '$' },
      defaults: { 'R&D Spend': 100000, 'Marketing Spend': 150000, Administration: 120000, State: 'California' },
    },
    p2: {
      id: 'p2',
      label: 'Problem 2 · Insurance → Charges',
      title: 'Medical Insurance',
      badge: 'Insurance Dataset',
      url: 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv',
      fallback: 'data/insurance.csv',
      target: 'charges',
      description: 'Predict individual medical insurance charges of 1,338 US policyholders from age, sex, BMI, number of children, smoker status and region (one-hot, drop first).',
      sourceName: 'Medical Cost Personal Dataset (Machine Learning with R, Brett Lantz — stedy/Machine-Learning-with-R-datasets)',
      sourceUrl: 'https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv',
      money: true,
      entity: 'Policyholder', entityPlural: 'policyholders',
      predictTitle: 'Insurance Charges Predictor', predictLabel: 'Predicted Annual Charges',
      units: { age: 'years', bmi: 'kg/m²' },
      defaults: { age: 40, bmi: 30, children: 1, sex: 'female', smoker: 'no', region: 'northeast' },
    },
  };

  const state = {
    ctx: null,          // context currently shown
    cache: {},          // problem id -> ctx
    custom: null,       // { url, parsed, columns, fileName }
    token: 0,           // guards against out-of-order async results
    selected: 'p1',
  };

  const callbacks = {
    onMethod(method) {
      const ctx = state.ctx;
      ctx.view.method = method;
      ctx.view.k = method === 'compare' ? ctx.analysis.globalBest.k : ctx.analysis.methods[method].bestK;
      UI.renderView(ctx, callbacks);
    },
    onK(k) {
      state.ctx.view.k = k;
      UI.renderView(state.ctx, callbacks);
    },
  };

  /* ---------------- fetching ---------------- */
  async function fetchText(url) {
    const ctrl = typeof AbortController !== 'undefined' ? new AbortController() : null;
    const timer = ctrl && setTimeout(() => ctrl.abort(), 20000);
    let res;
    try {
      res = await fetch(url, ctrl ? { signal: ctrl.signal } : undefined);
    } catch (e) {
      if (e && e.name === 'AbortError') throw new CSV.DatasetError('The request timed out after 20 seconds.');
      throw new CSV.DatasetError('Could not fetch the URL — network error, or the server does not allow cross-origin (CORS) requests.',
        'Use a CORS-enabled host such as raw.githubusercontent.com or gist.githubusercontent.com.');
    } finally {
      if (timer) clearTimeout(timer);
    }
    if (!res.ok) {
      throw new CSV.DatasetError(`HTTP ${res.status} ${res.statusText || ''}`.trim() + (res.status === 404 ? ' — file not found.' : ' — the server refused the request.'),
        'Check that the URL is correct and publicly accessible.');
    }
    const len = Number(res.headers.get('content-length'));
    if (len > 50 * 1024 * 1024) throw new CSV.DatasetError('The file is larger than 50 MB.', 'Use a smaller CSV for in-browser analysis.');
    const type = (res.headers.get('content-type') || '').toLowerCase();
    if (type.includes('text/html')) {
      throw new CSV.DatasetError('The URL returned an HTML page, not CSV.', 'For GitHub, open the file and click "Raw" to get a raw.githubusercontent.com link.');
    }
    return res.text();
  }

  /** github.com/u/r/blob/branch/file.csv -> raw.githubusercontent.com/u/r/branch/file.csv */
  function normalizeUrl(input) {
    let u;
    try { u = new URL(input.trim()); } catch (e) {
      throw new CSV.DatasetError('That is not a valid URL.', 'Example: https://raw.githubusercontent.com/user/repo/main/data.csv');
    }
    if (!/^https?:$/.test(u.protocol)) throw new CSV.DatasetError('Only http(s) URLs are supported.');
    const m = u.hostname === 'github.com' && u.pathname.match(/^\/([^/]+)\/([^/]+)\/blob\/(.+)$/);
    if (m) return `https://raw.githubusercontent.com/${m[1]}/${m[2]}/${m[3]}`;
    return u.href;
  }

  /* ---------------- analysis → context ---------------- */
  async function analyze(problem, dataset) {
    const analysis = await FeatureSelection.runAnalysis(dataset, { onProgress: (msg) => UI.setOverlayText(msg) });
    // remap design indices if collinear columns were removed, and build display labels
    const map = analysis.designMap;
    const features = [];
    const labels = new Array(analysis.p);
    dataset.features.forEach((f) => {
      const design = f.design.map((j) => map[j]);
      if (design.every((j) => j < 0)) return;
      const nf = Object.assign({}, f, { design });
      features.push(nf);
      if (f.type === 'numeric') labels[design[0]] = f.name;
      else f.dummyLevels.forEach((lv, i) => { if (design[i] >= 0) labels[design[i]] = `${f.name}: ${lv}`; });
    });
    const view = { method: 'lasso', k: analysis.methods.lasso.bestK };
    return { problem, dataset, analysis, features, labels, view, inputs: {} };
  }

  function show(ctx) {
    state.ctx = ctx;
    UI.renderAll(ctx, callbacks);
  }

  /* ---------------- built-in problems ---------------- */
  async function loadProblem(id) {
    const problem = PROBLEMS[id];
    if (state.cache[id]) { show(state.cache[id]); return; }
    const token = ++state.token;
    UI.showOverlay(`Loading ${problem.title}…`);
    try {
      let text, from = 'GitHub';
      try {
        text = await fetchText(problem.url);
        CSV.parseCSV(text);
      } catch (e) {
        console.warn(`Remote fetch failed for ${problem.url}, using local backup:`, e.message);
        text = await fetchText(problem.fallback);
        from = 'local backup (data/)';
      }
      const dataset = CSV.buildDataset(CSV.parseCSV(text), problem.target);
      dataset.loadedFrom = from;
      const ctx = await analyze(problem, dataset);
      state.cache[id] = ctx;
      if (token === state.token) show(ctx);
    } catch (e) {
      if (token === state.token) {
        UI.showStatus('error', `Could not load ${problem.title}: ${e.message}`,
          location.protocol === 'file:' ? 'Open the site through a local web server (python -m http.server), not file://.' : e.hint);
      }
      console.warn(e);
    } finally {
      if (token === state.token) UI.hideOverlay();
    }
  }

  /* ---------------- custom dataset ---------------- */
  async function loadCustom(rawUrl) {
    const btn = $('custom-load');
    const token = ++state.token;
    btn.disabled = true;
    UI.showStatus('', '');
    UI.showOverlay('Fetching CSV…');
    try {
      const url = normalizeUrl(rawUrl);
      const text = await fetchText(url);
      UI.setOverlayText('Parsing CSV…');
      const parsed = CSV.parseCSV(text);
      const columns = CSV.inferColumns(parsed);
      if (parsed.rows.length < 10) {
        throw new CSV.DatasetError(`Only ${parsed.rows.length} data row(s) — too few for an 80/20 train/test split.`, 'At least 10 complete rows are required.');
      }
      const numeric = columns.filter((c) => c.type === 'numeric');
      const usable = columns.filter((c) => c.type === 'numeric' || c.type === 'categorical');
      if (!numeric.length) {
        throw new CSV.DatasetError('No numeric column found — a numeric target y is required.',
          'The file may not be a data table, or numbers may use an unsupported format (e.g. "1,234" or "$100").');
      }
      if (usable.length < 2) {
        throw new CSV.DatasetError(`Only ${usable.length} usable column found (need a target plus at least one predictor).`,
          columns.filter((c) => c.type === 'ignored').map((c) => `"${c.name}": ${c.reason}`).join('; '));
      }
      if (token !== state.token) return;
      const fileName = decodeURIComponent(new URL(url).pathname.split('/').pop() || 'dataset.csv');
      state.custom = { url, parsed, columns, fileName };

      const sel = $('target-select');
      sel.innerHTML = numeric.map((c) => `<option value="${UI.esc(c.name)}">${UI.esc(c.name)}</option>`).join('');
      sel.value = numeric[numeric.length - 1].name;
      $('target-row').classList.remove('hidden');
      const cat = columns.filter((c) => c.type === 'categorical');
      const ign = columns.filter((c) => c.type === 'ignored');
      $('col-summary').textContent = `${parsed.rows.length} rows · ${numeric.length} numeric · ${cat.length} categorical` +
        (cat.length ? ` (${cat.map((c) => c.name).join(', ')})` : '') + (ign.length ? ` · ${ign.length} ignored` : '');

      await runCustom(sel.value, token);
    } catch (e) {
      if (token === state.token) UI.showStatus('error', e.message || String(e), e.hint);
      console.warn('Custom dataset failed:', e);
    } finally {
      btn.disabled = false;
      if (token === state.token) UI.hideOverlay();
    }
  }

  async function runCustom(target, token) {
    token = token || ++state.token;
    const c = state.custom;
    UI.showOverlay('Running analysis…');
    try {
      const dataset = CSV.buildDataset(c.parsed, target, c.columns);
      const problem = {
        id: 'custom',
        title: c.fileName.replace(/\.(csv|txt)$/i, ''),
        badge: c.fileName,
        description: `Custom dataset loaded from URL: predict ${target} from ${dataset.features.length} column(s).`,
        sourceName: c.url, sourceUrl: c.url,
        money: false, entity: 'Row', entityPlural: 'rows', target,
      };
      const ctx = await analyze(problem, dataset);
      if (token !== state.token) return;
      state.cache.custom = ctx;
      ensureCustomOption(problem.title);
      $('problem-select').value = 'custom';
      state.selected = 'custom';
      show(ctx);
      UI.showStatus('ok', `Loaded ${c.fileName}: ${dataset.y.length} complete rows, ${ctx.analysis.p} predictors after one-hot encoding, target = ${target}.`);
    } catch (e) {
      if (token === state.token) UI.showStatus('error', e.message || String(e), e.hint);
      console.warn('Custom analysis failed:', e);
    } finally {
      if (token === state.token) UI.hideOverlay();
    }
  }

  function ensureCustomOption(title) {
    const sel = $('problem-select');
    let opt = sel.querySelector('option[value="custom"]');
    if (!opt) { opt = document.createElement('option'); opt.value = 'custom'; sel.appendChild(opt); }
    opt.textContent = `Custom · ${title}`;
  }

  /* ---------------- init ---------------- */
  function init() {
    $('style-ref').href = STYLE_REF_URL;
    $('footer-style-ref').href = STYLE_REF_URL;

    const sel = $('problem-select');
    sel.innerHTML = Object.values(PROBLEMS).map((p) => `<option value="${p.id}">${UI.esc(p.label)}</option>`).join('');
    sel.addEventListener('change', () => {
      state.selected = sel.value;
      UI.showStatus('', '');
      if (sel.value === 'custom') { if (state.cache.custom) show(state.cache.custom); }
      else loadProblem(sel.value);
    });

    const examples = [
      ['50_Startups.csv', PROBLEMS.p1.url],
      ['insurance.csv', PROBLEMS.p2.url],
      ['Boston housing', 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'],
    ];
    examples.forEach(([name, url]) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'chip'; b.textContent = name; b.title = url;
      b.addEventListener('click', () => { $('custom-url').value = url; loadCustom(url); });
      $('examples').appendChild(b);
    });

    $('custom-form').addEventListener('submit', (e) => { e.preventDefault(); loadCustom($('custom-url').value); });
    $('target-select').addEventListener('change', (e) => { if (state.custom) runCustom(e.target.value); });

    window.addEventListener('unhandledrejection', (e) => {
      UI.hideOverlay();
      UI.showStatus('error', `Unexpected error: ${e.reason && e.reason.message ? e.reason.message : e.reason}`);
    });

    loadProblem('p1');
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})(this);
