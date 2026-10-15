/*
 * featureSelection.js — five feature selection algorithms + the analysis pipeline.
 *
 * Each method returns subsets[k] (k = 1..p): the design-column indices selected
 * at k features. Every subset is then refit by plain OLS on the training set and
 * scored on the held-out test set.
 *
 *  - lasso       : L1 coordinate-descent path on standardized X; rank = order of entry
 *  - forward     : Forward Sequential Feature Selection (greedy, min training SSE)
 *  - rfe         : Recursive Feature Elimination (backward, drop smallest |t|)
 *  - elasticnet  : L1+L2 (l1_ratio = 0.5) coordinate-descent path; rank = order of entry
 *  - exhaustive  : best training-SSE subset for every k (only when p <= 8)
 */
(function (root) {
  'use strict';
  const R = typeof module !== 'undefined' && module.exports ? require('./regression.js') : root.Regression;

  const EXHAUSTIVE_MAX_P = 8;
  const METHODS = {
    lasso: { key: 'lasso', name: 'Lasso', tag: 'L1 Ranking', color: '#1e3a5f' },
    forward: { key: 'forward', name: 'Forward SFS', tag: 'Greedy', color: '#f4845f' },
    rfe: { key: 'rfe', name: 'RFE', tag: 'Backward', color: '#8b5cf6' },
    elasticnet: { key: 'elasticnet', name: 'ElasticNet', tag: 'L1 + L2', color: '#ca8a04' },
    exhaustive: { key: 'exhaustive', name: 'Exhaustive Search', tag: 'Best Subset', color: '#64748b' },
  };

  /** Standardize training columns (population SD, like sklearn StandardScaler). */
  function standardize(X, y, rows, p) {
    const n = rows.length;
    const Z = [];
    for (let j = 0; j < p; j++) {
      let m = 0;
      for (const r of rows) m += X[r][j];
      m /= n;
      let v = 0;
      for (const r of rows) v += (X[r][j] - m) ** 2;
      const sd = Math.sqrt(v / n) || 1;
      Z.push(Float64Array.from(rows, (r) => (X[r][j] - m) / sd));
    }
    let ym = 0;
    for (const r of rows) ym += y[r];
    ym /= n;
    const yc = Float64Array.from(rows, (r) => y[r] - ym);
    return { Z, yc, n };
  }

  /**
   * Elastic-net path by cyclic coordinate descent (sklearn objective):
   *   1/(2n)||y - Zb||² + λ·α·|b|₁ + λ·(1-α)/2·||b||²
   * Returns the feature ranking by order of entry into the active set.
   */
  function penalizedRanking(X, y, rows, p, l1Ratio) {
    const { Z, yc, n } = standardize(X, y, rows, p);
    const resid = Float64Array.from(yc);
    const b = new Float64Array(p);
    let lamMax = 0;
    for (let j = 0; j < p; j++) {
      let g = 0;
      for (let i = 0; i < n; i++) g += Z[j][i] * yc[i];
      lamMax = Math.max(lamMax, Math.abs(g) / (n * l1Ratio));
    }
    const nLambda = 300, eps = 1e-5;
    const entry = new Array(p).fill(Infinity);
    const entryMag = new Array(p).fill(0);
    const yScale = Math.sqrt(yc.reduce((s, v) => s + v * v, 0) / n) || 1;
    for (let step = 0; step < nLambda; step++) {
      const lam = lamMax * Math.pow(eps, step / (nLambda - 1));
      const l1 = lam * l1Ratio, l2 = lam * (1 - l1Ratio);
      for (let iter = 0; iter < 2000; iter++) {
        let maxDelta = 0;
        for (let j = 0; j < p; j++) {
          const zj = Z[j];
          let rho = 0;
          for (let i = 0; i < n; i++) rho += zj[i] * resid[i];
          rho = rho / n + b[j]; // columns have unit variance
          const nb = Math.sign(rho) * Math.max(Math.abs(rho) - l1, 0) / (1 + l2);
          const d = nb - b[j];
          if (d !== 0) {
            for (let i = 0; i < n; i++) resid[i] -= d * zj[i];
            b[j] = nb;
            maxDelta = Math.max(maxDelta, Math.abs(d));
          }
        }
        if (maxDelta < 1e-9 * yScale) break;
      }
      for (let j = 0; j < p; j++) {
        if (entry[j] === Infinity && b[j] !== 0) { entry[j] = step; entryMag[j] = Math.abs(b[j]); }
      }
    }
    const finalMag = Array.from(b, Math.abs);
    return Array.from({ length: p }, (_, j) => j).sort((a, c) =>
      entry[a] - entry[c] || entryMag[c] - entryMag[a] || finalMag[c] - finalMag[a]);
  }

  const nested = (order) => order.map((_, i) => order.slice(0, i + 1));

  function forwardSFS(X, y, rows, p) {
    const chosen = [], subsets = [];
    const remaining = new Set(Array.from({ length: p }, (_, j) => j));
    while (remaining.size) {
      let best = null, bestSSE = Infinity;
      for (const j of remaining) {
        const sse = safeSSE(X, y, chosen.concat(j), rows);
        if (sse < bestSSE) { bestSSE = sse; best = j; }
      }
      if (best === null) best = remaining.values().next().value;
      chosen.push(best); remaining.delete(best);
      subsets.push(chosen.slice());
    }
    return subsets;
  }

  function rfe(X, y, rows, p) {
    let active = Array.from({ length: p }, (_, j) => j);
    const subsets = [];
    subsets[p - 1] = active.slice();
    while (active.length > 1) {
      const m = R.ols(X, y, active, rows);
      let worst = 0;
      for (let i = 1; i < active.length; i++) if (Math.abs(m.t[i + 1]) < Math.abs(m.t[worst + 1])) worst = i;
      active = active.filter((_, i) => i !== worst);
      subsets[active.length - 1] = active.slice();
    }
    return subsets;
  }

  function exhaustive(X, y, rows, p) {
    const best = new Array(p).fill(null).map(() => ({ sse: Infinity, cols: null }));
    for (let mask = 1; mask < 1 << p; mask++) {
      const cols = [];
      for (let j = 0; j < p; j++) if (mask & (1 << j)) cols.push(j);
      const sse = safeSSE(X, y, cols, rows);
      if (sse < best[cols.length - 1].sse) best[cols.length - 1] = { sse, cols };
    }
    return best.map((b) => b.cols);
  }

  function safeSSE(X, y, cols, rows) {
    try { return R.ols(X, y, cols, rows).sse; } catch (e) { return Infinity; }
  }

  /**
   * Drop design columns that are (numerically) exact linear combinations of
   * earlier ones on the training rows, so that the full model is estimable.
   */
  function findCollinear(X, rows, p) {
    const basis = [], dropped = [];
    for (let j = 0; j < p; j++) {
      let v = Float64Array.from(rows, (r) => X[r][j]);
      const mean = v.reduce((s, x) => s + x, 0) / v.length;
      v = v.map((x) => x - mean);
      const norm0 = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      for (const q of basis) {
        const d = v.reduce((s, x, i) => s + x * q[i], 0);
        v = v.map((x, i) => x - d * q[i]);
      }
      const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      if (norm0 === 0 || norm / norm0 < 1e-8) dropped.push(j);
      else basis.push(v.map((x) => x / norm));
    }
    return dropped;
  }

  const nextFrame = () => new Promise((resolve) => {
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => resolve());
    else setTimeout(resolve, 0);
  });

  /**
   * Full pipeline: split -> (collinearity check) -> 5 methods -> OLS refit per k -> test metrics.
   * Async so that large datasets yield to the browser between steps (keeps the UI responsive).
   */
  async function runAnalysis(dataset, opts) {
    opts = Object.assign({ testSize: 0.2, seed: 0, onProgress: null }, opts);
    let { X, y, designNames } = dataset;
    const notes = [];
    const n = y.length;
    const split = R.trainTestSplit(n, opts.testSize, opts.seed);
    const big = n > 5000;
    const tick = async (msg) => { if (opts.onProgress) opts.onProgress(msg); if (big || opts.onProgress) await nextFrame(); };

    const collinear = findCollinear(X, split.train, designNames.length);
    if (collinear.length) {
      notes.push(`Removed perfectly collinear column(s): ${collinear.map((j) => designNames[j]).join(', ')} (XᵀX would be singular).`);
      const keep = designNames.map((_, j) => j).filter((j) => !collinear.includes(j));
      X = X.map((r) => keep.map((j) => r[j]));
      designNames = keep.map((j) => designNames[j]);
    }
    // old design index -> new index (-1 when removed), used to remap dataset.features
    const designMap = dataset.designNames.map((_, j) => (collinear.includes(j) ? -1 : j - collinear.filter((c) => c < j).length));
    const p = designNames.length;
    if (p < 1) throw new Error('No estimable predictors remain after removing collinear columns.');

    const methodKeys = ['lasso', 'forward', 'rfe', 'elasticnet'];
    if (p <= EXHAUSTIVE_MAX_P) methodKeys.push('exhaustive');
    else notes.push(`Exhaustive Search disabled: p = ${p} > ${EXHAUSTIVE_MAX_P} predictors (2^${p} subsets).`);

    const fitCache = new Map();
    const fitSubset = (cols) => {
      const key = cols.slice().sort((a, b) => a - b).join(',');
      if (!fitCache.has(key)) {
        const model = R.ols(X, y, cols, split.train);
        const test = R.evaluate(model, X, y, split.test);
        fitCache.set(key, { model, test });
      }
      return fitCache.get(key);
    };

    const methods = {};
    for (const key of methodKeys) {
      await tick(`Running ${METHODS[key].name}…`);
      let subsets;
      if (key === 'lasso') subsets = nested(penalizedRanking(X, y, split.train, p, 1));
      else if (key === 'elasticnet') subsets = nested(penalizedRanking(X, y, split.train, p, 0.5));
      else if (key === 'forward') subsets = forwardSFS(X, y, split.train, p);
      else if (key === 'rfe') subsets = rfe(X, y, split.train, p);
      else subsets = exhaustive(X, y, split.train, p);

      const results = subsets.map((cols, i) => {
        const prev = i ? subsets[i - 1] : [];
        const added = cols.filter((c) => !prev.includes(c));
        return Object.assign({ k: i + 1, cols, added }, fitSubset(cols));
      });
      let best = 0;
      results.forEach((r, i) => { if (r.test.rmse < results[best].test.rmse - 1e-9) best = i; });
      methods[key] = Object.assign({}, METHODS[key], { results, bestK: best + 1 });
    }

    let globalBest = null;
    for (const key of methodKeys) {
      const m = methods[key];
      const r = m.results[m.bestK - 1];
      if (!globalBest || r.test.rmse < globalBest.rmse - 1e-9 || (Math.abs(r.test.rmse - globalBest.rmse) < 1e-9 && r.k < globalBest.k)) {
        globalBest = { method: key, k: r.k, rmse: r.test.rmse, r2: r.test.r2 };
      }
    }
    const full = methods[methodKeys[0]].results[p - 1];
    return {
      X, y, designNames, designMap, p, n, split, methodKeys, methods, globalBest, notes,
      full: { k: p, rmse: full.test.rmse, r2: full.test.r2, overfit: full.test.rmse > globalBest.rmse * 1.01 },
    };
  }

  const api = { METHODS, EXHAUSTIVE_MAX_P, penalizedRanking, forwardSFS, rfe, exhaustive, findCollinear, runAnalysis };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.FeatureSelection = api;
})(typeof self !== 'undefined' ? self : this);
