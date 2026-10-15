/*
 * regression.js — OLS and inferential statistics, implemented from scratch.
 *
 *  - Reproducible train/test split: MT19937 + NumPy's legacy shuffle, so
 *    trainTestSplit(n, 0.2, 0) gives exactly the same split as
 *    sklearn.model_selection.train_test_split(..., test_size=0.2, random_state=0).
 *  - OLS: beta = (X'X)^-1 X'y via Cholesky, with a partial-pivot Gauss-Jordan fallback.
 *  - Distributions: t and F via the regularized incomplete beta function.
 *  - Metrics: R², adjusted R², RMSE, MAE, F + p-value, t + p-value, 95% CI, VIF.
 *
 * Works both in the browser (window.Regression) and in Node (module.exports).
 */
(function (root) {
  'use strict';

  /* ------------------------------------------------------------------ *
   * Random numbers: MT19937 compatible with numpy.random.RandomState
   * ------------------------------------------------------------------ */
  function MT19937(seed) {
    this.mt = new Uint32Array(624);
    this.mti = 624;
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const prev = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      // 1812433253 * prev + i, computed in 32-bit pieces to avoid precision loss
      this.mt[i] = ((((prev & 0xffff0000) >>> 16) * 1812433253) << 16) + (prev & 0x0000ffff) * 1812433253 + i;
    }
  }
  MT19937.prototype.nextUint32 = function () {
    const mt = this.mt;
    if (this.mti >= 624) {
      for (let k = 0; k < 624; k++) {
        const y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7fffffff);
        mt[k] = mt[(k + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      }
      this.mti = 0;
    }
    let y = mt[this.mti++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  };
  // numpy legacy random_interval(max): rejection sampling with a bit mask
  MT19937.prototype.interval = function (max) {
    if (max === 0) return 0;
    let mask = max;
    mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4;
    mask |= mask >>> 8; mask |= mask >>> 16;
    mask >>>= 0;
    let v;
    while ((v = (this.nextUint32() & mask) >>> 0) > max);
    return v;
  };

  /** np.random.RandomState(seed).permutation(n) */
  function permutation(n, seed) {
    const rng = new MT19937(seed);
    const arr = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i >= 1; i--) {
      const j = rng.interval(i);
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }

  /** Same index sets as sklearn train_test_split(test_size, random_state=seed). */
  function trainTestSplit(n, testSize, seed) {
    const nTest = Math.ceil(testSize * n);
    const nTrain = Math.floor((1 - testSize) * n + 1e-9);
    const perm = permutation(n, seed);
    return { test: perm.slice(0, nTest), train: perm.slice(nTest, nTest + nTrain) };
  }

  /* ------------------------------------------------------------------ *
   * Linear algebra
   * ------------------------------------------------------------------ */
  /** Inverse of a symmetric positive definite matrix via Cholesky; null if not SPD. */
  function choleskyInverse(A) {
    const n = A.length;
    const L = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) {
          if (!(s > 1e-12 * Math.max(1, Math.abs(A[i][i])))) return null;
          L[i][i] = Math.sqrt(s);
        } else {
          L[i][j] = s / L[j][j];
        }
      }
    }
    // inv(L), then inv(A) = inv(L)' inv(L)
    const Li = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      Li[i][i] = 1 / L[i][i];
      for (let j = 0; j < i; j++) {
        let s = 0;
        for (let k = j; k < i; k++) s -= L[i][k] * Li[k][j];
        Li[i][j] = s / L[i][i];
      }
    }
    const inv = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = 0;
        for (let k = i; k < n; k++) s += Li[k][i] * Li[k][j];
        inv[i][j] = s; inv[j][i] = s;
      }
    }
    return inv;
  }

  /** Gauss-Jordan inverse with partial pivoting; throws SingularMatrixError. */
  function gaussInverse(A) {
    const n = A.length;
    const M = A.map((row, i) => {
      const r = Array.from(row);
      for (let j = 0; j < n; j++) r.push(i === j ? 1 : 0);
      return r;
    });
    const scale = Math.max(...A.map((r, i) => Math.abs(r[i])), 1e-300);
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      if (Math.abs(M[piv][c]) < 1e-10 * scale) throw new SingularMatrixError(c);
      [M[c], M[piv]] = [M[piv], M[c]];
      const d = M[c][c];
      for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
      for (let r = 0; r < n; r++) {
        if (r === c || M[r][c] === 0) continue;
        const f = M[r][c];
        for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
      }
    }
    return M.map((r) => r.slice(n));
  }

  function SingularMatrixError(column) {
    this.name = 'SingularMatrixError';
    this.column = column;
    this.message = 'XᵀX is singular (perfect multicollinearity) — at least one predictor is an exact linear combination of the others.';
  }
  SingularMatrixError.prototype = Object.create(Error.prototype);

  function invertSPD(A) {
    return choleskyInverse(A) || gaussInverse(A);
  }

  /* ------------------------------------------------------------------ *
   * Special functions & distributions
   * ------------------------------------------------------------------ */
  const LANCZOS = [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  function lnGamma(x) {
    if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lnGamma(1 - x);
    x -= 1;
    let a = 0.99999999999980993;
    const t = x + 7.5;
    for (let i = 0; i < 8; i++) a += LANCZOS[i] / (x + i + 1);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  // Continued fraction for the incomplete beta (modified Lentz, Numerical Recipes betacf)
  function betacf(a, b, x) {
    const MAXIT = 500, EPS = 1e-15, FPMIN = 1e-300;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }

  /** Regularized incomplete beta I_x(a, b). */
  function ibeta(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const lbt = lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * Math.log(x) + b * Math.log(1 - x);
    if (x < (a + 1) / (a + b + 2)) return Math.exp(lbt) * betacf(a, b, x) / a;
    return 1 - Math.exp(lbt) * betacf(b, a, 1 - x) / b;
  }

  /** Two-sided p-value of a t statistic with df degrees of freedom. */
  function tTwoSidedP(t, df) {
    if (!isFinite(t)) return isNaN(t) ? NaN : 0;
    return ibeta(df / (df + t * t), df / 2, 0.5);
  }

  /** CDF of Student's t. */
  function tCdf(t, df) {
    const p = 0.5 * tTwoSidedP(t, df);
    return t >= 0 ? 1 - p : p;
  }

  /** Quantile of Student's t (bisection on the CDF). */
  function tInv(p, df) {
    let lo = -1e3, hi = 1e3;
    for (let i = 0; i < 200; i++) {
      const mid = (lo + hi) / 2;
      if (tCdf(mid, df) < p) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  /** Upper tail P(F > f) for F(d1, d2). */
  function fSurvival(f, d1, d2) {
    if (!(f > 0)) return 1;
    if (!isFinite(f)) return 0;
    return ibeta(d2 / (d2 + d1 * f), d2 / 2, d1 / 2);
  }

  /* ------------------------------------------------------------------ *
   * OLS
   * ------------------------------------------------------------------ */
  /**
   * Fit y = b0 + sum b_j x_j by OLS.
   * @param {number[][]} X  rows of the full design matrix (no intercept column)
   * @param {number[]}   y
   * @param {number[]}   cols  column indices of X to use
   * @param {number[]}   [rows] row indices to use (default: all)
   */
  function ols(X, y, cols, rows) {
    rows = rows || X.map((_, i) => i);
    const n = rows.length, k = cols.length, m = k + 1;
    const XtX = Array.from({ length: m }, () => new Array(m).fill(0));
    const Xty = new Array(m).fill(0);
    const xr = new Float64Array(m);
    let ySum = 0;
    for (const r of rows) {
      xr[0] = 1;
      for (let j = 0; j < k; j++) xr[j + 1] = X[r][cols[j]];
      const yr = y[r];
      ySum += yr;
      for (let a = 0; a < m; a++) {
        Xty[a] += xr[a] * yr;
        for (let b = 0; b <= a; b++) XtX[a][b] += xr[a] * xr[b];
      }
    }
    for (let a = 0; a < m; a++) for (let b = 0; b < a; b++) XtX[b][a] = XtX[a][b];

    const inv = invertSPD(XtX);
    const beta = inv.map((row) => row.reduce((s, v, j) => s + v * Xty[j], 0));

    const yMean = ySum / n;
    let sse = 0, sst = 0;
    for (const r of rows) {
      let pred = beta[0];
      for (let j = 0; j < k; j++) pred += beta[j + 1] * X[r][cols[j]];
      sse += (y[r] - pred) ** 2;
      sst += (y[r] - yMean) ** 2;
    }
    const dfResid = n - m;
    const sigma2 = dfResid > 0 ? sse / dfResid : NaN;
    const se = inv.map((row, i) => Math.sqrt(Math.max(row[i], 0) * sigma2));
    const tStat = beta.map((b, i) => b / se[i]);
    const pValues = tStat.map((t) => (dfResid > 0 ? tTwoSidedP(Math.abs(t), dfResid) : NaN));
    const tCrit = dfResid > 0 ? tInv(0.975, dfResid) : NaN;
    const ci = beta.map((b, i) => [b - tCrit * se[i], b + tCrit * se[i]]);
    const r2 = sst > 0 ? 1 - sse / sst : NaN;
    const adjR2 = dfResid > 0 ? 1 - (1 - r2) * (n - 1) / dfResid : NaN;
    const F = k > 0 && dfResid > 0 ? ((sst - sse) / k) / (sse / dfResid) : NaN;
    const Fp = isNaN(F) ? NaN : fSurvival(F, k, dfResid);

    return { cols: cols.slice(), n, k, beta, se, t: tStat, p: pValues, ci, sse, sst, r2, adjR2, F, Fp, dfResid, sigma2 };
  }

  function predictRow(model, xRow) {
    let s = model.beta[0];
    for (let j = 0; j < model.k; j++) s += model.beta[j + 1] * xRow[model.cols[j]];
    return s;
  }

  /** Out-of-sample metrics; adjusted R² uses the test-set n and the model's k. */
  function evaluate(model, X, y, rows) {
    const n = rows.length;
    let mean = 0;
    for (const r of rows) mean += y[r];
    mean /= n;
    let sse = 0, sae = 0, sst = 0;
    const preds = rows.map((r) => {
      const p = predictRow(model, X[r]);
      sse += (y[r] - p) ** 2;
      sae += Math.abs(y[r] - p);
      sst += (y[r] - mean) ** 2;
      return p;
    });
    const r2 = sst > 0 ? 1 - sse / sst : NaN;
    const dof = n - model.k - 1;
    return {
      rmse: Math.sqrt(sse / n),
      mae: sae / n,
      r2,
      adjR2: dof > 0 ? 1 - (1 - r2) * (n - 1) / dof : NaN,
      preds,
    };
  }

  /** Variance inflation factor of each column in cols (computed on rows). */
  function vif(X, cols, rows) {
    if (cols.length < 2) return cols.map(() => 1);
    return cols.map((c, i) => {
      const others = cols.filter((_, j) => j !== i);
      try {
        const m = ols(X, X.map((r) => r[c]), others, rows);
        return m.r2 >= 1 ? Infinity : 1 / (1 - m.r2);
      } catch (e) {
        return Infinity;
      }
    });
  }

  const api = {
    MT19937, permutation, trainTestSplit,
    choleskyInverse, gaussInverse, invertSPD, SingularMatrixError,
    lnGamma, ibeta, tCdf, tInv, tTwoSidedP, fSurvival,
    ols, predictRow, evaluate, vif,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Regression = api;
})(typeof self !== 'undefined' ? self : this);
