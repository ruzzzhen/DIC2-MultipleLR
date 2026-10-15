/*
 * csv.js — CSV parsing, column type detection and one-hot encoding.
 *
 * parseCSV(text)            -> { headers, rows }            (throws DatasetError)
 * inferColumns(parsed)      -> [{ name, type, ... }]
 * buildDataset(parsed, target) -> design matrix ready for regression (throws DatasetError)
 */
(function (root) {
  'use strict';

  function DatasetError(message, hint) {
    this.name = 'DatasetError';
    this.message = message;
    this.hint = hint || '';
  }
  DatasetError.prototype = Object.create(Error.prototype);

  const MISSING = new Set(['', 'na', 'n/a', 'nan', 'null', 'none', '?', '#n/a']);
  const isMissing = (v) => MISSING.has(String(v).trim().toLowerCase());
  const toNumber = (v) => {
    const s = String(v).trim();
    if (s === '' || !/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(s)) return NaN;
    return Number(s);
  };

  function detectDelimiter(firstLine) {
    let best = ',', bestCount = 0;
    for (const d of [',', ';', '\t', '|']) {
      let count = 0, inQ = false;
      for (const ch of firstLine) {
        if (ch === '"') inQ = !inQ;
        else if (ch === d && !inQ) count++;
      }
      if (count > bestCount) { best = d; bestCount = count; }
    }
    return best;
  }

  /** RFC 4180 style parser (quoted fields, escaped quotes, CRLF, BOM). */
  function parseCSV(text) {
    if (typeof text !== 'string' || !text.trim()) {
      throw new DatasetError('The file is empty.', 'Check that the URL points to a CSV file with a header row and data.');
    }
    text = text.replace(/^﻿/, '');
    const head = text.slice(0, 500).trim().toLowerCase();
    if (head.startsWith('<!doctype') || head.startsWith('<html') || head.startsWith('<?xml') || head.startsWith('{') || head.startsWith('[')) {
      throw new DatasetError('The response is not CSV (it looks like HTML/JSON/XML).',
        'For GitHub files use the "Raw" link (raw.githubusercontent.com/...), not the github.com page.');
    }
    const firstLine = text.slice(0, text.search(/\r?\n|$/));
    const delim = detectDelimiter(firstLine);

    const records = [];
    let field = '', row = [], inQuotes = false, i = 0;
    while (i < text.length) {
      const ch = text[i];
      if (inQuotes) {
        if (ch === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; } else inQuotes = false;
        } else field += ch;
      } else if (ch === '"') inQuotes = true;
      else if (ch === delim) { row.push(field); field = ''; }
      else if (ch === '\n' || ch === '\r') {
        row.push(field); field = '';
        records.push(row); row = [];
        if (ch === '\r' && text[i + 1] === '\n') i++;
      } else field += ch;
      i++;
    }
    if (inQuotes) throw new DatasetError('Malformed CSV: a quoted field is never closed.');
    if (field !== '' || row.length) { row.push(field); records.push(row); }

    const nonEmpty = records.filter((r) => !(r.length === 1 && r[0].trim() === ''));
    if (nonEmpty.length < 2) throw new DatasetError('The CSV has a header but no data rows.');
    const headers = nonEmpty[0].map((h, j) => h.trim() || `Column ${j + 1}`);
    if (headers.length < 2) {
      throw new DatasetError(`The CSV has only one column ("${headers[0]}").`,
        'Multiple regression needs a numeric target plus at least one predictor column.');
    }
    const seen = new Set();
    headers.forEach((h, j) => {
      let name = h, s = 2;
      while (seen.has(name)) name = `${h} (${s++})`;
      headers[j] = name; seen.add(name);
    });
    const rows = nonEmpty.slice(1);
    rows.forEach((r, idx) => {
      if (r.length !== headers.length) {
        throw new DatasetError(`Malformed CSV: line ${idx + 2} has ${r.length} fields but the header has ${headers.length}.`,
          'Every row must have the same number of comma-separated values as the header.');
      }
    });
    return { headers, rows, delimiter: delim };
  }

  /** Classify every column as numeric / categorical / ignored. */
  function inferColumns(parsed) {
    const { headers, rows } = parsed;
    const n = rows.length;
    return headers.map((name, j) => {
      let missing = 0, numeric = 0;
      const distinct = new Set();
      for (const r of rows) {
        const v = r[j];
        if (isMissing(v)) { missing++; continue; }
        distinct.add(v.trim());
        if (!isNaN(toNumber(v))) numeric++;
      }
      const present = n - missing;
      const col = { name, index: j, missing, distinct: distinct.size };
      if (present === 0) return Object.assign(col, { type: 'ignored', reason: 'all values missing' });
      if (numeric === present) {
        if (distinct.size === 1) return Object.assign(col, { type: 'ignored', reason: 'constant column' });
        return Object.assign(col, { type: 'numeric' });
      }
      const levels = Array.from(distinct).sort(); // code-point order, same as pandas
      if (levels.length < 2) return Object.assign(col, { type: 'ignored', reason: 'constant column' });
      if (levels.length > Math.min(30, Math.max(2, Math.floor(present / 2)))) {
        return Object.assign(col, { type: 'ignored', reason: `${levels.length} distinct text values (looks like an ID/free text)` });
      }
      return Object.assign(col, { type: 'categorical', levels });
    });
  }

  const median = (arr) => {
    const s = arr.slice().sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };

  /**
   * Build the regression design: numeric predictors as-is, categorical predictors
   * one-hot encoded with the first (alphabetical) level dropped as reference,
   * exactly like pandas.get_dummies(drop_first=True).
   */
  function buildDataset(parsed, targetName, columns) {
    columns = columns || inferColumns(parsed);
    const target = columns.find((c) => c.name === targetName);
    if (!target) throw new DatasetError(`Target column "${targetName}" was not found.`);
    if (target.type !== 'numeric') {
      throw new DatasetError(`Target column "${targetName}" is not numeric.`, 'Choose a numeric column as the response y.');
    }
    const predictors = columns.filter((c) => c !== target && (c.type === 'numeric' || c.type === 'categorical'));
    const notes = [];
    columns.filter((c) => c.type === 'ignored' && c !== target)
      .forEach((c) => notes.push(`Ignored column "${c.name}": ${c.reason}.`));
    if (!predictors.length) {
      throw new DatasetError('No usable predictor columns besides the target.',
        'The CSV needs at least one numeric or categorical column in addition to y.');
    }

    // drop rows with missing values in the used columns
    const used = [target, ...predictors];
    const keep = [];
    parsed.rows.forEach((r, i) => { if (used.every((c) => !isMissing(r[c.index]))) keep.push(i); });
    const dropped = parsed.rows.length - keep.length;
    if (dropped) {
      const cols = used.filter((c) => c.missing).map((c) => `${c.name} (${c.missing})`).join(', ');
      notes.push(`Dropped ${dropped} row(s) with missing values — missing counts: ${cols}.`);
    }

    const features = [];   // original predictors (for the live predictor UI)
    const designNames = []; // one column per numeric feature / dummy
    predictors.forEach((c) => {
      if (c.type === 'numeric') {
        const vals = keep.map((i) => toNumber(parsed.rows[i][c.index]));
        const f = {
          name: c.name, type: 'numeric', design: [designNames.length],
          min: vals.reduce((a, b) => Math.min(a, b)), max: vals.reduce((a, b) => Math.max(a, b)), median: median(vals),
          integer: vals.every((v) => Number.isInteger(v)),
        };
        designNames.push(c.name);
        features.push(f);
      } else {
        const present = new Set(keep.map((i) => parsed.rows[i][c.index].trim()));
        const levels = c.levels.filter((l) => present.has(l));
        if (levels.length < 2) { notes.push(`Ignored column "${c.name}": only one level after dropping missing rows.`); return; }
        const f = { name: c.name, type: 'categorical', levels, reference: levels[0], design: [], dummyLevels: levels.slice(1) };
        levels.slice(1).forEach((lv) => { f.design.push(designNames.length); designNames.push(`${c.name}_${lv}`); });
        features.push(f);
      }
    });

    const X = keep.map((i) => {
      const r = parsed.rows[i];
      const x = [];
      features.forEach((f) => {
        const raw = r[predictors.find((c) => c.name === f.name).index];
        if (f.type === 'numeric') x.push(toNumber(raw));
        else f.dummyLevels.forEach((lv) => x.push(raw.trim() === lv ? 1 : 0));
      });
      return x;
    });
    const y = keep.map((i) => toNumber(parsed.rows[i][target.index]));
    const p = designNames.length;
    if (p < 1) throw new DatasetError('No usable predictor columns besides the target.');
    if (y.length < 10) {
      throw new DatasetError(`Only ${y.length} complete row(s) — too few for an 80/20 split.`, 'At least 10 complete rows are required.');
    }
    if (Math.floor(y.length * 0.8) < p + 2) {
      throw new DatasetError(`Too many predictors (${p} after one-hot encoding) for ${y.length} rows.`,
        'The training set (80%) must have more rows than predictors + 1.');
    }
    return { target: target.name, designNames, features, X, y, rowIds: keep, notes, nRaw: parsed.rows.length };
  }

  const api = { parseCSV, inferColumns, buildDataset, DatasetError, isMissing, toNumber };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.CSV = api;
})(typeof self !== 'undefined' ? self : this);
