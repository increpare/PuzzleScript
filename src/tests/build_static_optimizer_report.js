#!/usr/bin/env node
'use strict';

/**
 * Run JS solver corpus twice (baseline vs --solver-opt all), aggregate per-game
 * static-optimization telemetry, write HTML + JSON summary.
 *
 * Usage:
 *   node src/tests/build_static_optimizer_report.js --corpus src/tests/solver_tests --out build/static-optimizer-report/index.html
 *   node src/tests/build_static_optimizer_report.js --corpus src/tests/solver_smoke_tests --out /tmp/report.html --timeout-ms 5000
 */

const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const RUNNER = path.join(__dirname, 'run_solver_tests_js.js');

function usage(code = 1) {
    const t = [
        'Usage: node src/tests/build_static_optimizer_report.js --corpus DIR --out PATH.html',
        '  [--timeout-ms N] [--game SUBSTRING] [--json PATH] [--keep-temp-dir DIR]',
        '',
        'Runs solver JSON twice (baseline, then --solver-opt all), merges per game.',
        'Telemetry (removals, merge counts, hook ms) is compile-once per game on the first level row.',
    ].join('\n');
    (code ? process.stderr : process.stdout).write(`${t}\n`);
    process.exit(code);
}

function parseArgs(argv) {
    const o = {
        corpus: null,
        outHtml: null,
        outJson: null,
        timeoutMs: 250,
        game: null,
        keepTempDir: null,
    };
    const a = argv.slice(2);
    if (a.length === 0 || a.includes('--help') || a.includes('-h')) usage(0);
    for (let i = 0; i < a.length; i++) {
        const arg = a[i];
        if (arg === '--corpus' && i + 1 < a.length) o.corpus = path.resolve(a[++i]);
        else if (arg === '--out' && i + 1 < a.length) o.outHtml = path.resolve(a[++i]);
        else if (arg === '--json' && i + 1 < a.length) o.outJson = path.resolve(a[++i]);
        else if (arg === '--timeout-ms' && i + 1 < a.length) o.timeoutMs = Number.parseInt(a[++i], 10);
        else if (arg === '--game' && i + 1 < a.length) o.game = a[++i];
        else if (arg === '--keep-temp-dir' && i + 1 < a.length) o.keepTempDir = path.resolve(a[++i]);
        else throw new Error(`Unknown or incomplete argument: ${arg}`);
    }
    if (!o.corpus || !fs.existsSync(o.corpus)) {
        throw new Error(`--corpus must exist: ${o.corpus}`);
    }
    if (!o.outHtml) {
        throw new Error('--out PATH.html is required');
    }
    if (!Number.isFinite(o.timeoutMs) || o.timeoutMs < 0) {
        throw new Error('invalid --timeout-ms');
    }
    return o;
}

function runJson(corpus, timeoutMs, game, solverOptAll) {
    const args = [RUNNER, corpus, '--quiet', '--json', '--no-solutions', '--timeout-ms', String(timeoutMs)];
    if (game) {
        args.push('--game', game);
    }
    if (solverOptAll) {
        args.push('--solver-opt', 'all');
    }
    const buf = execFileSync(process.execPath, args, {
        encoding: 'utf8',
        maxBuffer: 256 * 1024 * 1024,
        cwd: REPO_ROOT,
    });
    return JSON.parse(buf);
}

function num(x) {
    const n = Number(x);
    return Number.isFinite(n) ? n : 0;
}

/** First row per game carries compile-once metrics in run_solver_tests_js. */
function metricsRowForGame(rows, game) {
    const forGame = rows.filter(r => r && r.game === game);
    const hit = forGame.find(r => num(r.compile_ms) > 0);
    return hit || forGame[0] || null;
}

function aggregateGame(payload) {
    const rows = payload.results || [];
    const games = [...new Set(rows.map(r => r.game).filter(Boolean))].sort();
    const out = [];
    for (const game of games) {
        const gameRows = rows.filter(r => r.game === game);
        const mr = metricsRowForGame(rows, game);
        const levels = gameRows.length;
        const solved = gameRows.filter(r => r.status === 'solved').length;
        out.push({
            game,
            levels,
            solved,
            compile_ms: mr ? num(mr.compile_ms) : 0,
            static_analysis_ms: mr ? num(mr.static_analysis_ms) : 0,
            static_optimization_removed_rules: mr ? num(mr.static_optimization_removed_rules) : 0,
            removed_inert_rules: mr ? num(mr.removed_inert_rules) : 0,
            removed_cosmetic_objects: mr ? num(mr.removed_cosmetic_objects) : 0,
            removed_collision_layers: mr ? num(mr.removed_collision_layers) : 0,
            merged_object_aliases: mr ? num(mr.merged_object_aliases) : 0,
            merged_object_groups: mr ? num(mr.merged_object_groups) : 0,
            solver_opt_ms_inert: mr ? num(mr.solver_opt_ms_inert) : 0,
            solver_opt_ms_cosmetic: mr ? num(mr.solver_opt_ms_cosmetic) : 0,
            solver_opt_ms_merge: mr ? num(mr.solver_opt_ms_merge) : 0,
            errors: gameRows.filter(r => ['compile_error', 'level_error'].includes(r.status)).length,
        });
    }
    return out;
}

function mergeBaselineOptimized(baselineGames, optimizedGames) {
    const byGame = new Map();
    for (const g of baselineGames) byGame.set(g.game, { baseline: g, optimized: null });
    for (const g of optimizedGames) {
        const row = byGame.get(g.game);
        if (row) {
            row.optimized = g;
        } else {
            byGame.set(g.game, { baseline: null, optimized: g });
        }
    }
    const merged = [];
    for (const [game, { baseline, optimized }] of [...byGame.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
        const b = baseline || {};
        const o = optimized || {};
        const impact =
            num(o.removed_inert_rules)
            + num(o.removed_cosmetic_objects)
            + num(o.removed_collision_layers)
            + num(o.merged_object_aliases)
            + num(o.merged_object_groups)
            + num(o.static_optimization_removed_rules);
        merged.push({
            game,
            levels: num(o.levels || b.levels),
            solved_b: num(b.solved),
            solved_o: optimized ? num(o.solved) : num(b.solved),
            compile_b: num(b.compile_ms),
            compile_o: num(o.compile_ms),
            static_b: num(b.static_analysis_ms),
            static_o: num(o.static_analysis_ms),
            static_opt_rules_b: num(b.static_optimization_removed_rules),
            static_opt_rules_o: num(o.static_optimization_removed_rules),
            removed_inert: num(o.removed_inert_rules),
            removed_cosmetic: num(o.removed_cosmetic_objects),
            removed_layers: num(o.removed_collision_layers),
            merged_aliases: num(o.merged_object_aliases),
            merged_groups: num(o.merged_object_groups),
            ms_inert: num(o.solver_opt_ms_inert),
            ms_cosmetic: num(o.solver_opt_ms_cosmetic),
            ms_merge: num(o.solver_opt_ms_merge),
            errors_b: num(b.errors),
            errors_o: num(o.errors),
            impact,
        });
    }
    merged.sort((a, b) => b.impact - a.impact || a.game.localeCompare(b.game));
    return merged;
}

function escapeHtml(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function fmtMs(n) {
    return num(n).toFixed(2);
}

function fmtInt(n) {
    return String(Math.round(num(n)));
}

function buildHtml(meta, rows, totals) {
    const title = 'Solver static optimizer — corpus report';
    const rowHtml = rows
        .map(r => {
            const changed = r.impact > 0 ? ' class="hit"' : '';
            const dCompile = r.compile_o - r.compile_b;
            const dStatic = r.static_o - r.static_b;
            return `<tr${changed}>
  <td class="mono">${escapeHtml(r.game)}</td>
  <td class="num">${fmtInt(r.levels)}</td>
  <td class="num">${fmtInt(r.solved_b)}→${fmtInt(r.solved_o)}</td>
  <td class="num">${fmtMs(r.compile_b)}</td>
  <td class="num">${fmtMs(r.compile_o)}</td>
  <td class="num">${dCompile === 0 ? '—' : (dCompile > 0 ? '+' : '') + fmtMs(dCompile)}</td>
  <td class="num">${fmtMs(r.static_b)}</td>
  <td class="num">${fmtMs(r.static_o)}</td>
  <td class="num">${r.removed_inert ? fmtInt(r.removed_inert) : '—'}</td>
  <td class="num">${r.removed_cosmetic ? fmtInt(r.removed_cosmetic) : '—'}</td>
  <td class="num">${r.removed_layers ? fmtInt(r.removed_layers) : '—'}</td>
  <td class="num">${r.merged_aliases ? fmtInt(r.merged_aliases) : '—'}</td>
  <td class="num">${r.merged_groups ? fmtInt(r.merged_groups) : '—'}</td>
  <td class="num">${fmtMs(r.ms_inert + r.ms_cosmetic + r.ms_merge)}</td>
</tr>`;
        })
        .join('\n');

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>${escapeHtml(title)}</title>
  <style>
    :root { --bg:#0f1419; --panel:#1a2332; --text:#e6edf3; --muted:#8b9cb3; --accent:#58a6ff; --border:#30363d; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 1.5rem; line-height: 1.45; }
    h1 { font-size: 1.35rem; font-weight: 600; margin: 0 0 0.5rem; }
    .meta { color: var(--muted); font-size: 0.9rem; margin-bottom: 1.25rem; }
    .meta code { background: var(--panel); padding: 0.12rem 0.35rem; border-radius: 4px; color: var(--accent); }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; background: var(--panel); border-radius: 8px; overflow: hidden; box-shadow: 0 1px 0 var(--border); }
    th, td { padding: 0.45rem 0.55rem; text-align: left; border-bottom: 1px solid var(--border); vertical-align: top; }
    th { background: #141c2a; color: var(--muted); font-weight: 600; white-space: nowrap; }
    tr:last-child td { border-bottom: none; }
    tr.hit td:first-child { border-left: 3px solid var(--accent); }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    td.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.78rem; word-break: break-all; }
    tfoot td { font-weight: 600; background: #141c2a; }
    .note { margin-top: 1.25rem; font-size: 0.85rem; color: var(--muted); max-width: 72rem; }
  </style>
</head>
<body>
  <h1>${escapeHtml(title)}</h1>
  <div class="meta">
    Generated <code>${escapeHtml(meta.generatedIso)}</code> ·
    Corpus <code>${escapeHtml(meta.corpus)}</code> ·
    timeout-ms=${escapeHtml(String(meta.timeoutMs))}${meta.game ? ` · game filter <code>${escapeHtml(meta.game)}</code>` : ''}
  </div>
  <table>
    <thead>
      <tr>
        <th>Game</th>
        <th class="num">Lvls</th>
        <th class="num">Solved B→O</th>
        <th class="num">compile ms (B)</th>
        <th class="num">compile ms (O)</th>
        <th class="num">Δ compile</th>
        <th class="num">static ms (B)</th>
        <th class="num">static ms (O)</th>
        <th class="num">inert rules −</th>
        <th class="num">cosmetic obj −</th>
        <th class="num">layers −</th>
        <th class="num">merge alias</th>
        <th class="num">merge groups</th>
        <th class="num">hook ms</th>
      </tr>
    </thead>
    <tbody>
${rowHtml}
    </tbody>
    <tfoot>
      <tr>
        <td>Totals (${escapeHtml(String(totals.games))} games)</td>
        <td class="num">${fmtInt(totals.levels)}</td>
        <td class="num">${fmtInt(totals.solved_b)}→${fmtInt(totals.solved_o)}</td>
        <td class="num">${fmtMs(totals.compile_b)}</td>
        <td class="num">${fmtMs(totals.compile_o)}</td>
        <td class="num">${totals.d_compile === 0 ? '—' : (totals.d_compile > 0 ? '+' : '') + fmtMs(totals.d_compile)}</td>
        <td class="num">${fmtMs(totals.static_b)}</td>
        <td class="num">${fmtMs(totals.static_o)}</td>
        <td class="num">${fmtInt(totals.removed_inert)}</td>
        <td class="num">${fmtInt(totals.removed_cosmetic)}</td>
        <td class="num">${fmtInt(totals.removed_layers)}</td>
        <td class="num">${fmtInt(totals.merged_aliases)}</td>
        <td class="num">${fmtInt(totals.merged_groups)}</td>
        <td class="num">${fmtMs(totals.hook_ms)}</td>
      </tr>
    </tfoot>
  </table>
  <p class="note">
    <strong>B</strong> = baseline compile (no <code>--solver-opt</code>).
    <strong>O</strong> = <code>--solver-opt all</code> (inert + cosmetic + merge).
    Per-game removal counts come from the first level row after compile (same convention as JSON totals).
    Rows with a blue left edge had at least one non-zero optimizer change (impact score &gt; 0).
  </p>
</body>
</html>
`;
}

function footerTotals(rows) {
    const t = {
        games: rows.length,
        levels: 0,
        solved_b: 0,
        solved_o: 0,
        compile_b: 0,
        compile_o: 0,
        static_b: 0,
        static_o: 0,
        removed_inert: 0,
        removed_cosmetic: 0,
        removed_layers: 0,
        merged_aliases: 0,
        merged_groups: 0,
        hook_ms: 0,
    };
    for (const r of rows) {
        t.levels += r.levels;
        t.solved_b += r.solved_b;
        t.solved_o += r.solved_o;
        t.compile_b += r.compile_b;
        t.compile_o += r.compile_o;
        t.static_b += r.static_b;
        t.static_o += r.static_o;
        t.removed_inert += r.removed_inert;
        t.removed_cosmetic += r.removed_cosmetic;
        t.removed_layers += r.removed_layers;
        t.merged_aliases += r.merged_aliases;
        t.merged_groups += r.merged_groups;
        t.hook_ms += r.ms_inert + r.ms_cosmetic + r.ms_merge;
    }
    t.d_compile = t.compile_o - t.compile_b;
    return t;
}

function main() {
    const opts = parseArgs(process.argv);
    const tempParent = opts.keepTempDir || path.dirname(opts.outHtml);
    fs.mkdirSync(tempParent, { recursive: true });
    const tempDir = opts.keepTempDir || fs.mkdtempSync(path.join(tempParent, 'static-opt-report-'));
    const baselinePath = path.join(tempDir, 'baseline.json');
    const optimizedPath = path.join(tempDir, 'optimized.json');

    process.stderr.write(`static_optimizer_report: corpus=${opts.corpus}\n`);
    process.stderr.write(`static_optimizer_report: writing temp JSON → ${tempDir}\n`);

    fs.writeFileSync(baselinePath, JSON.stringify(runJson(opts.corpus, opts.timeoutMs, opts.game, false), null, 0));
    fs.writeFileSync(optimizedPath, JSON.stringify(runJson(opts.corpus, opts.timeoutMs, opts.game, true), null, 0));

    const baseline = JSON.parse(fs.readFileSync(baselinePath, 'utf8'));
    const optimized = JSON.parse(fs.readFileSync(optimizedPath, 'utf8'));

    const bGames = aggregateGame(baseline);
    const oGames = aggregateGame(optimized);
    const merged = mergeBaselineOptimized(bGames, oGames);
    const totalsRow = footerTotals(merged);

    const outJsonPath = opts.outJson || opts.outHtml.replace(/\.html?$/i, '') + '.summary.json';
    const payload = {
        generated: new Date().toISOString(),
        corpus: opts.corpus,
        timeout_ms: opts.timeoutMs,
        game: opts.game || null,
        totals_baseline: baseline.totals || {},
        totals_optimized: optimized.totals || {},
        games: merged,
    };
    fs.mkdirSync(path.dirname(opts.outHtml), { recursive: true });
    fs.writeFileSync(outJsonPath, JSON.stringify(payload, null, 2), 'utf8');
    fs.writeFileSync(
        opts.outHtml,
        buildHtml(
            {
                generatedIso: payload.generated,
                corpus: path.relative(REPO_ROOT, opts.corpus) || opts.corpus,
                timeoutMs: opts.timeoutMs,
                game: opts.game,
            },
            merged,
            totalsRow,
        ),
        'utf8',
    );

    if (!opts.keepTempDir) {
        fs.unlinkSync(baselinePath);
        fs.unlinkSync(optimizedPath);
        try {
            fs.rmdirSync(tempDir);
        } catch {
            /* ignore */
        }
    }

    process.stderr.write(`static_optimizer_report: HTML → ${opts.outHtml}\n`);
    process.stderr.write(`static_optimizer_report: JSON → ${outJsonPath}\n`);
}

main();
