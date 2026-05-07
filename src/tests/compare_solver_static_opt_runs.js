#!/usr/bin/env node
'use strict';

/**
 * Pretty-print totals from two run_solver_tests_js.js --json runs
 * (baseline vs --solver-opt all) for quick A/B.
 */

const fs = require('fs');

function readJson(path) {
    return JSON.parse(fs.readFileSync(path, 'utf8'));
}

function num(x) {
    const n = Number(x);
    return Number.isFinite(n) ? n : 0;
}

function fmt(k, v) {
    const n = num(v);
    if (k.endsWith('_ms') || k.includes('_ms_')) {
        return n.toFixed(3);
    }
    return String(Math.round(n));
}

function resultKey(row) {
    return `${row.game || ''}#${Number.isFinite(row.level) ? row.level : ''}`;
}

function collectComparisonFailures(b, o) {
    const failures = [];
    const bt = b.totals || {};
    const ot = o.totals || {};
    for (const key of ['levels', 'solved', 'timeout', 'exhausted', 'skipped_message', 'errors']) {
        if (num(bt[key]) !== num(ot[key])) {
            failures.push(`${key} baseline=${fmt(key, bt[key])} optimized=${fmt(key, ot[key])}`);
        }
    }

    const bRows = new Map((b.results || []).map(row => [resultKey(row), row]));
    const oRows = new Map((o.results || []).map(row => [resultKey(row), row]));
    for (const [key, row] of bRows) {
        const opt = oRows.get(key);
        if (!opt) {
            failures.push(`missing optimized result ${key}`);
            continue;
        }
        if (row.status !== opt.status) {
            failures.push(`status ${key} baseline=${row.status} optimized=${opt.status}`);
        } else if (row.status === 'solved' && num(row.solution_length) !== num(opt.solution_length)) {
            failures.push(`solution_length ${key} baseline=${fmt('solution_length', row.solution_length)} optimized=${fmt('solution_length', opt.solution_length)}`);
        }
    }
    for (const key of oRows.keys()) {
        if (!bRows.has(key)) failures.push(`extra optimized result ${key}`);
    }
    return failures;
}

function main() {
    const baselinePath = process.argv[2];
    const optimizedPath = process.argv[3];
    if (!baselinePath || !optimizedPath) {
        process.stderr.write(
            'Usage: node src/tests/compare_solver_static_opt_runs.js baseline.json optimized.json\n',
        );
        process.exit(2);
    }
    const b = readJson(baselinePath);
    const o = readJson(optimizedPath);
    const bt = b.totals || {};
    const ot = o.totals || {};

    const keys = [
        'levels',
        'solved',
        'timeout',
        'exhausted',
        'skipped_message',
        'errors',
        'expanded',
        'generated',
        'hash_collisions',
        'compile_ms',
        'static_analysis_ms',
        'static_optimization_removed_rules',
        'removed_cosmetic_objects',
        'removed_collision_layers',
        'merged_object_aliases',
        'merged_object_groups',
        'solver_opt_ms_inert',
        'solver_opt_ms_cosmetic',
        'solver_opt_ms_merge',
    ];

    const w = 28;
    process.stdout.write(`${'metric'.padEnd(w)}${'baseline'.padStart(14)}${'optimized'.padStart(14)}${'delta'.padStart(14)}\n`);
    process.stdout.write(`${'-'.repeat(w + 42)}\n`);
    for (const k of keys) {
        const bv = num(bt[k]);
        const ov = num(ot[k]);
        const d = ov - bv;
        const isMs = k.endsWith('_ms') || k.includes('_ms_');
        const dStr =
            d === 0
                ? '—'
                : isMs
                  ? (d > 0 ? '+' : '') + d.toFixed(3)
                  : d > 0
                    ? `+${Math.round(d)}`
                    : String(Math.round(d));
        process.stdout.write(
            `${k.padEnd(w)}${fmt(k, bv).padStart(14)}${fmt(k, ov).padStart(14)}${dStr.padStart(14)}\n`,
        );
    }
    if (bt.solver_optimization_gated || ot.solver_optimization_gated) {
        process.stdout.write(
            `\nNote: solver_optimization_gated baseline=${!!bt.solver_optimization_gated} optimized=${!!ot.solver_optimization_gated}\n`,
        );
    }
    if (ot.solver_optimization) {
        process.stdout.write(`\noptimized.solver_optimization (JSON nest): ${JSON.stringify(ot.solver_optimization)}\n`);
    }
    const failures = collectComparisonFailures(b, o);
    if (failures.length > 0) {
        process.stderr.write(
            `\nstatic optimization comparison failed:\n${failures.map(f => `  - ${f}`).join('\n')}\n`,
        );
        process.exit(1);
    }
}

main();
