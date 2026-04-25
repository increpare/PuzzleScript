#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

function usage() {
    console.error('Usage: node src/tests/compare_solver_focus_benchmarks.js <interpreted.json> <compiled.json> [--detail] [--goal-ratio N]');
    process.exit(1);
}

if (process.argv.length < 4) {
    usage();
}

const options = {
    detail: false,
    goalRatio: null,
};
for (let index = 4; index < process.argv.length; index++) {
    const arg = process.argv[index];
    if (arg === '--detail') {
        options.detail = true;
    } else if (arg === '--goal-ratio' && index + 1 < process.argv.length) {
        options.goalRatio = Number.parseFloat(process.argv[++index]);
        if (!Number.isFinite(options.goalRatio) || options.goalRatio <= 0) {
            throw new Error(`--goal-ratio must be positive: ${process.argv[index]}`);
        }
    } else {
        usage();
    }
}

function readJson(filePath) {
    return JSON.parse(fs.readFileSync(path.resolve(filePath), 'utf8'));
}

function targetKey(target) {
    return `${target.game}#${target.level}`;
}

function formatNumber(value, digits = 1) {
    if (value === null || value === undefined || !Number.isFinite(value)) {
        return 'n/a';
    }
    return Number(value).toFixed(digits);
}

function ratio(newValue, oldValue) {
    if (!Number.isFinite(newValue) || !Number.isFinite(oldValue) || oldValue === 0) {
        return null;
    }
    return newValue / oldValue;
}

function formatDelta(newValue, oldValue, digits = 1) {
    const valueRatio = ratio(newValue, oldValue);
    if (valueRatio === null) {
        return 'n/a';
    }
    const percent = (valueRatio - 1) * 100;
    const sign = percent > 0 ? '+' : '';
    return `${formatNumber(valueRatio, 3)}x (${sign}${formatNumber(percent, digits)}%)`;
}

const interpreted = readJson(process.argv[2]);
const compiled = readJson(process.argv[3]);
const interpretedKeys = (interpreted.targets || []).map(targetKey);
const compiledKeys = (compiled.targets || []).map(targetKey);
const sameTargets = interpretedKeys.length === compiledKeys.length
    && interpretedKeys.every((key, index) => key === compiledKeys[index]);

const interpretedStatus = {};
const compiledStatus = {};
for (const target of interpreted.targets || []) {
    for (const [status, count] of Object.entries(target.status_counts || {})) {
        interpretedStatus[status] = (interpretedStatus[status] || 0) + count;
    }
}
for (const target of compiled.targets || []) {
    for (const [status, count] of Object.entries(target.status_counts || {})) {
        compiledStatus[status] = (compiledStatus[status] || 0) + count;
    }
}

function countersMedian(summary, key) {
    const values = (summary.samples || [])
        .map((sample) => sample.runtime_counters && sample.runtime_counters[key])
        .filter((value) => Number.isFinite(value));
    if (values.length === 0) {
        return null;
    }
    values.sort((a, b) => a - b);
    return values[Math.floor(values.length / 2)];
}

function sampleBoolean(summary, key) {
    const values = (summary.samples || [])
        .map((sample) => sample[key])
        .filter((value) => value !== null && value !== undefined);
    if (values.length === 0) {
        return null;
    }
    return values.some(Boolean);
}

function targetRows() {
    const compiledByKey = new Map((compiled.targets || []).map((target) => [targetKey(target), target]));
    return (interpreted.targets || []).map((target) => {
        const key = targetKey(target);
        const compiledTarget = compiledByKey.get(key);
        if (!compiledTarget) {
            return null;
        }
        const compiledRuleHits = countersMedian(compiledTarget, 'compiled_rule_group_hits');
        const compiledTickHits = countersMedian(compiledTarget, 'compiled_tick_hits');
        const compiledRulesAttached = sampleBoolean(compiledTarget, 'compiled_rules_attached');
        const compiledTickAttached = sampleBoolean(compiledTarget, 'compiled_tick_attached');
        return {
            key,
            interpreted: target,
            compiled: compiledTarget,
            wallRatio: ratio(compiledTarget.median.wall_ms, target.median.wall_ms),
            elapsedRatio: ratio(compiledTarget.median.elapsed_ms, target.median.elapsed_ms),
            generatedRatio: ratio(compiledTarget.median.generated, target.median.generated),
            compiledRuleHits,
            compiledTickHits,
            bucket: compiledUsageBucket(compiledTickHits, compiledRuleHits),
            reason: compiledUsageReason(compiledTickHits, compiledRuleHits, compiledRulesAttached, compiledTickAttached),
            compiledRulesAttached,
            compiledTickAttached,
            candidateCells: countersMedian(compiledTarget, 'candidate_cells_tested'),
            rowScans: countersMedian(compiledTarget, 'row_scans'),
            patternTests: countersMedian(compiledTarget, 'pattern_tests'),
            maskRebuilds: countersMedian(compiledTarget, 'mask_rebuild_calls'),
        };
    }).filter(Boolean);
}

function compiledUsageBucket(compiledTickHits, compiledRuleHits) {
    if (compiledTickHits === null || compiledRuleHits === null) {
        return 'no_counters';
    }
    if (compiledTickHits === 0 && compiledRuleHits === 0) {
        return 'no_tick_no_rules';
    }
    if (compiledTickHits > 0 && compiledRuleHits === 0) {
        return 'tick_no_rules';
    }
    if (compiledRuleHits > 0) {
        return 'compiled_rules';
    }
    return 'unknown';
}

function compiledUsageReason(compiledTickHits, compiledRuleHits, compiledRulesAttached, compiledTickAttached) {
    if (compiledTickHits === null || compiledRuleHits === null) {
        return 'runtime_counters_missing';
    }
    if (!compiledTickAttached) {
        return 'compiled_tick_backend_not_attached';
    }
    if (!compiledRulesAttached) {
        return 'compiled_rules_backend_not_attached';
    }
    if (compiledTickHits === 0) {
        return 'compiled_tick_not_called';
    }
    if (compiledRuleHits === 0) {
        return 'compiled_tick_bypassed_generic_rule_counter';
    }
    return 'compiled_rule_groups_hit';
}

function printCompiledUsageSummary(rows) {
    const counts = new Map();
    const reasons = new Map();
    for (const row of rows) {
        counts.set(row.bucket, (counts.get(row.bucket) || 0) + 1);
        reasons.set(row.reason, (reasons.get(row.reason) || 0) + 1);
    }
    const labels = ['compiled_rules', 'tick_no_rules', 'no_tick_no_rules', 'no_counters', 'unknown'];
    const parts = labels
        .filter((label) => counts.has(label))
        .map((label) => `${label}=${counts.get(label)}`);
    process.stdout.write(`  compiled_usage: ${parts.length === 0 ? 'n/a' : parts.join(' ')}\n`);
    const reasonParts = Array.from(reasons.entries())
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
        .map(([reason, count]) => `${reason}=${count}`);
    process.stdout.write(`  compiled_usage_reasons: ${reasonParts.length === 0 ? 'n/a' : reasonParts.join(' ')}\n`);
}

function printTargetTable(label, rows) {
    process.stdout.write(`  ${label}\n`);
    for (const row of rows) {
        process.stdout.write(
            `    ${formatNumber(row.elapsedRatio, 3)}x elapsed` +
            ` wall=${formatNumber(row.wallRatio, 3)}x` +
            ` generated=${formatNumber(row.generatedRatio, 3)}x` +
            ` bucket=${row.bucket}` +
            ` reason=${row.reason}` +
            ` ${row.key}` +
            ` hits=${row.compiledRuleHits === null ? 'n/a' : row.compiledRuleHits}` +
            ` tick=${row.compiledTickHits === null ? 'n/a' : row.compiledTickHits}` +
            ` rows=${row.rowScans === null ? 'n/a' : row.rowScans}` +
            ` cells=${row.candidateCells === null ? 'n/a' : row.candidateCells}` +
            ` pattern_tests=${row.patternTests === null ? 'n/a' : row.patternTests}` +
            ` mask_rebuilds=${row.maskRebuilds === null ? 'n/a' : row.maskRebuilds}` +
            `\n`
        );
    }
}

process.stdout.write('solver_focus_compare\n');
process.stdout.write(`  targets: interpreted=${interpreted.target_count} compiled=${compiled.target_count} same=${sameTargets ? 'yes' : 'no'}\n`);
process.stdout.write(`  runs_per_target: interpreted=${interpreted.runs_per_target} compiled=${compiled.runs_per_target}\n`);
process.stdout.write(`  status: interpreted=${JSON.stringify(interpretedStatus)} compiled=${JSON.stringify(compiledStatus)}\n`);
process.stdout.write(
    `  median_wall_ms: interpreted=${formatNumber(interpreted.median.wall_ms)}` +
    ` compiled=${formatNumber(compiled.median.wall_ms)}` +
    ` compiled/interpreted=${formatDelta(compiled.median.wall_ms, interpreted.median.wall_ms)}\n`
);
process.stdout.write(
    `  median_elapsed_ms: interpreted=${formatNumber(interpreted.median.elapsed_ms)}` +
    ` compiled=${formatNumber(compiled.median.elapsed_ms)}` +
    ` compiled/interpreted=${formatDelta(compiled.median.elapsed_ms, interpreted.median.elapsed_ms)}\n`
);
process.stdout.write(
    `  median_generated: interpreted=${formatNumber(interpreted.median.generated, 0)}` +
    ` compiled=${formatNumber(compiled.median.generated, 0)}` +
    ` compiled/interpreted=${formatDelta(compiled.median.generated, interpreted.median.generated)}\n`
);
if (options.goalRatio !== null) {
    const elapsedRatio = ratio(compiled.median.elapsed_ms, interpreted.median.elapsed_ms);
    process.stdout.write(
        `  goal_elapsed_ratio: target<=${formatNumber(options.goalRatio, 3)}` +
        ` actual=${elapsedRatio === null ? 'n/a' : formatNumber(elapsedRatio, 3)}` +
        ` pass=${elapsedRatio !== null && elapsedRatio <= options.goalRatio ? 'yes' : 'no'}\n`
    );
}

if (options.detail) {
    const rows = targetRows();
    printCompiledUsageSummary(rows);
    const bySlowest = rows.slice().sort((a, b) => (b.elapsedRatio || 0) - (a.elapsedRatio || 0));
    const byFastest = rows.slice().sort((a, b) => (a.elapsedRatio || Infinity) - (b.elapsedRatio || Infinity));
    printTargetTable('slowest_targets:', bySlowest.slice(0, 10));
    printTargetTable('fastest_targets:', byFastest.slice(0, 10));
}

if (!sameTargets) {
    process.exitCode = 1;
}
if (options.goalRatio !== null) {
    const elapsedRatio = ratio(compiled.median.elapsed_ms, interpreted.median.elapsed_ms);
    if (elapsedRatio === null || elapsedRatio > options.goalRatio) {
        process.exitCode = 2;
    }
}
