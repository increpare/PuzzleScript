#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

function usage() {
    console.error('Usage: node src/tests/compare_solver_focus_benchmarks.js <interpreted.json> <compiled.json>');
    process.exit(1);
}

if (process.argv.length < 4) {
    usage();
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

if (!sameTargets) {
    process.exitCode = 1;
}
