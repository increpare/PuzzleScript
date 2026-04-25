#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

function usage() {
    console.error('Usage: node src/tests/check_solver_focus_benchmark_fresh.js <benchmark.json> <manifest.json> --runs N --corpus PATH --strategy NAME');
    process.exit(2);
}

const args = process.argv.slice(2);
if (args.length < 2) {
    usage();
}

const benchmarkPath = path.resolve(args[0]);
const manifestPath = path.resolve(args[1]);
let expectedRuns = null;
let expectedCorpus = null;
let expectedStrategy = null;

function parsePositiveInt(value, label) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new Error(`${label} must be a positive integer: ${value}`);
    }
    return parsed;
}

for (let index = 2; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--runs' && index + 1 < args.length) {
        expectedRuns = parsePositiveInt(args[++index], '--runs');
    } else if (arg === '--corpus' && index + 1 < args.length) {
        expectedCorpus = path.resolve(args[++index]);
    } else if (arg === '--strategy' && index + 1 < args.length) {
        expectedStrategy = args[++index];
    } else {
        usage();
    }
}

if (expectedRuns === null || expectedCorpus === null || expectedStrategy === null) {
    usage();
}

function stale(reason) {
    process.stderr.write(`solver_focus_compare stale ${benchmarkPath}: ${reason}\n`);
    process.exit(1);
}

function key(target) {
    return `${target.game}#${target.level}`;
}

if (!fs.existsSync(benchmarkPath)) {
    stale('missing');
}
if (!fs.existsSync(manifestPath)) {
    stale(`missing manifest ${manifestPath}`);
}

const benchmarkStat = fs.statSync(benchmarkPath);
const manifestStat = fs.statSync(manifestPath);
if (benchmarkStat.mtimeMs < manifestStat.mtimeMs) {
    stale('older than focus manifest');
}

const benchmark = JSON.parse(fs.readFileSync(benchmarkPath, 'utf8'));
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
const manifestTargets = manifest.targets || [];
const benchmarkTargets = benchmark.targets || [];

if (benchmark.runs_per_target !== expectedRuns) {
    stale(`runs_per_target=${benchmark.runs_per_target}, expected ${expectedRuns}`);
}
if (path.resolve(benchmark.corpus || '') !== expectedCorpus) {
    stale(`corpus=${benchmark.corpus}, expected ${expectedCorpus}`);
}
if (path.resolve(benchmark.manifest || '') !== manifestPath) {
    stale(`manifest=${benchmark.manifest}, expected ${manifestPath}`);
}
if (benchmark.strategy !== expectedStrategy) {
    stale(`strategy=${benchmark.strategy}, expected ${expectedStrategy}`);
}
if (benchmark.target_count !== manifestTargets.length || benchmarkTargets.length !== manifestTargets.length) {
    stale(`target_count=${benchmark.target_count}, expected ${manifestTargets.length}`);
}

for (let index = 0; index < manifestTargets.length; index++) {
    if (key(manifestTargets[index]) !== key(benchmarkTargets[index])) {
        stale(`target[${index}]=${key(benchmarkTargets[index])}, expected ${key(manifestTargets[index])}`);
    }
}

process.exit(0);
