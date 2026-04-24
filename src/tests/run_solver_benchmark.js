#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function usage() {
    console.error('Usage: node src/tests/run_solver_benchmark.js <puzzlescript_solver> <solver_tests_dir> [--runs N] [--timeout-ms N] [--out PATH] [--baseline PATH] [--jobs auto|N] [--strategy NAME]');
    process.exit(1);
}

const args = process.argv.slice(2);
if (args.length < 2) {
    usage();
}

const solverPath = path.resolve(args[0]);
const corpusPath = path.resolve(args[1]);
let runs = 5;
let timeoutMs = 250;
let outPath = path.resolve('build/native/solver_benchmark.json');
let baselinePath = path.resolve('solver_perf_baseline.json');
let jobs = '1';
let strategy = 'portfolio';

for (let index = 2; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--runs' && index + 1 < args.length) {
        runs = Number.parseInt(args[++index], 10);
    } else if (arg === '--timeout-ms' && index + 1 < args.length) {
        timeoutMs = Number.parseInt(args[++index], 10);
    } else if (arg === '--out' && index + 1 < args.length) {
        outPath = path.resolve(args[++index]);
    } else if (arg === '--baseline' && index + 1 < args.length) {
        baselinePath = path.resolve(args[++index]);
    } else if (arg === '--jobs' && index + 1 < args.length) {
        jobs = args[++index];
    } else if (arg === '--strategy' && index + 1 < args.length) {
        strategy = args[++index];
    } else {
        throw new Error(`Unsupported argument: ${arg}`);
    }
}

function median(values) {
    const sorted = values.slice().sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
}

function runOnce() {
    const started = process.hrtime.bigint();
    const result = spawnSync(solverPath, [
        corpusPath,
        '--timeout-ms', String(timeoutMs),
        '--jobs', jobs,
        '--strategy', strategy,
        '--no-solutions',
        '--quiet',
        '--json',
    ], {
        encoding: 'utf8',
        maxBuffer: 512 * 1024 * 1024,
    });
    const elapsedMs = Number(process.hrtime.bigint() - started) / 1e6;
    if (result.error) throw result.error;
    if (result.status !== 0) {
        throw new Error(`solver exited ${result.status}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
    }
    const json = JSON.parse(result.stdout);
    const totals = json.totals;
    if (totals.errors !== 0) {
        throw new Error(`solver reported errors=${totals.errors}`);
    }
    const accounted = totals.solved + totals.timeout + totals.exhausted + totals.skipped_message + totals.errors;
    if (accounted !== totals.levels) {
        throw new Error(`totals do not add up: ${accounted} != ${totals.levels}`);
    }
    return {
        wall_ms: elapsedMs,
        totals,
        expanded_per_sec: totals.expanded / (elapsedMs / 1000),
        generated_per_sec: totals.generated / (elapsedMs / 1000),
    };
}

const samples = [];
for (let index = 0; index < runs; index++) {
    const sample = runOnce();
    samples.push(sample);
    process.stderr.write(`solver_benchmark run=${index + 1}/${runs} solved=${sample.totals.solved} wall_ms=${sample.wall_ms.toFixed(1)} generated_per_sec=${sample.generated_per_sec.toFixed(0)}\n`);
}

const levelCounts = new Set(samples.map((sample) => sample.totals.levels));
if (levelCounts.size !== 1) {
    throw new Error('level count differed across runs');
}

const summary = {
    runs,
    timeout_ms: timeoutMs,
    jobs,
    strategy,
    median: {
        wall_ms: median(samples.map((sample) => sample.wall_ms)),
        solved: median(samples.map((sample) => sample.totals.solved)),
        timeout: median(samples.map((sample) => sample.totals.timeout)),
        exhausted: median(samples.map((sample) => sample.totals.exhausted)),
        expanded: median(samples.map((sample) => sample.totals.expanded)),
        generated: median(samples.map((sample) => sample.totals.generated)),
        expanded_per_sec: median(samples.map((sample) => sample.expanded_per_sec)),
        generated_per_sec: median(samples.map((sample) => sample.generated_per_sec)),
    },
    samples,
};

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, `${JSON.stringify(summary, null, 2)}\n`);

if (fs.existsSync(baselinePath)) {
    const baseline = JSON.parse(fs.readFileSync(baselinePath, 'utf8'));
    const base = baseline.median || baseline;
    if (summary.median.wall_ms > base.wall_ms * 1.05) {
        throw new Error(`wall time regression ${summary.median.wall_ms.toFixed(1)} > ${(base.wall_ms * 1.05).toFixed(1)}`);
    }
    if (summary.median.generated_per_sec < base.generated_per_sec * 0.95) {
        throw new Error(`throughput regression ${summary.median.generated_per_sec.toFixed(0)} < ${(base.generated_per_sec * 0.95).toFixed(0)}`);
    }
}

process.stdout.write(`solver_benchmark wrote ${outPath}\n`);
