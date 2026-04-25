#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function usage() {
    console.error('Usage: node src/tests/mine_solver_focus_group.js <puzzlescript_solver> <solver_tests_dir> [--timeout-ms N] [--min-elapsed-ms N] [--max-targets N] [--out PATH] [--strategy NAME] [--jobs N]');
    process.exit(1);
}

const args = process.argv.slice(2);
if (args.length < 2) {
    usage();
}

const solverPath = path.resolve(args[0]);
const corpusPath = path.resolve(args[1]);
let timeoutMs = 500;
let minElapsedMs = 250;
let maxTargets = 50;
let outPath = path.resolve('build/native/solver_focus_group.json');
let strategy = 'portfolio';
let jobs = '1';

function parsePositiveInt(value, label) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new Error(`${label} must be a positive integer: ${value}`);
    }
    return parsed;
}

function parseNonNegativeInt(value, label) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed < 0) {
        throw new Error(`${label} must be a non-negative integer: ${value}`);
    }
    return parsed;
}

for (let index = 2; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--timeout-ms' && index + 1 < args.length) {
        timeoutMs = parsePositiveInt(args[++index], '--timeout-ms');
    } else if (arg === '--min-elapsed-ms' && index + 1 < args.length) {
        minElapsedMs = parseNonNegativeInt(args[++index], '--min-elapsed-ms');
    } else if (arg === '--max-targets' && index + 1 < args.length) {
        maxTargets = parsePositiveInt(args[++index], '--max-targets');
    } else if (arg === '--out' && index + 1 < args.length) {
        outPath = path.resolve(args[++index]);
    } else if (arg === '--strategy' && index + 1 < args.length) {
        strategy = args[++index];
    } else if (arg === '--jobs' && index + 1 < args.length) {
        jobs = args[++index];
    } else {
        throw new Error(`Unsupported argument: ${arg}`);
    }
}

function resultKey(result) {
    return `${result.game}#${result.level}`;
}

const commandArgs = [
    corpusPath,
    '--timeout-ms', String(timeoutMs),
    '--jobs', jobs,
    '--strategy', strategy,
    '--no-solutions',
    '--quiet',
    '--json',
];

const started = process.hrtime.bigint();
const result = spawnSync(solverPath, commandArgs, {
    encoding: 'utf8',
    maxBuffer: 512 * 1024 * 1024,
});
const wallMs = Number(process.hrtime.bigint() - started) / 1e6;
if (result.error) {
    throw result.error;
}
if (result.status !== 0) {
    throw new Error(`solver exited ${result.status}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
}

const json = JSON.parse(result.stdout);
if (json.totals.errors !== 0) {
    throw new Error(`solver reported errors=${json.totals.errors}`);
}

const candidates = json.results
    .filter((entry) => entry.status === 'solved')
    .filter((entry) => entry.elapsed_ms >= minElapsedMs && entry.elapsed_ms <= timeoutMs)
    .sort((a, b) => {
        if (a.elapsed_ms !== b.elapsed_ms) return a.elapsed_ms - b.elapsed_ms;
        return resultKey(a).localeCompare(resultKey(b));
    });

const selectedTargets = candidates.slice(0, maxTargets).map((entry) => ({
    game: entry.game,
    level: entry.level,
    first_solved_timeout_ms: timeoutMs,
    previous_timeout_ms: minElapsedMs,
    previous_status: 'above_focus_min_elapsed',
    solved_elapsed_ms: entry.elapsed_ms,
    solved_elapsed_ratio: timeoutMs > 0 ? entry.elapsed_ms / timeoutMs : 0,
    solved_expanded: entry.expanded,
    solved_generated: entry.generated,
    observations: [{
        timeout_ms: timeoutMs,
        status: entry.status,
        elapsed_ms: entry.elapsed_ms,
        expanded: entry.expanded,
        generated: entry.generated,
        unique_states: entry.unique_states,
        duplicates: entry.duplicates,
        max_frontier: entry.max_frontier,
        solution_length: entry.solution_length,
    }],
}));

const manifest = {
    schema_version: 1,
    kind: 'solver_focus_group',
    generated_at: new Date().toISOString(),
    solver: solverPath,
    corpus: corpusPath,
    strategy,
    jobs,
    timeout_ms: timeoutMs,
    min_elapsed_ms: minElapsedMs,
    max_targets: maxTargets,
    target_count: selectedTargets.length,
    candidate_count: candidates.length,
    totals: json.totals,
    wall_ms: wallMs,
    targets: selectedTargets,
};

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, `${JSON.stringify(manifest, null, 2)}\n`);
process.stdout.write(`solver_focus_mine wrote ${outPath} targets=${selectedTargets.length} candidates=${candidates.length} wall_ms=${wallMs.toFixed(1)}\n`);
