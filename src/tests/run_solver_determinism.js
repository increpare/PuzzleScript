#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const path = require('path');

function usage() {
    console.error('Usage: node src/tests/run_solver_determinism.js <puzzlescript_solver> <solver_tests_dir> [--runs N] [--timeout-ms N]');
    process.exit(1);
}

const args = process.argv.slice(2);
if (args.length < 2) {
    usage();
}

const solverPath = path.resolve(args[0]);
const fixtureDir = path.resolve(args[1]);
let runs = 5;
let timeoutMs = 1000;
for (let index = 2; index < args.length; index++) {
    if (args[index] === '--runs' && index + 1 < args.length) {
        runs = Number.parseInt(args[++index], 10);
    } else if (args[index] === '--timeout-ms' && index + 1 < args.length) {
        timeoutMs = Number.parseInt(args[++index], 10);
    } else {
        throw new Error(`Unsupported argument: ${args[index]}`);
    }
}

function runSolver(jobs) {
    const result = spawnSync(solverPath, [
        fixtureDir,
        '--timeout-ms', String(timeoutMs),
        '--jobs', jobs,
        '--strategy', 'bfs',
        '--no-solutions',
        '--quiet',
        '--json',
    ], {
        encoding: 'utf8',
        maxBuffer: 128 * 1024 * 1024,
    });
    if (result.error) throw result.error;
    if (result.status !== 0) {
        throw new Error(`solver exited ${result.status}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
    }
    return JSON.parse(result.stdout);
}

function normalize(json) {
    if (json.totals.timeout !== 0) throw new Error(`unexpected timeout count ${json.totals.timeout}`);
    if (json.totals.errors !== 0) throw new Error(`unexpected error count ${json.totals.errors}`);
    return json.results
        .slice()
        .sort((a, b) => `${a.game}#${a.level}`.localeCompare(`${b.game}#${b.level}`))
        .map((result) => ({
            game: result.game,
            level: result.level,
            status: result.status,
            error: result.error || '',
            solution: result.solution || [],
            solution_length: result.solution_length,
            expanded: result.expanded,
            generated: result.generated,
            unique_states: result.unique_states,
            duplicates: result.duplicates,
            max_frontier: result.max_frontier,
            timeout_ms: result.timeout_ms,
        }));
}

const baseline = JSON.stringify(normalize(runSolver('1')));
for (let index = 1; index < runs; index++) {
    const candidate = JSON.stringify(normalize(runSolver('1')));
    if (candidate !== baseline) {
        throw new Error(`serial run ${index + 1} differed from run 1`);
    }
}
const parallel = JSON.stringify(normalize(runSolver('auto')));
if (parallel !== baseline) {
    throw new Error('--jobs auto differed from --jobs 1 on smoke corpus');
}

process.stdout.write(`solver_determinism passed runs=${runs} plus_jobs_auto=1\n`);

