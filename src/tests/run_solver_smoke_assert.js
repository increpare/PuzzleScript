#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const path = require('path');

function usage() {
    console.error('Usage: node src/tests/run_solver_smoke_assert.js <puzzlescript_solver> <solver_tests_dir> [--timeout-ms N] [--require-specialized-full-turn] [--compact-turn-oracle] [--require-compact-oracle-checks]');
    process.exit(1);
}

const args = process.argv.slice(2);
if (args.length < 2) {
    usage();
}

const solverPath = path.resolve(args[0]);
const fixtureDir = path.resolve(args[1]);
let timeoutMs = 1000;
const extraSolverArgs = [];
let requireCompactOracleChecks = false;
for (let index = 2; index < args.length; index++) {
    if (args[index] === '--timeout-ms' && index + 1 < args.length) {
        timeoutMs = Number.parseInt(args[++index], 10);
    } else if (args[index] === '--require-specialized-full-turn') {
        extraSolverArgs.push('--require-specialized-full-turn');
    } else if (args[index] === '--require-compiled-tick') {
        extraSolverArgs.push('--require-specialized-full-turn');
    } else if (args[index] === '--compact-turn-oracle' || args[index] === '--compact-tick-oracle') {
        extraSolverArgs.push('--compact-turn-oracle');
    } else if (args[index] === '--require-compact-oracle-checks') {
        requireCompactOracleChecks = true;
    } else {
        throw new Error(`Unsupported argument: ${args[index]}`);
    }
}

function total(json, field, oldField = null) {
    if (json.totals[field] !== undefined) return json.totals[field];
    if (oldField !== null && json.totals[oldField] !== undefined) return json.totals[oldField];
    return 0;
}

function runSolver(extraArgs = []) {
    const result = spawnSync(solverPath, [
        fixtureDir,
        '--timeout-ms', String(timeoutMs),
        '--jobs', '1',
        '--strategy', 'bfs',
        '--no-solutions',
        '--quiet',
        '--json',
        ...extraSolverArgs,
        ...extraArgs,
    ], {
        encoding: 'utf8',
        maxBuffer: 128 * 1024 * 1024,
    });
    if (result.error) {
        throw result.error;
    }
    if (result.status !== 0) {
        throw new Error(`solver exited ${result.status}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
    }
    return JSON.parse(result.stdout);
}

function resultKey(result) {
    return `${result.game}#${result.level}`;
}

function assertSmoke(json) {
    if (json.totals.levels !== 7) throw new Error(`levels ${json.totals.levels}, expected 7`);
    if (json.totals.solved !== 5) throw new Error(`solved ${json.totals.solved}, expected 5`);
    if (json.totals.exhausted !== 1) throw new Error(`exhausted ${json.totals.exhausted}, expected 1`);
    if (json.totals.skipped_message !== 1) throw new Error(`skipped ${json.totals.skipped_message}, expected 1`);
    if (json.totals.timeout !== 0) throw new Error(`timeout ${json.totals.timeout}, expected 0`);
    if (json.totals.errors !== 0) throw new Error(`errors ${json.totals.errors}, expected 0`);
    const compactTurnOracleFailures = total(json, 'compact_turn_oracle_failures', 'compact_tick_oracle_failures');
    const compactTurnOracleChecks = total(json, 'compact_turn_oracle_checks', 'compact_tick_oracle_checks');
    if (compactTurnOracleFailures !== 0) {
        throw new Error(`compact_turn_oracle_failures ${compactTurnOracleFailures}, expected 0`);
    }
    if (requireCompactOracleChecks && !(compactTurnOracleChecks > 0)) {
        throw new Error(`compact_turn_oracle_checks ${compactTurnOracleChecks}, expected > 0`);
    }

    const expected = new Map([
        ['impossible.txt#0', { status: 'exhausted', solution: [] }],
        ['message_skip.txt#0', { status: 'skipped_message', solution: [] }],
        ['message_skip.txt#1', { status: 'solved', solution: ['right'] }],
        ['multi_level.txt#0', { status: 'solved', solution: ['right'] }],
        ['multi_level.txt#1', { status: 'solved', solution: ['left'] }],
        ['one_move.txt#0', { status: 'solved', solution: ['right'] }],
        ['push_goal.txt#0', { status: 'solved', solution: ['right', 'right'] }],
    ]);
    const byKey = new Map(json.results.map((result) => [resultKey(result), result]));
    if (byKey.size !== expected.size) {
        throw new Error(`result count ${byKey.size}, expected ${expected.size}`);
    }
    for (const [key, wanted] of expected) {
        const actual = byKey.get(key);
        if (!actual) throw new Error(`missing ${key}`);
        if (actual.status !== wanted.status) throw new Error(`${key} status ${actual.status}, expected ${wanted.status}`);
        if (JSON.stringify(actual.solution || []) !== JSON.stringify(wanted.solution)) {
            throw new Error(`${key} solution ${JSON.stringify(actual.solution)}, expected ${JSON.stringify(wanted.solution)}`);
        }
        if (actual.solution_length !== (actual.solution || []).length) {
            throw new Error(`${key} solution_length mismatch`);
        }
        if (actual.status === 'solved' && actual.solution.length === 0) {
            throw new Error(`${key} solved without solution`);
        }
        if (actual.status !== 'solved' && actual.solution.length !== 0) {
            throw new Error(`${key} non-solved with solution`);
        }
    }
}

const json = runSolver();
assertSmoke(json);
let suffix = '';
const compactTurnOracleChecks = total(json, 'compact_turn_oracle_checks', 'compact_tick_oracle_checks');
const compactTurnOracleFailures = total(json, 'compact_turn_oracle_failures', 'compact_tick_oracle_failures');
if (compactTurnOracleChecks > 0 || compactTurnOracleFailures > 0 || requireCompactOracleChecks) {
    suffix += ` compact_turn_oracle_checks=${compactTurnOracleChecks}`;
    suffix += ` compact_turn_oracle_failures=${compactTurnOracleFailures}`;
}
process.stdout.write(`solver_smoke_assert passed cases=7${suffix}\n`);
