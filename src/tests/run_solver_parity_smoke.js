#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const path = require('path');

function usage() {
    console.error('Usage: node src/tests/run_solver_parity_smoke.js <puzzlescript_solver> <solver_tests_dir>');
    process.exit(1);
}

if (process.argv.length < 4) {
    usage();
}

const solverPath = path.resolve(process.argv[2]);
const fixtureDir = path.resolve(process.argv[3]);
const jsSolverPath = path.join(__dirname, 'run_solver_tests_js.js');

function runJson(command, args, label) {
    const result = spawnSync(command, args, {
        encoding: 'utf8',
        maxBuffer: 128 * 1024 * 1024,
    });
    if (result.error) {
        throw new Error(`${label} failed to run: ${result.error.message}`);
    }
    if (result.status !== 0) {
        throw new Error(`${label} exited ${result.status}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
    }
    try {
        return JSON.parse(result.stdout);
    } catch (error) {
        throw new Error(`${label} did not emit JSON: ${error.message}\nstdout:\n${result.stdout}`);
    }
}

function resultKey(result) {
    return `${result.game}#${result.level}`;
}

const native = runJson(solverPath, [fixtureDir, '--timeout-ms', '1000', '--json'], 'native solver');
const js = runJson(process.execPath, [jsSolverPath, fixtureDir, '--timeout-ms', '1000', '--json'], 'JS solver');

const nativeByKey = new Map(native.results.map((result) => [resultKey(result), result]));
const jsByKey = new Map(js.results.map((result) => [resultKey(result), result]));
const expected = new Map([
    ['impossible.txt#0', { status: 'exhausted', solution: [] }],
    ['message_skip.txt#0', { status: 'skipped_message', solution: [] }],
    ['message_skip.txt#1', { status: 'solved', solution: ['right'] }],
    ['multi_level.txt#0', { status: 'solved', solution: ['right'] }],
    ['multi_level.txt#1', { status: 'solved', solution: ['left'] }],
    ['one_move.txt#0', { status: 'solved', solution: ['right'] }],
]);

for (const [key, expectation] of expected) {
    const nativeResult = nativeByKey.get(key);
    const jsResult = jsByKey.get(key);
    if (!nativeResult) {
        throw new Error(`native result missing ${key}`);
    }
    if (!jsResult) {
        throw new Error(`JS result missing ${key}`);
    }
    if (nativeResult.status !== expectation.status) {
        throw new Error(`native ${key} status ${nativeResult.status}, expected ${expectation.status}`);
    }
    if (jsResult.status !== expectation.status) {
        throw new Error(`JS ${key} status ${jsResult.status}, expected ${expectation.status}`);
    }
    const nativeSolution = nativeResult.solution || [];
    const jsSolution = jsResult.solution || [];
    if (JSON.stringify(nativeSolution) !== JSON.stringify(expectation.solution)) {
        throw new Error(`native ${key} solution ${JSON.stringify(nativeSolution)}, expected ${JSON.stringify(expectation.solution)}`);
    }
    if (JSON.stringify(jsSolution) !== JSON.stringify(expectation.solution)) {
        throw new Error(`JS ${key} solution ${JSON.stringify(jsSolution)}, expected ${JSON.stringify(expectation.solution)}`);
    }
    if (JSON.stringify(nativeSolution) !== JSON.stringify(jsSolution)) {
        throw new Error(`${key} solution mismatch native=${JSON.stringify(nativeSolution)} js=${JSON.stringify(jsSolution)}`);
    }
}

process.stdout.write(`solver_parity_smoke passed cases=${expected.size}\n`);
