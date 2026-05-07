#!/usr/bin/env node
'use strict';

const assert = require('assert');
const { spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

function writeJson(dir, name, payload) {
    const file = path.join(dir, name);
    fs.writeFileSync(file, JSON.stringify(payload), 'utf8');
    return file;
}

function runComparator(baseline, optimized) {
    return spawnSync(
        process.execPath,
        [path.join(__dirname, 'compare_solver_static_opt_runs.js'), baseline, optimized],
        { encoding: 'utf8' },
    );
}

function main() {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ps-static-opt-compare-'));
    const baseline = writeJson(dir, 'baseline.json', {
        results: [
            { game: 'a.txt', level: 0, status: 'solved', solution_length: 3 },
            { game: 'a.txt', level: 1, status: 'solved', solution_length: 4 },
        ],
        totals: { levels: 2, solved: 2, timeout: 0, exhausted: 0, skipped_message: 0, errors: 0 },
    });
    const optimizedBad = writeJson(dir, 'optimized-bad.json', {
        results: [
            { game: 'a.txt', level: 0, status: 'solved', solution_length: 3 },
            { game: 'a.txt', level: -1, status: 'compile_error' },
        ],
        totals: { levels: 1, solved: 1, timeout: 0, exhausted: 0, skipped_message: 0, errors: 1 },
    });
    const bad = runComparator(baseline, optimizedBad);
    assert.notStrictEqual(bad.status, 0, 'status regressions must fail the comparison gate');
    assert.match(bad.stderr + bad.stdout, /static optimization comparison failed/);

    const optimizedGood = writeJson(dir, 'optimized-good.json', {
        results: [
            { game: 'a.txt', level: 0, status: 'solved', solution_length: 3 },
            { game: 'a.txt', level: 1, status: 'solved', solution_length: 4 },
        ],
        totals: { levels: 2, solved: 2, timeout: 0, exhausted: 0, skipped_message: 0, errors: 0 },
    });
    const good = runComparator(baseline, optimizedGood);
    assert.strictEqual(good.status, 0, good.stderr || good.stdout);

    process.stdout.write('compare_solver_static_opt_runs_node: ok\n');
}

main();
