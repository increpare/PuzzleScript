#!/usr/bin/env node
'use strict';

const assert = require('assert');

const {
    runSimulationWithStaticChecks,
} = require('./run_static_analysis_runtime_contracts_node');
const { loadPuzzleScript } = require('./js_oracle/lib/puzzlescript_node_env');

loadPuzzleScript({ includeTests: true, messageSink: [] });

const sokoban = global.testdata.find(entry => entry[0] === 'sokoban with win condition');
assert.ok(sokoban, 'sokoban fixture should be available');

const result = runSimulationWithStaticChecks(sokoban[0], sokoban[1]);

assert.strictEqual(result.staticObjectCount, 3, 'sokoban should have three static objects');
assert.strictEqual(result.countInvariantObjectCount, 5, 'sokoban should have five count-invariant objects');
assert.ok(
    result.countBoundaryChecks > result.objectBoundaryChecks,
    'count-invariant checks should include movable count-preserved objects'
);

console.log('run_static_analysis_runtime_contracts_node_test: ok');
