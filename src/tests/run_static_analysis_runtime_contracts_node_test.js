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
assert.strictEqual(result.constantQuantityObjectCount, 5, 'sokoban should have five constant-quantity objects');
assert.strictEqual(result.noRandomProved, true, 'sokoban should have no random rules or random RHS objects');
assert.ok(
    result.quantityBoundaryChecks > result.objectBoundaryChecks,
    'quantity checks should include movable constant-quantity objects'
);
assert.ok(
    result.noRandomReplayChecks > 0,
    'no-random checks should compare replay boundaries under an alternate seed'
);

const restartBoundarySource = [
    '========',
    'OBJECTS',
    '========',
    '',
    'Background',
    'Black',
    '',
    'Player',
    'Pink',
    '',
    'Marker',
    'Yellow',
    '',
    '=======',
    'LEGEND',
    '=======',
    '',
    'X = Player and Marker',
    '',
    '======',
    'SOUNDS',
    '======',
    '',
    '================',
    'COLLISIONLAYERS',
    '================',
    '',
    'Background',
    'Player',
    'Marker',
    '',
    '======',
    'RULES',
    '======',
    '',
    '[ action Player Marker ] -> [ Player ]',
    '[ action Player no Marker ] -> restart',
    '',
    '==============',
    'WINCONDITIONS',
    '==============',
    '',
    '=======',
    'LEVELS',
    '=======',
    '',
    'X',
].join('\n');

const restartBoundary = runSimulationWithStaticChecks('quantity semantic restart boundary', [
    restartBoundarySource,
    [4, 4],
    'background marker player:0,\n',
]);

assert.ok(
    restartBoundary.quantityBoundaryChecks > 0,
    'semantic restart regression should exercise quantity contract checks before restart'
);

console.log('run_static_analysis_runtime_contracts_node_test: ok');
