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
const autowin = global.testdata.find(entry => entry[0] === 'Autowin');
assert.ok(autowin, 'Autowin fixture should be available');

const result = runSimulationWithStaticChecks(sokoban[0], sokoban[1]);

assert.strictEqual(result.staticObjectCount, 3, 'sokoban should have three static objects');
assert.strictEqual(result.staticLayerCount, 2, 'sokoban should have two static layers');
assert.strictEqual(result.inertLayerCount, 1, 'sokoban should have one inert collision layer');
assert.strictEqual(result.constantQuantityObjectCount, 5, 'sokoban should have five constant-quantity objects');
assert.strictEqual(result.actionNoopProved, true, 'sokoban should prove action-noop');
assert.strictEqual(result.tickNoopProved, true, 'sokoban should have no autonomous tick rules');
assert.strictEqual(result.noAgainProved, true, 'sokoban should have no AGAIN rules');
assert.strictEqual(result.noRandomProved, true, 'sokoban should have no random rules or random RHS objects');
assert.ok(
    result.quantityBoundaryChecks > result.objectBoundaryChecks,
    'quantity checks should include movable constant-quantity objects'
);
assert.ok(
    result.staticLayerBoundaryChecks > 0,
    'static layer checks should include stable replay boundaries'
);
assert.ok(
    result.inertLayerBoundaryChecks > 0,
    'inert layer checks should include stable replay boundaries'
);
assert.ok(
    result.actionNoopBoundaryChecks > 0,
    'action-noop checks should probe action at stable replay boundaries'
);
assert.ok(
    result.tickNoopBoundaryChecks > 0,
    'tick-noop checks should probe no-input ticks at stable replay boundaries'
);
assert.ok(
    result.noAgainBoundaryChecks > 0,
    'no-again checks should include replay boundaries'
);
assert.ok(
    result.noRandomReplayChecks > 0,
    'no-random checks should compare replay boundaries under an alternate seed'
);

const autowinResult = runSimulationWithStaticChecks(autowin[0], autowin[1]);
assert.strictEqual(autowinResult.actionNoopProved, true, 'Autowin should prove action-noop');
assert.ok(
    autowinResult.actionNoopBoundaryChecks > 0,
    'action-noop checks should ignore pre-existing message text while probing solver state'
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

const temporaryBoundarySource = [
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
    'Spark',
    'Yellow',
    '',
    '=======',
    'LEGEND',
    '=======',
    '',
    'P = Player',
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
    'Spark',
    '',
    '======',
    'RULES',
    '======',
    '',
    '[ action Player ] -> [ Player Spark ]',
    'late [ Spark ] -> []',
    '',
    '==============',
    'WINCONDITIONS',
    '==============',
    '',
    'Some Player',
    '',
    '=======',
    'LEVELS',
    '=======',
    '',
    'P',
].join('\n');

const temporaryBoundary = runSimulationWithStaticChecks('temporary boundary', [
    temporaryBoundarySource,
    [4],
    'background player:0,\n',
]);

assert.strictEqual(temporaryBoundary.temporaryObjectCount, 1, 'temporary fixture should have one temporary object');
assert.ok(
    temporaryBoundary.temporaryBoundaryChecks > 0,
    'temporary checks should run for temporary objects'
);

console.log('run_static_analysis_runtime_contracts_node_test: ok');
