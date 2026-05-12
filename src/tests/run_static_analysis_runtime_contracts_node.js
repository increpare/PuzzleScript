#!/usr/bin/env node
'use strict';

const assert = require('assert');

const { analyzeSource } = require('./ps_static_analysis');
const { loadPuzzleScript } = require('./js_oracle/lib/puzzlescript_node_env');

let runtimeLoaded = false;
const MAX_AGAIN_DRAIN_STEPS = 10000;

const ANALYSIS_UNAVAILABLE_TESTS = new Map([
    ['by your side', { status: 'compile_error', diagnostic: 'Object "TARGET" included in multiple collision layers' }],
    ['test testing starting at level N', { status: 'compile_error', diagnostic: 'Object "TARGET" included in multiple collision layers' }],
    ['collapse simple', { status: 'compile_error', diagnostic: 'YouTube support' }],
    ['collapse long', { status: 'compile_error', diagnostic: 'YouTube support' }],
    ['damn I\'m huge', { status: 'compile_error', diagnostic: 'YouTube support' }],
    ['rule application hat test', { status: 'compile_error', diagnostic: 'Object "WALL" included in multiple collision layers' }],
    ['too many rigid bodies', { status: 'compile_error', diagnostic: 'Name "1" already in use' }],
    ['dang I\'m huge', { status: 'compile_error', diagnostic: 'YouTube support' }],
    ['Drop Swap', { status: 'compile_error', diagnostic: 'Name "P" already in use' }],
    ['Drop Swap 2', { status: 'compile_error', diagnostic: 'Name "P" already in use' }],
    ['Drop Swap 3', { status: 'compile_error', diagnostic: 'Name "P" already in use' }],
    ['a = b and b #393', { status: 'compile_error', diagnostic: 'Trying to create an aggregate object' }],
    ['gallery:cyber-lasso', { status: 'compile_error', diagnostic: 'You named an object "|", but this is a keyword' }],
    ['Putting Bicycle Helmets on Young Children', { status: 'compile_error', diagnostic: 'strap occurs multiple times' }],
    ['[testing for recording through level-changes A]  Level-Change test', { status: 'compile_error', diagnostic: 'unexpected sound token "un"' }],
    ['parser rigid in strange place highlighting test', { status: 'compile_error', diagnostic: 'malformed cell rule' }],
]);

function parseArgs(argv) {
    const options = { filter: null, help: false };
    for (let index = 2; index < argv.length; index++) {
        const arg = argv[index];
        if (arg === '--help' || arg === '-h') {
            options.help = true;
        } else if (arg === '--filter') {
            assert.ok(index + 1 < argv.length, '--filter requires a value');
            options.filter = argv[++index];
        } else if (!arg.startsWith('-') && options.filter === null) {
            options.filter = arg;
        } else {
            throw new Error(`Unexpected argument: ${arg}`);
        }
    }
    return options;
}

function usage() {
    return [
        'Usage: node src/tests/run_static_analysis_runtime_contracts_node.js [--filter NAME]',
        '',
        'Replays src/tests/resources/testdata.js and checks static object occupancy invariants.',
    ].join('\n');
}

function ensureRuntimeLoaded() {
    if (!runtimeLoaded) {
        loadPuzzleScript({ includeTests: true, messageSink: [] });
        runtimeLoaded = true;
    }
}

function resetParserErrors() {
    if (typeof resetParserErrorState === 'function') {
        resetParserErrorState();
    } else {
        errorStrings = [];
        errorCount = 0;
    }
}

function diagnosticText(errors) {
    return (errors || []).map(error => String(error).replace(/<[^>]*>/g, ' ')).join('\n');
}

function drainAgain(context) {
    let stepCount = 0;
    while (againing) {
        stepCount++;
        if (stepCount > MAX_AGAIN_DRAIN_STEPS) {
            throw new Error(`${context}: exceeded ${MAX_AGAIN_DRAIN_STEPS} again-drain steps`);
        }
        againing = false;
        processInput(-1);
    }
}

function staticContractForSource(source, testName) {
    const sourcePath = `testdata:${testName}`;
    const report = analyzeSource(source, {
        sourcePath,
        familyFilter: 'count_layer_invariants',
    });
    if (report.status !== 'ok') {
        const expected = ANALYSIS_UNAVAILABLE_TESTS.get(testName);
        if (!expected) {
            throw new Error(`${sourcePath}: static analysis status ${report.status}`);
        }
        if (report.status !== expected.status) {
            throw new Error(`${sourcePath}: static analysis status ${report.status}, expected ${expected.status}`);
        }
        const diagnostics = diagnosticText(report.errors);
        if (!diagnostics.includes(expected.diagnostic)) {
            throw new Error(`${sourcePath}: static analysis diagnostic changed; expected ${JSON.stringify(expected.diagnostic)}`);
        }
        return {
            objectNames: [],
            inertLayerIds: [],
            constantQuantityObjectNames: [],
            quantityContracts: [],
            unavailableReason: `${report.status}: ${expected.diagnostic}`,
        };
    }
    const objects = ((report.ps_tagged && report.ps_tagged.objects) || []);
    const layers = ((report.ps_tagged && report.ps_tagged.collision_layers) || []);
    const quantityContracts = objects
        .filter(object => object.tags && object.tags.quantity)
        .map(object => ({
            objectName: object.name,
            neverIncreases: object.tags.quantity.never_increases === true,
            neverDecreases: object.tags.quantity.never_decreases === true,
        }))
        .filter(contract => contract.neverIncreases || contract.neverDecreases);
    return {
        objectNames: objects
            .filter(object => object.tags && object.tags.static === true)
            .map(object => object.name),
        inertLayerIds: layers
            .filter(layer => layer.tags && layer.tags.inert === true)
            .map(layer => layer.id),
        constantQuantityObjectNames: quantityContracts
            .filter(contract => contract.neverIncreases && contract.neverDecreases)
            .map(contract => contract.objectName),
        quantityContracts,
        unavailableReason: null,
    };
}

function engineObjectName(displayName) {
    const target = String(displayName).toLowerCase();
    const match = Object.keys(state.objects || {}).find(name =>
        name === target
        || (state.original_case_names && state.original_case_names[name] === displayName)
    );
    if (!match) {
        const available = Object.keys(state.objects || {}).sort().join(', ');
        throw new Error(`runtime object not found for static analyser object ${JSON.stringify(displayName)}; available: ${available}`);
    }
    return match;
}

function canSnapshotBoard() {
    return level && Number.isInteger(level.n_tiles);
}

function boardIdentity() {
    if (!canSnapshotBoard()) {
        return {
            available: false,
            textMode: Boolean(textMode),
            titleScreen: Boolean(titleScreen),
        };
    }
    return {
        available: true,
        curlevel: typeof curlevel === 'number' ? curlevel : null,
        curlevelTarget: typeof curlevelTarget === 'number' ? curlevelTarget : null,
        width: level.width,
        height: level.height,
        nTiles: level.n_tiles,
        textMode: Boolean(textMode),
        titleScreen: Boolean(titleScreen),
    };
}

function sameBoardIdentity(left, right) {
    return JSON.stringify(left) === JSON.stringify(right);
}

function objectOccupancySnapshot(displayName) {
    if (!canSnapshotBoard()) {
        throw new Error(`cannot snapshot ${displayName}: no active board level`);
    }
    const runtimeName = engineObjectName(displayName);
    const object = state.objects[runtimeName];
    const cells = [];
    for (let cellIndex = 0; cellIndex < level.n_tiles; cellIndex++) {
        cells.push(level.getCell(cellIndex).get(object.id) ? 1 : 0);
    }
    return cells;
}

function snapshotStaticObjects(objectNames) {
    const snapshots = new Map();
    if (!canSnapshotBoard()) return snapshots;
    for (const objectName of objectNames) {
        snapshots.set(objectName, objectOccupancySnapshot(objectName));
    }
    return snapshots;
}

function layerOccupancySnapshot(layerId) {
    if (!canSnapshotBoard()) {
        throw new Error(`cannot snapshot layer ${layerId}: no active board level`);
    }
    const objectNames = Array.from(state.collisionLayers[layerId] || []);
    const snapshots = [];
    for (let cellIndex = 0; cellIndex < level.n_tiles; cellIndex++) {
        const cell = level.getCell(cellIndex);
        snapshots.push(objectNames
            .filter(objectName => cell.get(state.objects[objectName].id))
            .sort()
            .join('|'));
    }
    return snapshots;
}

function snapshotLayers(layerIds) {
    const snapshots = new Map();
    if (!canSnapshotBoard()) return snapshots;
    for (const layerId of layerIds) {
        snapshots.set(layerId, layerOccupancySnapshot(layerId));
    }
    return snapshots;
}

function objectCountSnapshot(displayName) {
    return objectOccupancySnapshot(displayName).reduce((sum, present) => sum + present, 0);
}

function snapshotObjectCounts(objectNames) {
    const snapshots = new Map();
    if (!canSnapshotBoard()) return snapshots;
    for (const objectName of objectNames) {
        snapshots.set(objectName, objectCountSnapshot(objectName));
    }
    return snapshots;
}

function firstSnapshotDifference(beforeSnapshots, objectNames) {
    for (const objectName of objectNames) {
        const before = beforeSnapshots.get(objectName) || [];
        const after = objectOccupancySnapshot(objectName);
        const length = Math.max(before.length, after.length);
        for (let cellIndex = 0; cellIndex < length; cellIndex++) {
            const beforeValue = before[cellIndex] || 0;
            const afterValue = after[cellIndex] || 0;
            if (beforeValue !== afterValue) {
                return {
                    objectName,
                    cellIndex,
                    before: beforeValue,
                    after: afterValue,
                };
            }
        }
    }
    return null;
}

function firstLayerSnapshotDifference(beforeSnapshots, layerIds) {
    for (const layerId of layerIds) {
        const before = beforeSnapshots.get(layerId) || [];
        const after = layerOccupancySnapshot(layerId);
        const length = Math.max(before.length, after.length);
        for (let cellIndex = 0; cellIndex < length; cellIndex++) {
            const beforeValue = before[cellIndex] || '';
            const afterValue = after[cellIndex] || '';
            if (beforeValue !== afterValue) {
                return {
                    layerId,
                    cellIndex,
                    before: beforeValue,
                    after: afterValue,
                };
            }
        }
    }
    return null;
}

function firstQuantityDifference(beforeCounts, quantityContracts) {
    for (const contract of quantityContracts) {
        const before = beforeCounts.get(contract.objectName) || 0;
        const after = objectCountSnapshot(contract.objectName);
        if (contract.neverIncreases && after > before) {
            return {
                objectName: contract.objectName,
                before,
                after,
                claim: 'quantity.never_increases',
            };
        }
        if (contract.neverDecreases && after < before) {
            return {
                objectName: contract.objectName,
                before,
                after,
                claim: 'quantity.never_decreases',
            };
        }
    }
    return null;
}

function quantityClaimCount(quantityContracts) {
    let count = 0;
    for (const contract of quantityContracts) {
        if (contract.neverIncreases) count++;
        if (contract.neverDecreases) count++;
    }
    return count;
}

function quantityObjectNames(quantityContracts) {
    return quantityContracts.map(contract => contract.objectName);
}

function executeInputToken(inputToken) {
    if (inputToken === 'undo') {
        DoUndo(false, true);
        return { resetsSnapshot: true };
    }
    if (inputToken === 'restart') {
        DoRestart();
        return { resetsSnapshot: true };
    }
    if (inputToken === 'tick') {
        processInput(-1);
        return { resetsSnapshot: false };
    }
    processInput(inputToken);
    return { resetsSnapshot: false };
}

function compileSimulationSource(testName, source, targetLevel, randomSeed) {
    levelString = source;
    resetParserErrors();
    compile(['loadLevel', targetLevel], source, randomSeed);
    drainAgain(`${testName}: initial compile`);
}

function tokenLabel(inputToken) {
    return typeof inputToken === 'string' ? inputToken : String(inputToken);
}

function assertFinalReplayParity(testName, expectedSerializedLevel, expectedSounds) {
    const actualSerializedLevel = convertLevelToString();
    if (actualSerializedLevel !== expectedSerializedLevel) {
        throw new Error(`${testName}: final serialized level differs from simulation expectation`);
    }

    if (expectedSounds !== null) {
        const actualSounds = soundHistory.join(';');
        const expectedSoundText = expectedSounds.join(';');
        if (actualSounds !== expectedSoundText) {
            throw new Error(`${testName}: sound output expected ${JSON.stringify(expectedSoundText)}, got ${JSON.stringify(actualSounds)}`);
        }
    }
}

function runSimulationWithStaticChecks(testName, dataarray) {
    const source = dataarray[0];
    const inputs = dataarray[1];
    const expectedSerializedLevel = dataarray[2];
    const targetLevel = dataarray[3] === undefined ? 0 : dataarray[3];
    const randomSeed = dataarray[4] === undefined ? null : dataarray[4];
    const expectedSounds = dataarray[5] === undefined ? null : dataarray[5];
    const staticContract = staticContractForSource(source, testName);
    const staticObjects = staticContract.objectNames;
    const inertLayers = staticContract.inertLayerIds;
    const constantQuantityObjects = staticContract.constantQuantityObjectNames;
    const quantityContracts = staticContract.quantityContracts;
    const countedObjects = quantityObjectNames(quantityContracts);

    const previousUnitTesting = unitTesting;
    const previousLazyFunctionGeneration = lazyFunctionGeneration;
    unitTesting = true;
    lazyFunctionGeneration = false;

    let objectBoundaryChecks = 0;
    let inertLayerBoundaryChecks = 0;
    let quantityBoundaryChecks = 0;
    let restartBoundaryTriggered = false;
    const previousDoRestart = global.DoRestart;
    if (typeof previousDoRestart === 'function') {
        global.DoRestart = function (...args) {
            restartBoundaryTriggered = true;
            return previousDoRestart.apply(this, args);
        };
    }
    try {
        compileSimulationSource(testName, source, targetLevel, randomSeed);

        let currentIdentity = boardIdentity();
        let snapshots = snapshotStaticObjects(staticObjects);
        let layerSnapshots = snapshotLayers(inertLayers);
        let countSnapshots = snapshotObjectCounts(countedObjects);

        for (let inputIndex = 0; inputIndex < inputs.length; inputIndex++) {
            const inputToken = inputs[inputIndex];
            restartBoundaryTriggered = false;
            const result = executeInputToken(inputToken);
            drainAgain(`${testName}: input ${inputIndex} ${tokenLabel(inputToken)}`);

            const nextIdentity = boardIdentity();
            const resetBoundary =
                result.resetsSnapshot
                || restartBoundaryTriggered
                || !sameBoardIdentity(currentIdentity, nextIdentity)
                || !canSnapshotBoard();

            if (resetBoundary) {
                currentIdentity = nextIdentity;
                snapshots = snapshotStaticObjects(staticObjects);
                layerSnapshots = snapshotLayers(inertLayers);
                countSnapshots = snapshotObjectCounts(countedObjects);
                continue;
            }

            const diff = firstSnapshotDifference(snapshots, staticObjects);
            if (diff) {
                throw new Error([
                    `${testName}: static object occupancy changed`,
                    `  input ${inputIndex}: ${tokenLabel(inputToken)}`,
                    `  object: ${diff.objectName}`,
                    `  cell: ${diff.cellIndex}`,
                    `  before: ${diff.before}`,
                    `  after: ${diff.after}`,
                ].join('\n'));
            }

            const layerDiff = firstLayerSnapshotDifference(layerSnapshots, inertLayers);
            if (layerDiff) {
                throw new Error([
                    `${testName}: inert layer occupancy changed`,
                    `  input ${inputIndex}: ${tokenLabel(inputToken)}`,
                    `  layer: ${layerDiff.layerId}`,
                    `  cell: ${layerDiff.cellIndex}`,
                    `  before: ${layerDiff.before}`,
                    `  after: ${layerDiff.after}`,
                ].join('\n'));
            }

            const countDiff = firstQuantityDifference(countSnapshots, quantityContracts);
            if (countDiff) {
                throw new Error([
                    `${testName}: quantity monotonicity claim violated`,
                    `  input ${inputIndex}: ${tokenLabel(inputToken)}`,
                    `  claim: ${countDiff.claim}`,
                    `  object: ${countDiff.objectName}`,
                    `  before: ${countDiff.before}`,
                    `  after: ${countDiff.after}`,
                ].join('\n'));
            }

            objectBoundaryChecks += staticObjects.length;
            inertLayerBoundaryChecks += inertLayers.length;
            quantityBoundaryChecks += quantityClaimCount(quantityContracts);
            countSnapshots = snapshotObjectCounts(countedObjects);
            currentIdentity = nextIdentity;
        }

        assertFinalReplayParity(testName, expectedSerializedLevel, expectedSounds);

        return {
            staticObjectCount: staticObjects.length,
            inertLayerCount: inertLayers.length,
            constantQuantityObjectCount: constantQuantityObjects.length,
            objectBoundaryChecks,
            inertLayerBoundaryChecks,
            quantityBoundaryChecks,
            analysisUnavailableReason: staticContract.unavailableReason,
        };
    } finally {
        if (typeof previousDoRestart === 'function') {
            global.DoRestart = previousDoRestart;
        }
        unitTesting = previousUnitTesting;
        lazyFunctionGeneration = previousLazyFunctionGeneration;
    }
}

function testMatchesFilter(testName, filter) {
    return !filter || testName.toLowerCase().includes(filter.toLowerCase());
}

function progressLog(options, message) {
    if (options.progress) {
        console.error(message);
    }
}

function runAll(options = {}) {
    ensureRuntimeLoaded();
    assert.ok(Array.isArray(global.testdata), 'global.testdata should be loaded');

    const failures = [];
    let caseCount = 0;
    let casesWithStaticObjects = 0;
    let casesWithInertLayers = 0;
    let casesWithConstantQuantityObjects = 0;
    let objectBoundaryChecks = 0;
    let inertLayerBoundaryChecks = 0;
    let quantityBoundaryChecks = 0;
    let analysisUnavailableCount = 0;
    const entries = global.testdata.filter(entry => testMatchesFilter(entry[0], options.filter || null));

    if (entries.length === 0) {
        throw new Error(options.filter
            ? `No simulation tests matched filter ${JSON.stringify(options.filter)}`
            : 'No simulation tests were loaded');
    }

    progressLog(options, `static_analysis_runtime_contracts: running ${entries.length} simulation cases`);

    for (const [entryIndex, entry] of entries.entries()) {
        const testName = entry[0];
        const dataarray = entry[1];

        caseCount++;
        progressLog(options, `static_analysis_runtime_contracts: [${entryIndex + 1}/${entries.length}] ${testName}`);
        try {
            const result = runSimulationWithStaticChecks(testName, dataarray);
            if (result.staticObjectCount > 0) {
                casesWithStaticObjects++;
            }
            if (result.inertLayerCount > 0) {
                casesWithInertLayers++;
            }
            if (result.constantQuantityObjectCount > 0) {
                casesWithConstantQuantityObjects++;
            }
            objectBoundaryChecks += result.objectBoundaryChecks;
            inertLayerBoundaryChecks += result.inertLayerBoundaryChecks;
            quantityBoundaryChecks += result.quantityBoundaryChecks;
            if (result.analysisUnavailableReason) {
                analysisUnavailableCount++;
                progressLog(options, `static_analysis_runtime_contracts:   static analysis unavailable: ${result.analysisUnavailableReason}`);
            }
        } catch (error) {
            failures.push(`ERROR: ${testName}: ${error.message}`);
        }
    }

    return {
        ok: failures.length === 0,
        caseCount,
        casesWithStaticObjects,
        casesWithInertLayers,
        casesWithConstantQuantityObjects,
        objectBoundaryChecks,
        inertLayerBoundaryChecks,
        quantityBoundaryChecks,
        analysisUnavailableCount,
        failures,
    };
}

function main() {
    const options = parseArgs(process.argv);
    if (options.help) {
        console.log(usage());
        return 0;
    }

    const result = runAll({ ...options, progress: true });
    if (!result.ok) {
        console.error('static_analysis_runtime_contracts: failed');
        for (const failure of result.failures) {
            console.error(failure);
        }
        return 1;
    }

    console.log(
        `static_analysis_runtime_contracts: ok (${result.caseCount} cases, ${result.analysisUnavailableCount} analysis-unavailable, ${result.casesWithStaticObjects} with static objects, ${result.casesWithInertLayers} with inert layers, ${result.casesWithConstantQuantityObjects} with constant-quantity objects, ${result.objectBoundaryChecks} object-boundary checks, ${result.inertLayerBoundaryChecks} inert-layer-boundary checks, ${result.quantityBoundaryChecks} quantity-boundary checks)`
    );
    return 0;
}

if (require.main === module) {
    try {
        process.exitCode = main();
    } catch (error) {
        console.error(error.stack || error.message);
        process.exitCode = 1;
    }
}

module.exports = {
    ANALYSIS_UNAVAILABLE_TESTS,
    MAX_AGAIN_DRAIN_STEPS,
    boardIdentity,
    engineObjectName,
    firstLayerSnapshotDifference,
    firstQuantityDifference,
    firstSnapshotDifference,
    layerOccupancySnapshot,
    objectCountSnapshot,
    parseArgs,
    runAll,
    runSimulationWithStaticChecks,
    snapshotObjectCounts,
    snapshotLayers,
    snapshotStaticObjects,
    staticContractForSource,
};
