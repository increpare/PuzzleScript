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
            unavailableReason: `${report.status}: ${expected.diagnostic}`,
        };
    }
    return {
        objectNames: ((report.ps_tagged && report.ps_tagged.objects) || [])
            .filter(object => object.tags && object.tags.static === true)
            .map(object => object.name),
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

    const previousUnitTesting = unitTesting;
    const previousLazyFunctionGeneration = lazyFunctionGeneration;
    unitTesting = true;
    lazyFunctionGeneration = false;

    let objectBoundaryChecks = 0;
    try {
        compileSimulationSource(testName, source, targetLevel, randomSeed);

        let currentIdentity = boardIdentity();
        let snapshots = snapshotStaticObjects(staticObjects);

        for (let inputIndex = 0; inputIndex < inputs.length; inputIndex++) {
            const inputToken = inputs[inputIndex];
            const result = executeInputToken(inputToken);
            drainAgain(`${testName}: input ${inputIndex} ${tokenLabel(inputToken)}`);

            const nextIdentity = boardIdentity();
            const resetBoundary =
                result.resetsSnapshot
                || !sameBoardIdentity(currentIdentity, nextIdentity)
                || !canSnapshotBoard();

            if (resetBoundary) {
                currentIdentity = nextIdentity;
                snapshots = snapshotStaticObjects(staticObjects);
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

            objectBoundaryChecks += staticObjects.length;
            currentIdentity = nextIdentity;
        }

        assertFinalReplayParity(testName, expectedSerializedLevel, expectedSounds);

        return {
            staticObjectCount: staticObjects.length,
            objectBoundaryChecks,
            analysisUnavailableReason: staticContract.unavailableReason,
        };
    } finally {
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
    let objectBoundaryChecks = 0;
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
            objectBoundaryChecks += result.objectBoundaryChecks;
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
        objectBoundaryChecks,
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
        `static_analysis_runtime_contracts: ok (${result.caseCount} cases, ${result.analysisUnavailableCount} analysis-unavailable, ${result.casesWithStaticObjects} with static objects, ${result.objectBoundaryChecks} object-boundary checks)`
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
    firstSnapshotDifference,
    parseArgs,
    runAll,
    runSimulationWithStaticChecks,
    snapshotStaticObjects,
    staticContractForSource,
};
