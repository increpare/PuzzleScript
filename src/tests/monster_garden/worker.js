#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const garden = require('./garden');

const srcDir = path.join(__dirname, '..', '..');

const _storage = {};
global.localStorage = {
    getItem(key) { return _storage.hasOwnProperty(key) ? _storage[key] : null; },
    setItem(key, value) { _storage[key] = String(value); },
    removeItem(key) { delete _storage[key]; }
};

global.document = {
    URL: 'test://',
    body: {
        classList: { contains() { return false; } },
        addEventListener() {},
        removeEventListener() {}
    },
    createElement() {
        return { style: {}, innerHTML: '', textContent: '', getContext() { return null; } };
    },
    getElementById() { return null; }
};

global.window = global;
global.lastDownTarget = null;
global.canvas = null;
global.input = global.document.createElement('TEXTAREA');
global.canvasResize = function() {};
global.redraw = function() {};
global.forceRegenImages = function() {};
global.consolePrintFromRule = function() {};
global.consolePrint = function() {};
global.console_print_raw = function() {};
global.consoleError = function() {};
global.consoleCacheDump = function() {};
global.addToDebugTimeline = function() {};
global.killAudioButton = function() {};
global.showAudioButton = function() {};
global.regenSpriteImages = function() {};
global.jumpToLine = function() {};
global.printLevel = function() {};
global.playSound = function() {};
global.levelString = '';
global.editor = { getValue() { return global.levelString; } };
global.PuzzleScriptTestAssertions = { push() {}, equal() {} };
global.UnitTestingThrow = function(error) { throw error; };

const sourceFiles = [
    'js/storagewrapper.js',
    'js/bitvec.js',
    'js/level.js',
    'js/languageConstants.js',
    'js/globalVariables.js',
    'js/debug.js',
    'js/font.js',
    'js/rng.js',
    'js/riffwave.js',
    'js/sfxr.js',
    'js/codemirror/stringstream.js',
    'js/colorhelpers.js',
    'js/colors.js',
    'js/engine.js',
    'js/parser.js',
    'js/compiler.js',
    'js/soundbar.js'
];

let allCode = '';
for (let i = 0; i < sourceFiles.length; i++) {
    allCode += '\n// ---- ' + sourceFiles[i] + ' ----\n';
    allCode += fs.readFileSync(path.join(srcDir, sourceFiles[i]), 'utf8') + '\n';
}
// Top-level let/const in the concatenated compiler are script-scoped, not
// global properties. Bridge the ones the worker reads and writes.
allCode += '\n' + [
    'Object.defineProperties(global, {',
    '    unitTesting: { get: function() { return unitTesting; }, set: function(v) { unitTesting = v; }, configurable: true },',
    '    lazyFunctionGeneration: { get: function() { return lazyFunctionGeneration; }, set: function(v) { lazyFunctionGeneration = v; }, configurable: true },',
    '    IDE: { get: function() { return IDE; }, set: function(v) { IDE = v; }, configurable: true },',
    '    errorCount: { get: function() { return errorCount; }, set: function(v) { errorCount = v; }, configurable: true },',
    '    errorStrings: { get: function() { return errorStrings; }, set: function(v) { errorStrings = v; }, configurable: true },',
    '    level: { get: function() { return level; }, set: function(v) { level = v; }, configurable: true },',
    '    curlevel: { get: function() { return curlevel; }, set: function(v) { curlevel = v; }, configurable: true },',
    '    curlevelTarget: { get: function() { return curlevelTarget; }, set: function(v) { curlevelTarget = v; }, configurable: true },',
    '    winning: { get: function() { return winning; }, set: function(v) { winning = v; }, configurable: true },',
    '    messageselected: { get: function() { return messageselected; }, set: function(v) { messageselected = v; }, configurable: true },',
    '    messagetext: { get: function() { return messagetext; }, set: function(v) { messagetext = v; }, configurable: true },',
    '    hasUsedCheckpoint: { get: function() { return hasUsedCheckpoint; }, set: function(v) { hasUsedCheckpoint = v; }, configurable: true },',
    '    loadedLevelSeed: { get: function() { return loadedLevelSeed; }, set: function(v) { loadedLevelSeed = v; }, configurable: true },',
    '    againing: { get: function() { return againing; }, set: function(v) { againing = v; }, configurable: true },',
    '    backups: { get: function() { return backups; }, set: function(v) { backups = v; }, configurable: true },',
    '    RandomGen: { get: function() { return RandomGen; }, set: function(v) { RandomGen = v; }, configurable: true },',
    '    textMode: { get: function() { return textMode; }, set: function(v) { textMode = v; }, configurable: true },',
    '    titleScreen: { get: function() { return titleScreen; }, set: function(v) { titleScreen = v; }, configurable: true },',
    '    state: { get: function() { return state; }, set: function(v) { state = v; }, configurable: true },',
    '    STRIDE_OBJ: { get: function() { return STRIDE_OBJ; }, set: function(v) { STRIDE_OBJ = v; }, configurable: true },',
    '    STRIDE_MOV: { get: function() { return STRIDE_MOV; }, set: function(v) { STRIDE_MOV = v; }, configurable: true }',
    '});'
].join('\n') + '\n';
vm.runInThisContext(allCode, { filename: 'monster_garden_worker.js' });

function emit(result) {
    process.stdout.write(JSON.stringify(result) + '\n');
}

function resetWorkerRuntime() {
    const keys = Object.keys(_storage);
    for (let i = 0; i < keys.length; i++) {
        delete _storage[keys[i]];
    }
    global.backups = [];
    global.againing = false;
    global.winning = false;
    global.messageselected = false;
    global.textMode = true;
    global.titleScreen = true;
    global.hasUsedCheckpoint = false;
    global.curlevelTarget = null;
}

function snapshotRng() {
    const rng = global.RandomGen;
    if (!rng || !rng._state || !rng._state.s) {
        return null;
    }
    return {
        seed: rng.seed,
        _normal: rng._normal,
        s: rng._state.s.slice(),
        i: rng._state.i,
        j: rng._state.j
    };
}

function restoreRng(snap) {
    if (!snap || !global.RandomGen || !global.RandomGen._state) {
        return;
    }
    global.RandomGen.seed = snap.seed;
    global.RandomGen._normal = snap._normal;
    global.RandomGen._state.s = snap.s.slice();
    global.RandomGen._state.i = snap.i;
    global.RandomGen._state.j = snap.j;
}

function snapshotEngine() {
    const level = global.level;
    return {
        curlevel: global.curlevel,
        curlevelTarget: global.curlevelTarget,
        textMode: global.textMode,
        titleScreen: global.titleScreen,
        winning: global.winning,
        messageselected: global.messageselected,
        messagetext: global.messagetext,
        hasUsedCheckpoint: global.hasUsedCheckpoint,
        backups: (global.backups || []).slice(),
        rng: snapshotRng(),
        objects: level && level.objects ? new Int32Array(level.objects) : null,
        movements: level && level.movements ? new Int32Array(level.movements) : null,
        storage: Object.assign({}, _storage)
    };
}

function restoreEngine(snap) {
    if (!snap) {
        return;
    }
    global.curlevel = snap.curlevel;
    global.curlevelTarget = snap.curlevelTarget;
    global.textMode = snap.textMode;
    global.titleScreen = snap.titleScreen;
    global.winning = snap.winning;
    global.messageselected = snap.messageselected;
    global.messagetext = snap.messagetext;
    global.hasUsedCheckpoint = snap.hasUsedCheckpoint;
    global.backups = snap.backups.slice();
    restoreRng(snap.rng);
    if (global.level && snap.objects) {
        global.level.objects.set(snap.objects);
    }
    if (global.level && snap.movements && global.level.movements) {
        global.level.movements.set(snap.movements);
    }
    const keys = Object.keys(_storage);
    for (let i = 0; i < keys.length; i++) {
        delete _storage[keys[i]];
    }
    Object.keys(snap.storage).forEach(function(key) {
        _storage[key] = snap.storage[key];
    });
}

function checkPlayableInvariants(job) {
    if (isTextOrMessageLevel(job)) {
        return null;
    }
    return garden.checkLevelInvariants(global.level, global.STRIDE_OBJ, global.STRIDE_MOV, global.state);
}

function drainAgain(job) {
    while (global.againing) {
        global.againing = false;
        global.processInput(-1);
        const broken = checkPlayableInvariants(job);
        if (broken) {
            return broken;
        }
    }
    return null;
}

function isTextOrMessageLevel(job) {
    if (global.textMode || global.titleScreen) {
        return true;
    }
    const levels = global.state && global.state.levels;
    const selected = levels && job && levels[job.level];
    return !!(selected && Object.prototype.hasOwnProperty.call(selected, 'message'));
}

function canonicalEngineSeed(job) {
    if (job.engineSeed != null && String(job.engineSeed).length > 0) {
        return String(job.engineSeed);
    }
    if (job.randomSeed != null && String(job.randomSeed).length > 0) {
        return String(job.randomSeed);
    }
    return 'garden-' + String(job.level) + '-' + String((job.inputs || []).length);
}

function validateJob(job) {
    if (!Number.isInteger(job.level)) {
        throw new Error('job.level must be an integer');
    }
}

function compilerDiagnosticKind() {
    if (global.errorCount > 0) {
        return 'compiler-error';
    }
    if (global.errorStrings && global.errorStrings.length > 0) {
        return 'compiler-warning';
    }
    return null;
}

function withEngineSeed(job, result) {
    result.engineSeed = job.engineSeed;
    return result;
}

function applyInputs(inputs, job) {
    for (let i = 0; i < inputs.length; i++) {
        const value = inputs[i];
        if (value === 'undo') {
            global.DoUndo(false, true);
        } else if (value === 'restart') {
            global.DoRestart();
        } else if (value === 'tick') {
            global.processInput(-1);
        } else if (typeof value === 'number') {
            global.processInput(value);
        } else {
            throw new Error('Unknown input: ' + value);
        }
        const afterInput = checkPlayableInvariants(job);
        if (afterInput) {
            return afterInput;
        }
        const afterAgain = drainAgain(job);
        if (afterAgain) {
            return afterAgain;
        }
    }
    return null;
}

function fingerprintAfter(job) {
    const diagnostic = compilerDiagnosticKind();
    if (diagnostic) {
        return diagnostic + ':' + global.errorCount + ':' + JSON.stringify(global.errorStrings || []);
    }
    const message = isTextOrMessageLevel(job);
    return JSON.stringify({
        errorCount: global.errorCount,
        errorStrings: (global.errorStrings || []).slice(),
        curlevel: global.curlevel,
        textMode: !!global.textMode,
        titleScreen: !!global.titleScreen,
        winning: !!global.winning,
        messageselected: !!global.messageselected,
        messagetext: String(global.messagetext || ''),
        board: message ? null : global.convertLevelToString(),
        objects: message || !global.level || !global.level.objects ? null : Array.from(global.level.objects),
        movements: message || !global.level || !global.level.movements ? null : Array.from(global.level.movements),
        rng: snapshotRng()
    });
}

function runOnce(job) {
    resetWorkerRuntime();
    global.unitTesting = true;
    global.lazyFunctionGeneration = false;
    global.IDE = false;
    if (typeof global.resetParserErrorState === 'function') {
        global.resetParserErrorState();
    } else {
        global.errorStrings = [];
        global.errorCount = 0;
    }
    global.compile(['loadLevel', job.level], job.source, job.engineSeed);
    const diagnosticKind = compilerDiagnosticKind();
    if (diagnosticKind) {
        const errorCount = global.errorCount;
        const errorStrings = (global.errorStrings || []).slice();
        return withEngineSeed(job, {
            kind: diagnosticKind,
            error: null,
            fingerprint: diagnosticKind + ':' + errorCount + ':' + JSON.stringify(errorStrings),
            detail: '',
            errorCount: errorCount,
            errorStrings: errorStrings
        });
    }
    if (!global.state || !Array.isArray(global.state.levels)) {
        throw new Error('compile produced no state.levels');
    }
    if (job.level < 0 || job.level >= global.state.levels.length) {
        throw new Error('job.level ' + job.level + ' is out of range');
    }
    const errorCount = global.errorCount;
    const drainBroken = drainAgain(job);
    if (drainBroken) {
        return withEngineSeed(job, {
            kind: 'invariant',
            error: null,
            fingerprint: fingerprintAfter(job),
            detail: drainBroken,
            errorCount: errorCount
        });
    }
    const engineSnapshot = snapshotEngine();
    if (isTextOrMessageLevel(job)) {
        return withEngineSeed(job, {
            kind: 'ok',
            error: null,
            fingerprint: fingerprintAfter(job),
            detail: '',
            errorCount: errorCount,
            prefixLength: 0,
            engineSnapshot: engineSnapshot
        });
    }
    const broken = checkPlayableInvariants(job);
    if (broken) {
        return withEngineSeed(job, {
            kind: 'invariant',
            error: null,
            fingerprint: fingerprintAfter(job),
            detail: broken,
            errorCount: errorCount
        });
    }
    const prefix = (job.inputs || []).slice(0, job.maxInputs);
    const afterExec = applyInputs(prefix, job);
    if (afterExec) {
        return withEngineSeed(job, {
            kind: 'invariant',
            error: null,
            fingerprint: fingerprintAfter(job),
            detail: afterExec,
            errorCount: errorCount
        });
    }
    return withEngineSeed(job, {
        kind: 'ok',
        error: null,
        fingerprint: fingerprintAfter(job),
        detail: '',
        errorCount: errorCount,
        prefixLength: prefix.length,
        engineSnapshot: engineSnapshot
    });
}

function stripInternalFields(result) {
    if (!result) {
        return result;
    }
    delete result.undoBaseline;
    delete result.rngBaseline;
    delete result.engineSnapshot;
    return result;
}

function runJob(job) {
    try {
        job.engineSeed = canonicalEngineSeed(job);
        validateJob(job);
        const first = runOnce(job);
        if (first.kind !== 'ok') {
            return stripInternalFields(first);
        }
        const prefix = (job.inputs || []).slice(0, job.maxInputs);
        if (job.replay && prefix.length > 0 && first.prefixLength > 0) {
            restoreEngine(first.engineSnapshot);
            const replayBroken = applyInputs(prefix, job);
            if (replayBroken) {
                return withEngineSeed(job, {
                    kind: 'invariant',
                    error: null,
                    fingerprint: fingerprintAfter(job),
                    detail: replayBroken,
                    errorCount: global.errorCount
                });
            }
            const replayed = fingerprintAfter(job);
            if (replayed !== first.fingerprint) {
                return withEngineSeed(job, {
                    kind: 'replay-divergence',
                    error: null,
                    fingerprint: first.fingerprint,
                    detail: replayed,
                    errorCount: first.errorCount
                });
            }
        }
        resetWorkerRuntime();
        const second = runOnce(job);
        if (second.kind !== 'ok') {
            if (second.kind === 'crash' || second.kind === 'invariant' || second.kind === 'replay-divergence') {
                return stripInternalFields(second);
            }
            return withEngineSeed(job, {
                kind: 'nondeterministic',
                error: null,
                fingerprint: first.fingerprint,
                detail: second.kind,
                errorCount: first.errorCount
            });
        }
        if (second.fingerprint !== first.fingerprint) {
            return withEngineSeed(job, {
                kind: 'nondeterministic',
                error: null,
                fingerprint: first.fingerprint,
                detail: second.fingerprint,
                errorCount: first.errorCount
            });
        }
        return stripInternalFields(first);
    } catch (error) {
        const result = {
            kind: 'crash',
            error: { name: error.name, message: String(error.message || error) },
            fingerprint: '',
            detail: '',
            errorCount: typeof global.errorCount === 'number' ? global.errorCount : 0
        };
        if (job) {
            result.engineSeed = job.engineSeed;
        }
        return result;
    }
}

function readStdin() {
    return new Promise(function(resolve, reject) {
        let data = '';
        process.stdin.setEncoding('utf8');
        process.stdin.on('data', function(chunk) { data += chunk; });
        process.stdin.on('end', function() { resolve(data); });
        process.stdin.on('error', reject);
    });
}

readStdin().then(function(raw) {
    emit(runJob(JSON.parse(raw)));
}).catch(function(error) {
    emit({
        kind: 'crash',
        error: { name: error.name, message: String(error.message || error) },
        fingerprint: '',
        detail: '',
        errorCount: 0
    });
});
