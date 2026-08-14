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
    '    againing: { get: function() { return againing; }, set: function(v) { againing = v; }, configurable: true },',
    '    STRIDE_OBJ: { get: function() { return STRIDE_OBJ; }, set: function(v) { STRIDE_OBJ = v; }, configurable: true },',
    '    STRIDE_MOV: { get: function() { return STRIDE_MOV; }, set: function(v) { STRIDE_MOV = v; }, configurable: true }',
    '});'
].join('\n') + '\n';
vm.runInThisContext(allCode, { filename: 'monster_garden_worker.js' });

function emit(result) {
    process.stdout.write(JSON.stringify(result) + '\n');
}

function drainAgain() {
    while (global.againing) {
        global.againing = false;
        global.processInput(-1);
    }
}

function applyInputs(inputs) {
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
        drainAgain();
    }
}

function fingerprintAfter(errorCount) {
    return String(errorCount) + '\n' + global.convertLevelToString();
}

function runOnce(job) {
    global.unitTesting = true;
    global.lazyFunctionGeneration = false;
    global.IDE = false;
    if (typeof global.resetParserErrorState === 'function') {
        global.resetParserErrorState();
    } else {
        global.errorStrings = [];
        global.errorCount = 0;
    }
    global.compile(['loadLevel', job.level], job.source, job.randomSeed);
    const errorCount = global.errorCount;
    if (errorCount > 0) {
        return {
            kind: 'compiler-error',
            error: null,
            fingerprint: 'compiler-error:' + errorCount,
            detail: '',
            errorCount: errorCount
        };
    }
    const broken = garden.checkLevelInvariants(global.level, global.STRIDE_OBJ, global.STRIDE_MOV);
    if (broken) {
        return {
            kind: 'invariant',
            error: null,
            fingerprint: fingerprintAfter(errorCount),
            detail: broken,
            errorCount: errorCount
        };
    }
    const prefix = (job.inputs || []).slice(0, job.maxInputs);
    applyInputs(prefix);
    const afterExec = garden.checkLevelInvariants(global.level, global.STRIDE_OBJ, global.STRIDE_MOV);
    if (afterExec) {
        return {
            kind: 'invariant',
            error: null,
            fingerprint: fingerprintAfter(errorCount),
            detail: afterExec,
            errorCount: errorCount
        };
    }
    return {
        kind: 'ok',
        error: null,
        fingerprint: fingerprintAfter(errorCount),
        detail: '',
        errorCount: errorCount,
        prefixLength: prefix.length
    };
}

function runJob(job) {
    try {
        const first = runOnce(job);
        if (first.kind !== 'ok') {
            return first;
        }
        const prefix = (job.inputs || []).slice(0, job.maxInputs);
        if (job.replay && prefix.length > 0) {
            for (let i = 0; i < prefix.length; i++) {
                global.DoUndo(false, true);
            }
            applyInputs(prefix);
            const replayed = fingerprintAfter(global.errorCount);
            if (replayed !== first.fingerprint) {
                return {
                    kind: 'replay-divergence',
                    error: null,
                    fingerprint: first.fingerprint,
                    detail: replayed,
                    errorCount: first.errorCount
                };
            }
        }
        const second = runOnce(job);
        if (second.kind !== 'ok') {
            return second;
        }
        if (second.fingerprint !== first.fingerprint) {
            return {
                kind: 'nondeterministic',
                error: null,
                fingerprint: first.fingerprint,
                detail: second.fingerprint,
                errorCount: first.errorCount
            };
        }
        return first;
    } catch (error) {
        return {
            kind: 'crash',
            error: { name: error.name, message: String(error.message || error) },
            fingerprint: '',
            detail: '',
            errorCount: typeof global.errorCount === 'number' ? global.errorCount : 0
        };
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
