#!/usr/bin/env node
'use strict';

const fs = require('fs');
const vm = require('vm');
const path = require('path');
const { spawnSync } = require('child_process');

const srcDir = path.join(__dirname, '..');

function readNumericOption(args, name, fallback) {
    const prefix = `${name}=`;
    const inline = args.find(a => a.startsWith(prefix));
    if (inline) {
        const parsed = Number(inline.slice(prefix.length));
        return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
    }
    const index = args.indexOf(name);
    if (index >= 0 && index + 1 < args.length) {
        const parsed = Number(args[index + 1]);
        return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
    }
    return fallback;
}

function withoutOption(args, name) {
    const prefix = `${name}=`;
    const result = [];
    for (let i = 0; i < args.length; i++) {
        if (args[i] === name) {
            i++;
            continue;
        }
        if (args[i].startsWith(prefix)) {
            continue;
        }
        result.push(args[i]);
    }
    return result;
}

// ---- Profile: run this script multiple times in separate processes (cold) ----
if (process.argv.includes('--profile')) {
    const rawArgs = process.argv.slice(2);
    const runs = readNumericOption(rawArgs, '--profile-runs', 5);
    const childArgs = withoutOption(rawArgs.filter(a => a !== '--profile'), '--profile-runs');
    const scriptPath = path.join(__dirname, 'run_tests_node.js');
    const times = [];
    const breakdowns = [];
    let anyFailed = false;
    for (let i = 0; i < runs; i++) {
        const result = spawnSync(process.execPath, [scriptPath, ...childArgs], {
            encoding: 'utf8',
            cwd: path.join(__dirname, '..')
        });
        const match = result.stdout && result.stdout.match(/Total:\s*\d+ tests in ([\d.]+)s/);
        if (match) times.push(parseFloat(match[1]) * 1000);
        else anyFailed = true;
        const breakdownMatch = result.stdout && result.stdout.match(/Breakdown: compile ([\d.]+)ms \((\d+) calls\), processInput ([\d.]+)ms \((\d+) calls\), undo ([\d.]+)ms \((\d+) calls\), restart ([\d.]+)ms \((\d+) calls\)/);
        if (breakdownMatch) {
            breakdowns.push({
                compileMs: parseFloat(breakdownMatch[1]),
                compileCount: Number(breakdownMatch[2]),
                processInputMs: parseFloat(breakdownMatch[3]),
                processInputCount: Number(breakdownMatch[4]),
                undoMs: parseFloat(breakdownMatch[5]),
                undoCount: Number(breakdownMatch[6]),
                restartMs: parseFloat(breakdownMatch[7]),
                restartCount: Number(breakdownMatch[8]),
            });
        }
        if (result.status !== 0) anyFailed = true;
    }
    if (times.length > 0) {
        const sorted = [...times].sort((a, b) => a - b);
        const avgMs = times.reduce((a, b) => a + b, 0) / times.length;
        const medianMs = sorted[Math.floor(sorted.length / 2)];
        console.log(`\n--- Profile (cold, ${runs} separate processes) ---`);
        console.log(`Runs: ${times.map(ms => (ms / 1000).toFixed(2)).join('s, ')}s`);
        console.log(`Average: ${(avgMs / 1000).toFixed(2)}s`);
        console.log(`Median:  ${(medianMs / 1000).toFixed(2)}s`);
        console.log(`Min/Max: ${(sorted[0] / 1000).toFixed(2)}s / ${(sorted[sorted.length - 1] / 1000).toFixed(2)}s`);
        if (breakdowns.length > 0) {
            const avg = key => breakdowns.reduce((sum, b) => sum + b[key], 0) / breakdowns.length;
            console.log(`Breakdown avg: compile ${avg('compileMs').toFixed(0)}ms (${breakdowns[0].compileCount} calls), processInput ${avg('processInputMs').toFixed(0)}ms (${breakdowns[0].processInputCount} calls), undo ${avg('undoMs').toFixed(0)}ms (${breakdowns[0].undoCount} calls), restart ${avg('restartMs').toFixed(0)}ms (${breakdowns[0].restartCount} calls)`);
        }
    }
    process.exit(anyFailed ? 1 : 0);
}

// ---- Browser shims ----

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
    createElement(tag) {
        return {
            style: {},
            innerHTML: '',
            textContent: '',
            getContext() { return null; }
        };
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
global.console_print_raw = global.console.log;
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
global.inputString = '';
global.outputString = '';
global.editor = { getValue() { return global.levelString; } };

global.QUnit = {
    push() {},
    assert: { equal() {} }
};

global.UnitTestingThrow = function(error) {
    throw error;
};

// ---- Load all source files as a single concatenated script ----
// This is necessary because many files use top-level let/const,
// which are script-scoped in vm.runInThisContext(). Concatenating
// ensures they share the same scope, matching browser <script> behavior.

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
    'js/colors.js',
    'js/engine.js',
    'js/parser.js',
    'js/compiler.js',
    'js/soundbar.js',
];

let allCode = '';
for (const file of sourceFiles) {
    const code = fs.readFileSync(path.join(srcDir, file), 'utf8');
    allCode += `\n// ---- ${file} ----\n${code}\n`;
}

// Load test framework and test data into the same script scope
const extras = [
    'tests/resources/testingFrameWork.js',
    'tests/resources/testdata.js',
    'tests/resources/errormessage_testdata.js',
];
for (const file of extras) {
    const code = fs.readFileSync(path.join(srcDir, file), 'utf8');
    allCode += `\n// ---- ${file} ----\n${code}\n`;
}

vm.runInThisContext(allCode, { filename: 'combined_sources.js' });

// Override stripHTMLTags -- the debug.js version needs a real DOM.
// Only strip actual HTML tags (starting with <letter or </letter), not
// literal < characters that appear in error message text.
global.stripHTMLTags = function(html_str) {
    return html_str.replace(/<\/?[a-zA-Z][^>]*>/g, '').trim();
};

// ---- Run tests ----

let passed = 0;
let failed = 0;
let errored = 0;
let failures = [];

const args = process.argv.slice(2);
const verbose = args.includes('--verbose') || args.includes('-v');
const simOnly = args.includes('--sim-only');
const compilationOnly = args.includes('--compilation-only');
const breakdown = args.includes('--breakdown');
const filterArg = args.find(a => !a.startsWith('-'));

if (simOnly && compilationOnly) {
    console.error('Use only one of --sim-only or --compilation-only');
    process.exit(1);
}

const _origLog = console.log;
if (!verbose) {
    console.log = function() {};
}

const timing = {
    compileMs: 0,
    processInputMs: 0,
    undoMs: 0,
    restartMs: 0,
    compileCount: 0,
    processInputCount: 0,
    undoCount: 0,
    restartCount: 0,
};

if (breakdown) {
    const compileOrig = global.compile;
    global.compile = function(...args) {
        const start = performance.now();
        try {
            return compileOrig.apply(this, args);
        } finally {
            timing.compileMs += performance.now() - start;
            timing.compileCount++;
        }
    };

    const processInputOrig = global.processInput;
    global.processInput = function(...args) {
        const start = performance.now();
        try {
            return processInputOrig.apply(this, args);
        } finally {
            timing.processInputMs += performance.now() - start;
            timing.processInputCount++;
        }
    };

    const doUndoOrig = global.DoUndo;
    global.DoUndo = function(...args) {
        const start = performance.now();
        try {
            return doUndoOrig.apply(this, args);
        } finally {
            timing.undoMs += performance.now() - start;
            timing.undoCount++;
        }
    };

    const doRestartOrig = global.DoRestart;
    global.DoRestart = function(...args) {
        const start = performance.now();
        try {
            return doRestartOrig.apply(this, args);
        } finally {
            timing.restartMs += performance.now() - start;
            timing.restartCount++;
        }
    };
}

// Run one full test pass; returns { passed, failed, errored, failures, simMs, errMs, totalMs, timing }
function runOnePass() {
    let p = 0, f = 0, e = 0;
    const fails = [];
    timing.compileMs = 0;
    timing.processInputMs = 0;
    timing.undoMs = 0;
    timing.restartMs = 0;
    timing.compileCount = 0;
    timing.processInputCount = 0;
    timing.undoCount = 0;
    timing.restartCount = 0;

    const simTotal = global.testdata.length;
    const passStart = performance.now();
    const simStart = performance.now();

    if (!compilationOnly) {
        for (let i = 0; i < simTotal; i++) {
            const name = global.testdata[i][0];
            if (filterArg && !name.toLowerCase().includes(filterArg.toLowerCase())) continue;
            try {
                if (global.runTest(global.testdata[i][1], name)) {
                    p++;
                } else {
                    f++;
                    fails.push(`FAIL: ${name}`);
                }
            } catch (err) {
                e++;
                fails.push(`ERROR: ${name}: ${err.message}`);
            }
        }
    }
    const simMs = compilationOnly ? 0 : performance.now() - simStart;

    let errMs = 0;
    if (!simOnly) {
        const errStart = performance.now();
        const errTotal = global.errormessage_testdata.length;
        for (let i = 0; i < errTotal; i++) {
            const name = global.errormessage_testdata[i][0];
            if (filterArg && !name.toLowerCase().includes(filterArg.toLowerCase())) continue;
            try {
                if (global.runCompilationTest(global.errormessage_testdata[i][1], name)) {
                    p++;
                } else {
                    f++;
                    if (verbose) {
                        const td = global.errormessage_testdata[i][1];
                        _origLog(`  actual errorCount: ${global.errorCount}, expected: ${td[2]}`);
                        const stripped = global.errorStrings.map(global.stripHTMLTags);
                        _origLog(`  actual errors: ${JSON.stringify(stripped)}`);
                        _origLog(`  expected errors: ${JSON.stringify(td[1])}`);
                    }
                    fails.push(`FAIL: [err] ${name}`);
                }
            } catch (err) {
                e++;
                fails.push(`ERROR: [err] ${name}: ${err.message}`);
            }
        }
        errMs = performance.now() - errStart;
    }
    const totalMs = performance.now() - passStart;
    return { passed: p, failed: f, errored: e, failures: fails, simMs, errMs, totalMs, timing: { ...timing } };
}

const simTotal = global.testdata.length;
const errTotal = global.errormessage_testdata.length;
if (compilationOnly) {
    _origLog(`Running ${errTotal} compilation error tests...`);
} else if (simOnly) {
    _origLog(`Running ${simTotal} game simulation tests...`);
} else {
    _origLog(`Running ${simTotal} game simulation tests and ${errTotal} compilation error tests...`);
}

const result = runOnePass();
passed = result.passed;
failed = result.failed;
errored = result.errored;
failures = result.failures;
const simElapsed = (result.simMs / 1000).toFixed(2);
const errElapsed = (result.errMs / 1000).toFixed(2);
const totalElapsed = (result.totalMs / 1000).toFixed(2);
_origLog(`  Done in ${totalElapsed}s`);

// Report
console.log = _origLog;
console.log(`\n--- Results ---`);
console.log(`Passed:  ${passed}`);
console.log(`Failed:  ${failed}`);
console.log(`Errors:  ${errored}`);
console.log(`Total:   ${passed + failed + errored} tests in ${totalElapsed}s`);
console.log(`  Simulation: ${simElapsed}s | Error messages: ${errElapsed}s`);
if (breakdown) {
    console.log(`  Breakdown: compile ${timing.compileMs.toFixed(0)}ms (${timing.compileCount} calls), processInput ${timing.processInputMs.toFixed(0)}ms (${timing.processInputCount} calls), undo ${timing.undoMs.toFixed(0)}ms (${timing.undoCount} calls), restart ${timing.restartMs.toFixed(0)}ms (${timing.restartCount} calls)`);
}

if (failures.length > 0) {
    console.log(`\nFailures:`);
    for (const f of failures) {
        console.log(`  ${f}`);
    }
}

process.exit(failed + errored > 0 ? 1 : 0);
