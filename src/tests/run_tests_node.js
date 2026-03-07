#!/usr/bin/env node
'use strict';

const fs = require('fs');
const vm = require('vm');
const path = require('path');

const srcDir = path.join(__dirname, '..');

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
const failures = [];

const args = process.argv.slice(2);
const verbose = args.includes('--verbose') || args.includes('-v');
const filterArg = args.find(a => !a.startsWith('-'));

const _origLog = console.log;
if (!verbose) {
    console.log = function() {};
}

// Game simulation tests
const simStart = performance.now();
const simTotal = global.testdata.length;
_origLog(`Running ${simTotal} game simulation tests...`);

for (let i = 0; i < simTotal; i++) {
    const name = global.testdata[i][0];
    if (filterArg && !name.toLowerCase().includes(filterArg.toLowerCase())) continue;
    try {
        if (global.runTest(global.testdata[i][1], name)) {
            passed++;
        } else {
            failed++;
            failures.push(`FAIL: ${name}`);
        }
    } catch (e) {
        errored++;
        failures.push(`ERROR: ${name}: ${e.message}`);
    }
}

const simElapsed = ((performance.now() - simStart) / 1000).toFixed(2);
_origLog(`  Done in ${simElapsed}s`);

// Error message tests
const errStart = performance.now();
const errTotal = global.errormessage_testdata.length;
_origLog(`Running ${errTotal} error message tests...`);

for (let i = 0; i < errTotal; i++) {
    const name = global.errormessage_testdata[i][0];
    if (filterArg && !name.toLowerCase().includes(filterArg.toLowerCase())) continue;
    try {
        if (global.runCompilationTest(global.errormessage_testdata[i][1], name)) {
            passed++;
        } else {
            failed++;
            if (verbose) {
                const td = global.errormessage_testdata[i][1];
                _origLog(`  actual errorCount: ${global.errorCount}, expected: ${td[2]}`);
                const stripped = global.errorStrings.map(global.stripHTMLTags);
                _origLog(`  actual errors: ${JSON.stringify(stripped)}`);
                _origLog(`  expected errors: ${JSON.stringify(td[1])}`);
            }
            failures.push(`FAIL: [err] ${name}`);
        }
    } catch (e) {
        errored++;
        failures.push(`ERROR: [err] ${name}: ${e.message}`);
    }
}

const errElapsed = ((performance.now() - errStart) / 1000).toFixed(2);
_origLog(`  Done in ${errElapsed}s`);

// Report
console.log = _origLog;
const totalElapsed = ((performance.now() - simStart) / 1000).toFixed(2);
console.log(`\n--- Results ---`);
console.log(`Passed:  ${passed}`);
console.log(`Failed:  ${failed}`);
console.log(`Errors:  ${errored}`);
console.log(`Total:   ${passed + failed + errored} tests in ${totalElapsed}s`);
console.log(`  Simulation: ${simElapsed}s | Error messages: ${errElapsed}s`);

if (failures.length > 0) {
    console.log(`\nFailures:`);
    for (const f of failures) {
        console.log(`  ${f}`);
    }
}

process.exit(failed + errored > 0 ? 1 : 0);
