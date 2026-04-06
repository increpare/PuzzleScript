#!/usr/bin/env node
'use strict';

const fs = require('fs');
const vm = require('vm');
const path = require('path');

const srcDir = __dirname;

// ---- Usage ----

const args = process.argv.slice(2);
if (args.length < 1 || args.includes('--help') || args.includes('-h')) {
    console.error('Usage: node compile_cli.js <input.ps> [output.html]');
    console.error('');
    console.error('Compiles a PuzzleScript source file into a standalone HTML file.');
    console.error('If output is omitted, uses the game title (or "game.html").');
    console.error('Errors and warnings are printed to stderr.');
    process.exit(args.includes('--help') || args.includes('-h') ? 0 : 1);
}

const inputFile = args[0];
const outputFileArg = args[1]; // may be undefined

if (!fs.existsSync(inputFile)) {
    console.error(`Error: input file not found: ${inputFile}`);
    process.exit(1);
}

const gameSource = fs.readFileSync(inputFile, 'utf8');

// ---- Browser shims (minimal, enough for compilation) ----

const _storage = {};
global.localStorage = {
    getItem(key) { return _storage.hasOwnProperty(key) ? _storage[key] : null; },
    setItem(key, value) { _storage[key] = String(value); },
    removeItem(key) { delete _storage[key]; }
};

global.document = {
    URL: 'compile://',
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

// Collect console output for errors/warnings
const messages = [];
global.consolePrint = function(str) {
    if (str) messages.push(str);
};
global.consoleError = function(str) {
    if (str) messages.push(str);
};

global.UnitTestingThrow = function(error) {
    throw error;
};

// ---- Load engine source files ----

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

vm.runInThisContext(allCode, { filename: 'combined_sources.js' });

// Override stripHTMLTags for Node
global.stripHTMLTags = function(html_str) {
    return html_str.replace(/<\/?[a-zA-Z][^>]*>/g, '').trim();
};

// ---- Compile the game ----

compile(["restart"], gameSource + "\n");

// Print errors/warnings to stderr (strip HTML tags)
for (const msg of messages) {
    const stripped = stripHTMLTags(msg);
    if (stripped && stripped !== '=================================') {
        console.error(stripped);
    }
}

if (errorCount > 0) {
    console.error(`\nCompilation finished with ${errorCount} error(s).`);
    // Still try to produce output if state exists
    if (state === null) {
        process.exit(1);
    }
}

// ---- Build standalone HTML ----

// Use the pre-inlined standalone template (same one the browser export uses)
const standaloneTemplatePath = path.join(srcDir, 'standalone_inlined.txt');
if (!fs.existsSync(standaloneTemplatePath)) {
    console.error('Error: standalone_inlined.txt not found at ' + standaloneTemplatePath);
    console.error('Run "node compile.js" first to generate it, or export from the browser.');
    process.exit(1);
}
let htmlString = fs.readFileSync(standaloneTemplatePath, 'utf8');

// Substitute metadata
let title = 'PuzzleScript Game';
if (state && state.metadata && state.metadata.title !== undefined) {
    title = state.metadata.title;
}

let homepage = 'https://www.puzzlescript.net';
if (state && state.metadata && state.metadata.homepage !== undefined) {
    homepage = state.metadata.homepage;
    if (!homepage.match(/^https?:\/\//)) {
        homepage = 'https://' + homepage;
    }
}
let homepage_stripped = homepage.replace(/^https?:\/\//, '');

function escapeHtmlChars(unsafe) {
    return unsafe
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

let background_color = 'black';
if (state && state.metadata && 'background_color' in state.metadata) {
    background_color = state.bgcolor;
}
htmlString = htmlString.replace(/___BGCOLOR___/g, background_color);

let text_color = 'lightblue';
if (state && state.metadata && 'text_color' in state.metadata) {
    text_color = state.fgcolor;
}
htmlString = htmlString.replace(/___TEXTCOLOR___/g, text_color);

htmlString = htmlString.replace(/__GAMETITLE__/g, escapeHtmlChars(title));
htmlString = htmlString.replace(/__HOMEPAGE__/g, homepage);
htmlString = htmlString.replace(/__HOMEPAGE_STRIPPED_PROTOCOL__/g, escapeHtmlChars(homepage_stripped));

// Encode the game source for embedding (same approach as buildStandalone.js)
// JSON.stringify gives us a valid JS string literal including quotes
let sourceCodeLiteral = JSON.stringify(gameSource);
// String.replace treats $ specially, so double them first
sourceCodeLiteral = sourceCodeLiteral.replace(/\$/g, '$$$$');
htmlString = htmlString.replace(/"__GAMEDAT__"/g, sourceCodeLiteral);

// Determine output filename
let outputFile = outputFileArg;
if (!outputFile) {
    const safeName = title.replace(/[^a-zA-Z0-9_\- ]/g, '').trim() || 'game';
    outputFile = safeName + '.html';
}

fs.writeFileSync(outputFile, htmlString, 'utf8');

if (errorCount > 0) {
    console.error(`Output written to ${outputFile} (with errors).`);
    process.exit(1);
} else {
    console.error(`Output written to ${outputFile}`);
    process.exit(0);
}
