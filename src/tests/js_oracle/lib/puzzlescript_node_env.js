'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const SOURCE_FILES = [
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

function installBrowserShims(messageSink) {
    const sink = messageSink || [];

    const storage = {};
    global.localStorage = {
        getItem(key) { return Object.prototype.hasOwnProperty.call(storage, key) ? storage[key] : null; },
        setItem(key, value) { storage[key] = String(value); },
        removeItem(key) { delete storage[key]; },
    };

    global.document = {
        URL: 'native-export://',
        body: {
            classList: { contains() { return false; } },
            addEventListener() {},
            removeEventListener() {},
        },
        createElement() {
            return {
                style: {},
                innerHTML: '',
                textContent: '',
                getContext() { return null; },
            };
        },
        getElementById() { return null; },
    };

    global.window = global;
    global.lastDownTarget = null;
    global.canvas = null;
    global.input = global.document.createElement('TEXTAREA');
    global.levelString = '';
    global.inputString = '';
    global.outputString = '';
    global.editor = { getValue() { return global.levelString; } };

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
    global.playSound = function(seed) {
        if (typeof pushSoundToHistory === 'function') {
            pushSoundToHistory(seed);
        }
    };
    global.Audio = function Audio() {
        return {
            src: '',
            play() { return Promise.resolve(); },
            cloneNode() { return new global.Audio(); },
        };
    };

    global.consolePrint = function(message) {
        if (message) {
            sink.push(String(message));
        }
    };
    global.consoleError = function(message) {
        if (message) {
            sink.push(String(message));
        }
    };

    global.QUnit = {
        push() {},
        assert: { equal() {} },
    };
    global.UnitTestingThrow = function(error) {
        throw error;
    };

    return sink;
}

function loadPuzzleScript(options = {}) {
    const srcDir = path.join(__dirname, '..', '..', '..');
    const includeTests = Boolean(options.includeTests);
    const messageSink = installBrowserShims(options.messageSink);

    let allCode = 'var exports=undefined; var module=undefined;\n';
    for (const file of SOURCE_FILES) {
        allCode += `\n// ---- ${file} ----\n`;
        allCode += fs.readFileSync(path.join(srcDir, file), 'utf8');
        allCode += '\n';
    }

    if (includeTests) {
        const extraFiles = [
            'tests/resources/testingFrameWork.js',
            'tests/resources/testdata.js',
            'tests/resources/errormessage_testdata.js',
        ];
        for (const file of extraFiles) {
            allCode += `\n// ---- ${file} ----\n`;
            allCode += fs.readFileSync(path.join(srcDir, file), 'utf8');
            allCode += '\n';
        }
    }

    vm.runInThisContext(allCode, { filename: 'combined_sources.js' });

    global.stripHTMLTags = function(htmlStr) {
        return String(htmlStr).replace(/<\/?[a-zA-Z][^>]*>/g, '').trim();
    };

    return {
        srcDir,
        messageSink,
    };
}

module.exports = {
    SOURCE_FILES,
    loadPuzzleScript,
};
