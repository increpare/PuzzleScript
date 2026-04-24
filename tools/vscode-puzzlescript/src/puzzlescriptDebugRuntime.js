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

function defaultRepoRoot() {
    return path.resolve(__dirname, '..', '..', '..');
}

function createBrowserShims(messageSink) {
    const storage = {};
    const sink = Array.isArray(messageSink) ? messageSink : [];
    const element = () => ({
        style: {},
        className: '',
        innerHTML: '',
        textContent: '',
        children: [],
        clientHeight: 0,
        clientWidth: 0,
        addEventListener() {},
        removeEventListener() {},
        appendChild(child) {
            this.children.push(child);
            return child;
        },
        remove() {},
        getContext() { return null; },
        cloneNode() { return element(); },
    });

    return {
        console,
        setTimeout,
        clearTimeout,
        setInterval() { return 0; },
        clearInterval() {},
        exports: undefined,
        module: undefined,
        require: undefined,
        localStorage: {
            getItem(key) { return Object.prototype.hasOwnProperty.call(storage, key) ? storage[key] : null; },
            setItem(key, value) { storage[key] = String(value); },
            removeItem(key) { delete storage[key]; },
        },
        document: {
            URL: 'vscode-puzzlescript-debug://',
            title: 'PuzzleScript Debug',
            body: {
                classList: { contains() { return false; } },
                addEventListener() {},
                removeEventListener() {},
            },
            createElement: element,
            createTextNode(text) { return { nodeType: 3, textContent: String(text) }; },
            getElementById() { return element(); },
        },
        Audio: function Audio() {
            return {
                src: '',
                play() { return Promise.resolve(); },
                cloneNode() { return new Audio(); },
            };
        },
        QUnit: {
            push() {},
            assert: { equal() {} },
        },
        UnitTestingThrow(error) {
            throw error;
        },
        consolePrint(message) {
            if (message) sink.push(String(message));
        },
        consoleError(message) {
            if (message) sink.push(String(message));
        },
        consolePrintFromRule() {},
        consoleCacheDump() {},
        canvasResize() {},
        redraw() {},
        forceRegenImages() {},
        killAudioButton() {},
        showAudioButton() {},
        regenSpriteImages() {},
        jumpToLine() {},
        printLevel() {},
        playSound(seed) {
            if (typeof this.pushSoundToHistory === 'function') {
                this.pushSoundToHistory(seed);
            }
        },
    };
}

function loadRuntimeCode(srcRoot) {
    let code = 'var exports=undefined; var module=undefined;\n';
    for (const file of SOURCE_FILES) {
        code += `\n// ---- ${file} ----\n`;
        code += fs.readFileSync(path.join(srcRoot, file), 'utf8');
        code += '\n';
    }
    code += `
var __psDebugTimeline = [];
var __psDebugRuleLines = Object.create(null);
var __psDebugCurrentInput = null;
var __psDebugMessages = [];

function __psDebugArray(value) {
    if (!value) return [];
    try { return Array.from(value); } catch (error) { return []; }
}

function __psDebugCloneCommands(value) {
    return Array.isArray(value) ? value.slice() : [];
}

function __psDebugSerializedLevel() {
    try {
        return typeof convertLevelToString === "function" ? convertLevelToString() : "";
    } catch (error) {
        return "";
    }
}

function __psDebugBuildRuleLines() {
    var lines = Object.create(null);
    function visit(groups) {
        if (!Array.isArray(groups)) return;
        for (var groupIndex = 0; groupIndex < groups.length; groupIndex++) {
            var group = groups[groupIndex];
            if (!Array.isArray(group)) continue;
            for (var ruleIndex = 0; ruleIndex < group.length; ruleIndex++) {
                var line = group[ruleIndex] && group[ruleIndex].lineNumber;
                if (typeof line === "number" && line > 0) {
                    lines[line] = true;
                }
            }
        }
    }
    if (typeof state !== "undefined" && state) {
        visit(state.rules);
        visit(state.lateRules);
    }
    __psDebugRuleLines = lines;
    return Object.keys(lines).map(function(line) { return Number(line); }).sort(function(a, b) { return a - b; });
}

function __psDebugSnapshotLabel(lineNumber) {
    if (__psDebugRuleLines[lineNumber]) return "Rule line " + lineNumber;
    if (lineNumber === -1) return "Turn start";
    if (lineNumber === -2) return "Rule phase";
    return "Runtime phase";
}

function addToDebugTimeline(levelSnapshot, lineNumber) {
    var sourceLine = __psDebugRuleLines[lineNumber] ? lineNumber : null;
    if (typeof debug_visualisation_array !== "undefined" && typeof debugger_turnIndex !== "undefined") {
        if (!Object.prototype.hasOwnProperty.call(debug_visualisation_array, debugger_turnIndex)) {
            debug_visualisation_array[debugger_turnIndex] = [];
        }
    }
    var snapshot = {
        index: __psDebugTimeline.length,
        input: __psDebugCurrentInput,
        label: __psDebugSnapshotLabel(lineNumber),
        lineNumber: lineNumber,
        sourceLine: sourceLine,
        width: levelSnapshot && levelSnapshot.width || 0,
        height: levelSnapshot && levelSnapshot.height || 0,
        layerCount: levelSnapshot && levelSnapshot.layerCount || 0,
        objects: levelSnapshot ? __psDebugArray(levelSnapshot.objects) : [],
        movements: levelSnapshot ? __psDebugArray(levelSnapshot.movements) : [],
        commandQueue: levelSnapshot ? __psDebugCloneCommands(levelSnapshot.commandQueue) : [],
        commandQueueSourceRules: levelSnapshot ? __psDebugCloneCommands(levelSnapshot.commandQueueSourceRules) : [],
        serializedLevel: __psDebugSerializedLevel(),
        currentLevelIndex: typeof curlevel === "number" ? curlevel : 0,
        titleScreen: Boolean(typeof titleScreen !== "undefined" && titleScreen),
        textMode: Boolean(typeof textMode !== "undefined" && textMode),
        winning: Boolean(typeof winning !== "undefined" && winning),
        againing: Boolean(typeof againing !== "undefined" && againing),
        soundHistory: Array.isArray(soundHistory) ? soundHistory.slice() : [],
    };
    if (typeof debug_visualisation_array !== "undefined" && typeof debugger_turnIndex !== "undefined") {
        debug_visualisation_array[debugger_turnIndex][lineNumber] = snapshot;
    }
    __psDebugTimeline.push(snapshot);
    return String(snapshot.index) + "," + String(lineNumber);
}

function __psDebugExecuteToken(token) {
    if (token === "undo") {
        DoUndo(false, true);
    } else if (token === "restart") {
        DoRestart();
    } else if (token === "tick") {
        processInput(-1);
    } else {
        var inputMap = { up: 0, left: 1, down: 2, right: 3, action: 4 };
        if (!Object.prototype.hasOwnProperty.call(inputMap, token)) {
            throw new Error("Unsupported PuzzleScript debug input: " + token);
        }
        processInput(inputMap[token]);
    }
}

function __psDebugCompile(source, options) {
    options = options || {};
    __psDebugMessages = [];
    __psDebugTimeline = [];
    unitTesting = false;
    IDE = true;
    lazyFunctionGeneration = false;
    cache_console_messages = false;
    suppress_all_console_output = false;
    var command = typeof options.level === "number" ? ["loadLevel", options.level] : ["loadFirstNonMessageLevel"];
    compile(command, String(source || "") + "\\n", options.seed === undefined ? null : options.seed);
    verbose_logging = true;
    var ruleLines = __psDebugBuildRuleLines();
    return {
        ruleLines: ruleLines,
        diagnostics: Array.isArray(errorStrings) ? errorStrings.slice() : [],
        errorCount: typeof errorCount === "number" ? errorCount : 0,
        current: {
            label: "Program start",
            sourceLine: 1,
            serializedLevel: __psDebugSerializedLevel(),
            currentLevelIndex: typeof curlevel === "number" ? curlevel : 0,
            titleScreen: Boolean(typeof titleScreen !== "undefined" && titleScreen),
            textMode: Boolean(typeof textMode !== "undefined" && textMode),
            winning: Boolean(typeof winning !== "undefined" && winning),
            againing: Boolean(typeof againing !== "undefined" && againing),
            commandQueue: level && Array.isArray(level.commandQueue) ? level.commandQueue.slice() : [],
            commandQueueSourceRules: level && Array.isArray(level.commandQueueSourceRules) ? level.commandQueueSourceRules.slice() : [],
            soundHistory: Array.isArray(soundHistory) ? soundHistory.slice() : [],
        },
    };
}

function __psDebugInput(token) {
    __psDebugTimeline = [];
    __psDebugCurrentInput = token;
    if (typeof debug_visualisation_array !== "undefined") {
        debug_visualisation_array = [];
    }
    if (typeof debugger_turnIndex !== "undefined") {
        debugger_turnIndex = 0;
    }
    verbose_logging = true;
    __psDebugExecuteToken(token);
    var againIndex = 0;
    while (againing) {
        againing = false;
        __psDebugCurrentInput = token + ":again:" + againIndex;
        processInput(-1);
        againIndex += 1;
    }
    __psDebugCurrentInput = null;
    if (__psDebugTimeline.length === 0) {
        addToDebugTimeline(level, -2);
    }
    return __psDebugTimeline.slice();
}

globalThis.__psDebugExports = {
    compile: __psDebugCompile,
    input: __psDebugInput,
    ruleLines: function() { return __psDebugBuildRuleLines(); },
};
`;
    return code;
}

class PuzzleScriptDebugRuntime {
    constructor(options = {}) {
        this.repoRoot = options.repoRoot || defaultRepoRoot();
        this.srcRoot = path.join(this.repoRoot, 'src');
        this.messages = [];
        const context = createBrowserShims(this.messages);
        context.window = context;
        context.global = context;
        context.globalThis = context;
        context.input = context.document.createElement('TEXTAREA');
        context.canvas = context.document.createElement('CANVAS');
        context.levelString = '';
        context.inputString = '';
        context.outputString = '';
        context.lastDownTarget = null;
        context.editor = { getValue() { return context.levelString; } };
        this.context = vm.createContext(context);
        vm.runInContext(loadRuntimeCode(this.srcRoot), this.context, { filename: 'puzzlescript_debug_runtime.js' });
        this.exports = this.context.__psDebugExports;
        this.ruleLines = [];
        this.current = null;
    }

    compile(source, options = {}) {
        this.messages.length = 0;
        const result = this.exports.compile(String(source || ''), options);
        this.ruleLines = result.ruleLines || [];
        this.current = result.current || null;
        return {
            ...result,
            messages: this.messages.slice(),
        };
    }

    validateBreakpoints(lines) {
        const valid = new Set(this.ruleLines);
        return lines.map(line => ({
            line,
            verified: valid.has(line),
            message: valid.has(line) ? undefined : 'No compiled PuzzleScript rule starts on this line.',
        }));
    }

    input(token) {
        const snapshots = this.exports.input(String(token || ''));
        this.current = snapshots.length > 0 ? snapshots[snapshots.length - 1] : this.current;
        return snapshots;
    }
}

module.exports = {
    PuzzleScriptDebugRuntime,
};
