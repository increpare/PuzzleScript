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
var __psDebugPauseConfig = null;
var __psDebugPausableSeen = 0;
var __psDebugTransaction = null;

function __psDebugArray(value) {
    if (!value) return [];
    try { return Array.from(value); } catch (error) { return []; }
}

function __psDebugCloneCommands(value) {
    return Array.isArray(value) ? value.slice() : [];
}

function __psDebugCloneJson(value) {
    if (value === undefined || value === null) return value;
    return JSON.parse(JSON.stringify(value));
}

function __psDebugCloneLevelBackup(value) {
    if (!value) return value;
    var clone = {};
    for (var key in value) {
        if (!Object.prototype.hasOwnProperty.call(value, key)) continue;
        if (key === "dat") {
            clone.dat = Array.from(value.dat || []);
        } else if (Array.isArray(value[key])) {
            clone[key] = value[key].slice();
        } else {
            clone[key] = value[key];
        }
    }
    return clone;
}

function __psDebugCloneRandomGen() {
    if (typeof RandomGen === "undefined" || !RandomGen) return null;
    var stateClone = null;
    if (RandomGen._state) {
        stateClone = {
            i: RandomGen._state.i,
            j: RandomGen._state.j,
            s: Array.isArray(RandomGen._state.s) ? RandomGen._state.s.slice() : [],
        };
    }
    return {
        seed: RandomGen.seed,
        normal: RandomGen._normal,
        state: stateClone,
    };
}

function __psDebugRestoreRandomGen(snapshot) {
    if (!snapshot) return;
    RandomGen = new RNG(snapshot.seed);
    RandomGen._normal = snapshot.normal;
    if (snapshot.state) {
        RandomGen._state = new RC4("");
        RandomGen._state.i = snapshot.state.i;
        RandomGen._state.j = snapshot.state.j;
        RandomGen._state.s = snapshot.state.s.slice();
    }
}

function __psDebugCaptureState() {
    return {
        level: typeof level4Serialization === "function" ? level4Serialization() : null,
        movements: level && level.movements ? Array.from(level.movements) : [],
        rigidMovementAppliedMask: level && Array.isArray(level.rigidMovementAppliedMask)
            ? level.rigidMovementAppliedMask.map(function(mask) { return mask && mask.data ? Array.from(mask.data) : []; })
            : [],
        rigidGroupIndexMask: level && Array.isArray(level.rigidGroupIndexMask)
            ? level.rigidGroupIndexMask.map(function(mask) { return mask && mask.data ? Array.from(mask.data) : []; })
            : [],
        commandQueue: level && Array.isArray(level.commandQueue) ? level.commandQueue.slice() : [],
        commandQueueSourceRules: level && Array.isArray(level.commandQueueSourceRules) ? level.commandQueueSourceRules.slice() : [],
        curlevel: typeof curlevel === "number" ? curlevel : 0,
        curlevelTarget: typeof curlevelTarget === "undefined" ? null : __psDebugCloneJson(curlevelTarget),
        titleScreen: Boolean(typeof titleScreen !== "undefined" && titleScreen),
        textMode: Boolean(typeof textMode !== "undefined" && textMode),
        titleMode: typeof titleMode === "number" ? titleMode : 0,
        titleSelection: typeof titleSelection === "number" ? titleSelection : 0,
        titleSelected: Boolean(typeof titleSelected !== "undefined" && titleSelected),
        messageselected: Boolean(typeof messageselected !== "undefined" && messageselected),
        winning: Boolean(typeof winning !== "undefined" && winning),
        againing: Boolean(typeof againing !== "undefined" && againing),
        timer: typeof timer === "number" ? timer : 0,
        autotick: typeof autotick === "number" ? autotick : 0,
        backups: Array.isArray(backups) ? backups.map(__psDebugCloneLevelBackup) : [],
        restartTarget: __psDebugCloneLevelBackup(restartTarget),
        inputHistory: Array.isArray(inputHistory) ? inputHistory.slice() : [],
        soundHistory: Array.isArray(soundHistory) ? soundHistory.slice() : [],
        random: __psDebugCloneRandomGen(),
    };
}

function __psDebugRestoreState(snapshot) {
    if (!snapshot) return;
    if (snapshot.level) {
        restoreLevel(snapshot.level);
    }
    if (level && level.movements && snapshot.movements) {
        level.movements = new Int32Array(snapshot.movements);
    }
    if (level && Array.isArray(level.rigidMovementAppliedMask) && snapshot.rigidMovementAppliedMask) {
        for (var rma = 0; rma < level.rigidMovementAppliedMask.length && rma < snapshot.rigidMovementAppliedMask.length; rma++) {
            level.rigidMovementAppliedMask[rma] = new BitVec(snapshot.rigidMovementAppliedMask[rma]);
        }
    }
    if (level && Array.isArray(level.rigidGroupIndexMask) && snapshot.rigidGroupIndexMask) {
        for (var rgi = 0; rgi < level.rigidGroupIndexMask.length && rgi < snapshot.rigidGroupIndexMask.length; rgi++) {
            level.rigidGroupIndexMask[rgi] = new BitVec(snapshot.rigidGroupIndexMask[rgi]);
        }
    }
    if (level) {
        level.commandQueue = snapshot.commandQueue ? snapshot.commandQueue.slice() : [];
        level.commandQueueSourceRules = snapshot.commandQueueSourceRules ? snapshot.commandQueueSourceRules.slice() : [];
    }
    curlevel = snapshot.curlevel;
    curlevelTarget = __psDebugCloneJson(snapshot.curlevelTarget);
    titleScreen = snapshot.titleScreen;
    textMode = snapshot.textMode;
    titleMode = snapshot.titleMode;
    titleSelection = snapshot.titleSelection;
    titleSelected = snapshot.titleSelected;
    messageselected = snapshot.messageselected;
    winning = snapshot.winning;
    againing = snapshot.againing;
    timer = snapshot.timer;
    autotick = snapshot.autotick;
    backups = snapshot.backups ? snapshot.backups.map(__psDebugCloneLevelBackup) : [];
    restartTarget = __psDebugCloneLevelBackup(snapshot.restartTarget);
    inputHistory = snapshot.inputHistory ? snapshot.inputHistory.slice() : [];
    soundHistory = snapshot.soundHistory ? snapshot.soundHistory.slice() : [];
    __psDebugRestoreRandomGen(snapshot.random);
}

function __psDebugSerializedLevel() {
    try {
        return typeof convertLevelToString === "function" ? convertLevelToString() : "";
    } catch (error) {
        return "";
    }
}

function __psDebugObjectInfos() {
    var infos = [];
    if (typeof state === "undefined" || !state || !Array.isArray(state.idDict)) {
        return infos;
    }
    for (var id = 0; id < state.idDict.length; id++) {
        var name = state.idDict[id];
        var object = state.objects && state.objects[name];
        if (!object) continue;
        infos.push({
            id: id,
            name: name,
            layer: typeof object.layer === "number" ? object.layer : -1,
            colors: Array.isArray(object.colors) ? object.colors.slice() : [],
            spriteMatrix: Array.isArray(object.spritematrix)
                ? object.spritematrix.map(function(row) {
                    return Array.isArray(row) ? row.slice() : String(row || "").split("").map(function(ch) { return Number(ch); });
                })
                : [],
            background: typeof state.backgroundid === "number" && id === state.backgroundid,
        });
    }
    return infos;
}

function __psDebugSnapshotFromLevel(levelSnapshot, lineNumber, label) {
    var sourceLine = __psDebugRuleLines[lineNumber] ? lineNumber : null;
    return {
        index: __psDebugTimeline.length,
        kind: label && /^Rule group/.test(label) ? "rule-group" : "snapshot",
        input: __psDebugCurrentInput,
        label: label || __psDebugSnapshotLabel(lineNumber),
        lineNumber: lineNumber,
        sourceLine: sourceLine,
        width: levelSnapshot && levelSnapshot.width || 0,
        height: levelSnapshot && levelSnapshot.height || 0,
        layerCount: levelSnapshot && levelSnapshot.layerCount || 0,
        strideObject: typeof STRIDE_OBJ === "number" ? STRIDE_OBJ : 0,
        strideMovement: typeof STRIDE_MOV === "number" ? STRIDE_MOV : 0,
        backgroundId: typeof state !== "undefined" && state ? state.backgroundid : null,
        objectInfos: __psDebugObjectInfos(),
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

function __psDebugRuleGroupLines(ruleGroup) {
    var lines = [];
    if (!Array.isArray(ruleGroup)) return lines;
    for (var index = 0; index < ruleGroup.length; index++) {
        var line = ruleGroup[index] && ruleGroup[index].lineNumber;
        if (typeof line === "number" && line > 0 && lines.indexOf(line) < 0) {
            lines.push(line);
        }
    }
    return lines;
}

function __psDebugBreakpointLineForGroup(ruleGroup) {
    var lines = __psDebugRuleGroupLines(ruleGroup);
    if (!__psDebugPauseConfig || !__psDebugPauseConfig.breakpoints) {
        return null;
    }
    for (var index = 0; index < lines.length; index++) {
        if (__psDebugPauseConfig.breakpoints[lines[index]]) {
            return lines[index];
        }
    }
    return null;
}

function __psDebugBeforeRuleGroup(ruleGroup) {
    if (!__psDebugPauseConfig || (!__psDebugPauseConfig.step && !__psDebugPauseConfig.breakpoints)) {
        return;
    }
    var groupLines = __psDebugRuleGroupLines(ruleGroup);
    if (groupLines.length === 0) {
        return;
    }
    var breakpointLine = __psDebugBreakpointLineForGroup(ruleGroup);
    if (!__psDebugPauseConfig.step && breakpointLine == null) {
        return;
    }
    var sourceLine = breakpointLine || groupLines[0];
    var snapshot = __psDebugSnapshotFromLevel(level, sourceLine, "Rule group line " + sourceLine);
    snapshot.kind = "rule-group";
    snapshot.ruleGroupLines = groupLines;
    __psDebugTimeline.push(snapshot);
    __psDebugMaybePause(snapshot);
}

var __psDebugOriginalApplyRuleGroup = applyRuleGroup;
applyRuleGroup = function(ruleGroup) {
    __psDebugBeforeRuleGroup(ruleGroup);
    return __psDebugOriginalApplyRuleGroup(ruleGroup);
};

function addToDebugTimeline(levelSnapshot, lineNumber) {
    if (typeof debug_visualisation_array !== "undefined" && typeof debugger_turnIndex !== "undefined") {
        if (!Object.prototype.hasOwnProperty.call(debug_visualisation_array, debugger_turnIndex)) {
            debug_visualisation_array[debugger_turnIndex] = [];
        }
    }
    var snapshot = __psDebugSnapshotFromLevel(levelSnapshot, lineNumber);
    if (typeof debug_visualisation_array !== "undefined" && typeof debugger_turnIndex !== "undefined") {
        debug_visualisation_array[debugger_turnIndex][lineNumber] = snapshot;
    }
    __psDebugTimeline.push(snapshot);
    __psDebugMaybePause(snapshot);
    return String(snapshot.index) + "," + String(lineNumber);
}

function __psDebugBreakpointMap(lines) {
    var map = Object.create(null);
    (lines || []).forEach(function(line) {
        var n = Number(line);
        if (Number.isFinite(n)) map[n] = true;
    });
    return map;
}

function __psDebugShouldPause(snapshot) {
    if (!__psDebugPauseConfig || !snapshot) return false;
    if (snapshot.kind !== "rule-group") {
        return false;
    }
    if (__psDebugPauseConfig.step) {
        return snapshot.sourceLine != null;
    }
    if (snapshot.sourceLine == null) {
        return false;
    }
    return Boolean(__psDebugPauseConfig.breakpoints && __psDebugPauseConfig.breakpoints[snapshot.sourceLine]);
}

function __psDebugMaybePause(snapshot) {
    if (!__psDebugShouldPause(snapshot)) {
        return;
    }
    if (__psDebugPausableSeen < (__psDebugPauseConfig.skip || 0)) {
        __psDebugPausableSeen += 1;
        return;
    }
    var pause = {
        __puzzlescriptDebugPause: true,
        snapshot: snapshot,
        snapshots: __psDebugTimeline.slice(),
        nextSkip: __psDebugPausableSeen + 1,
    };
    throw pause;
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
        current: __psDebugSnapshotFromLevel(level, 1, "Program start"),
    };
}

function __psDebugRunToken(token, options) {
    options = options || {};
    __psDebugTimeline = [];
    __psDebugCurrentInput = token;
    __psDebugPauseConfig = {
        breakpoints: __psDebugBreakpointMap(options.breakpoints || []),
        step: Boolean(options.step),
        skip: Number(options.skip || 0),
    };
    __psDebugPausableSeen = 0;
    if (typeof debug_visualisation_array !== "undefined") {
        debug_visualisation_array = [];
    }
    if (typeof debugger_turnIndex !== "undefined") {
        debugger_turnIndex = 0;
    }
    try {
        verbose_logging = true;
        __psDebugExecuteToken(token);
        var againIndex = 0;
        while (againing) {
            againing = false;
            __psDebugCurrentInput = token + ":again:" + againIndex;
            processInput(-1);
            againIndex += 1;
        }
    } catch (error) {
        if (error && error.__puzzlescriptDebugPause) {
            if (__psDebugTransaction) {
                __psDebugTransaction.nextSkip = error.nextSkip;
                __psDebugTransaction.currentSnapshotIndex = error.snapshot ? error.snapshot.index : 0;
            }
            return {
                paused: true,
                snapshot: error.snapshot,
                snapshots: error.snapshots,
                nextSkip: error.nextSkip,
            };
        }
        throw error;
    } finally {
        __psDebugCurrentInput = null;
        __psDebugPauseConfig = null;
    }
    if (__psDebugTimeline.length === 0) {
        addToDebugTimeline(level, -2);
    }
    __psDebugTransaction = null;
    return {
        paused: false,
        snapshot: __psDebugTimeline.length > 0 ? __psDebugTimeline[__psDebugTimeline.length - 1] : __psDebugSnapshotFromLevel(level, -2, "Input complete"),
        snapshots: __psDebugTimeline.slice(),
        nextSkip: 0,
    };
}

function __psDebugInput(token, options) {
    __psDebugTransaction = {
        token: token,
        start: __psDebugCaptureState(),
        nextSkip: 0,
        currentSnapshotIndex: 0,
    };
    return __psDebugRunToken(token, options || {});
}

function __psDebugResume(options) {
    options = options || {};
    if (!__psDebugTransaction) {
        return {
            paused: false,
            snapshot: __psDebugSnapshotFromLevel(level, -2, "Input complete"),
            snapshots: [],
            nextSkip: 0,
        };
    }
    __psDebugRestoreState(__psDebugTransaction.start);
    options.skip = __psDebugTransaction.nextSkip || 0;
    return __psDebugRunToken(__psDebugTransaction.token, options);
}

globalThis.__psDebugExports = {
    compile: __psDebugCompile,
    input: __psDebugInput,
    resume: __psDebugResume,
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

    runInput(token, options = {}) {
        const result = this.exports.input(String(token || ''), options);
        this.current = result.snapshot || this.current;
        return result;
    }

    resume(options = {}) {
        const result = this.exports.resume(options);
        this.current = result.snapshot || this.current;
        return result;
    }

    input(token) {
        const result = this.runInput(token);
        return result.snapshots || [];
    }
}

module.exports = {
    PuzzleScriptDebugRuntime,
};
