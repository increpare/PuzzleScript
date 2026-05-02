#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

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

let runtime = null;
const CANONICAL_METADATA_KEYS = new Set([
    'noaction',
    'realtime_interval',
    'require_player_movement',
    'run_rules_on_level_start',
    'throttle_movement',
]);

function createRuntime() {
    const srcDir = __dirname;
    const context = {
        console,
        require,
        process,
        setTimeout,
        clearTimeout,
        setInterval,
        clearInterval,
        Buffer,
        performance,
    };

    const storage = {};
    context.localStorage = {
        getItem(key) { return Object.prototype.hasOwnProperty.call(storage, key) ? storage[key] : null; },
        setItem(key, value) { storage[key] = String(value); },
        removeItem(key) { delete storage[key]; }
    };

    context.document = {
        URL: 'canonicalize://',
        body: {
            classList: { contains() { return false; } },
            addEventListener() {},
            removeEventListener() {}
        },
        createElement() {
            return {
                style: {},
                innerHTML: '',
                textContent: '',
                getContext() { return null; }
            };
        },
        getElementById() { return null; }
    };

    context.window = context;
    context.global = context;
    context.globalThis = context;
    context.lastDownTarget = null;
    context.canvas = null;
    context.input = context.document.createElement('TEXTAREA');
    context.levelEditorOpened = false;
    context.canDump = false;
    context.compiledText = '';
    context.IDE = false;
    context.unitTesting = false;
    context.form1 = { code: { editorreference: { getValue() { return ''; } } } };

    context.canvasResize = function() {};
    context.redraw = function() {};
    context.forceRegenImages = function() {};
    context.consolePrintFromRule = function() {};
    context.consoleCacheDump = function() {};
    context.addToDebugTimeline = function() {};
    context.killAudioButton = function() {};
    context.showAudioButton = function() {};
    context.regenSpriteImages = function() {};
    context.jumpToLine = function() {};
    context.printLevel = function() {};
    context.playSound = function() {};

    context.Audio = function Audio() {
        return {
            src: '',
            play() { return Promise.resolve(); },
            cloneNode() { return new context.Audio(); }
        };
    };

    context.levelString = '';
    context.inputString = '';
    context.outputString = '';
    context.editor = { getValue() { return context.levelString; } };

    context.QUnit = {
        push() {},
        assert: { equal() {} }
    };

    context.consolePrint = function() {};
    context.consoleError = function() {};
    context.UnitTestingThrow = function(error) {
        throw error;
    };

    let allCode = '';
    for (const file of sourceFiles) {
        const code = fs.readFileSync(path.join(srcDir, file), 'utf8');
        allCode += `\n// ---- ${file} ----\n${code}\n`;
    }

    allCode += `
globalThis.__ps_exports = {
    parseSource: function(str) {
        resetParserErrorState();
        compiling = true;
        try {
            const processor = new codeMirrorFn();
            const state = processor.startState();
            const lines = str.split('\\n');
            for (let i = 0; i < lines.length; i++) {
                const ss = new CodeMirror.StringStream(lines[i], 4);
                do {
                    processor.token(ss, state);
                } while (ss.eol() === false);
            }
            return {
                state: processor.copyState(state),
                errorCount: errorCount,
                errorStrings: errorStrings.slice()
            };
        } finally {
            compiling = false;
        }
    },
    compileSemantic: function(str, includeWinConditions) {
        resetParserErrorState();
        compiling = true;
        try {
            const processor = new codeMirrorFn();
            const state = processor.startState();
            const lines = str.split('\\n');
            for (let i = 0; i < lines.length; i++) {
                const ss = new CodeMirror.StringStream(lines[i], 4);
                do {
                    processor.token(ss, state);
                } while (ss.eol() === false);
            }

            if (!isObjectDefined(state, "player")) {
                logErrorNoLine("Error, didn't find any object called player, either in the objects section, or the legends section. There must be a player!");
            }
            if (!isObjectDefined(state, "background")) {
                logErrorNoLine("Error, didn't find any object called background, either in the objects section, or the legends section. There must be a background!");
            }
            if (state.collisionLayers.length === 0) {
                logError("No collision layers defined.  All objects need to be in collision layers.");
                return {
                    state: null,
                    errorCount: errorCount,
                    errorStrings: errorStrings.slice()
                };
            }

            generateExtraMembers(state);
            generateMasks(state);
            state.backgroundMask = getMaskFromName(state, 'background');
            levelsToArray(state);
            rulesToArray(state);
            if (includeWinConditions) {
                processWinConditions(state);
            }
            checkObjectsAreLayered(state);

            return {
                state: state,
                errorCount: errorCount,
                errorStrings: errorStrings.slice()
            };
        } finally {
            compiling = false;
        }
    }
};
`;

    vm.createContext(context);
    vm.runInContext(allCode, context, { filename: 'canonicalize_runtime.js' });
    return context.__ps_exports;
}

function getRuntime() {
    if (runtime === null) {
        runtime = createRuntime();
    }
    return runtime;
}

function modeOptions(mode) {
    switch (mode) {
        case 'full':
            return {
                includeMetadata: true,
                includeVisuals: true,
                includeSounds: true,
                includeLevels: true,
                includeWinConditions: true,
                includeFlavorMetadata: true,
                includeMessageText: true,
                includeSynonyms: true,
            };
        case 'ruleset':
            return {
                includeMetadata: true,
                includeVisuals: false,
                includeSounds: false,
                includeLevels: false,
                includeWinConditions: false,
                includeFlavorMetadata: false,
                includeMessageText: false,
                includeSynonyms: false,
                collapseEquivalentObjects: true,
            };
        case 'semantic':
            return {
                includeMetadata: true,
                includeVisuals: false,
                includeSounds: false,
                includeLevels: true,
                includeWinConditions: true,
                includeFlavorMetadata: false,
                includeMessageText: false,
                includeSynonyms: false,
                collapseEquivalentObjects: true,
            };
        case 'family':
            return {
                includeMetadata: false,
                includeVisuals: false,
                includeSounds: false,
                includeLevels: false,
                includeWinConditions: false,
                includeFlavorMetadata: false,
                includeMessageText: false,
                includeSynonyms: false,
                collapseEquivalentObjects: true,
                canonicalFormat: 'puzzlescript-family-canonical-v1',
                objectNamePrefix: 'fam_',
            };
        case 'mechanics':
            return {
                includeMetadata: true,
                includeVisuals: false,
                includeSounds: false,
                includeLevels: false,
                includeWinConditions: true,
                includeFlavorMetadata: false,
                includeMessageText: false,
                includeSynonyms: false,
                collapseEquivalentObjects: true,
            };
        case 'no-levels':
            return {
                includeMetadata: true,
                includeVisuals: false,
                includeSounds: false,
                includeLevels: false,
                includeWinConditions: true,
                includeFlavorMetadata: false,
                includeMessageText: false,
                includeSynonyms: false,
            };
        case 'structural':
        default:
            return {
                includeMetadata: true,
                includeVisuals: false,
                includeSounds: false,
                includeLevels: true,
                includeWinConditions: true,
                includeFlavorMetadata: false,
                includeMessageText: false,
                includeSynonyms: false,
            };
    }
}

function buildNameData(state) {
    const map = new Map();
    const synonymTargets = new Map();

    const objectEntries = Object.entries(state.objects)
        .sort((a, b) => a[1].lineNumber - b[1].lineNumber || a[0].localeCompare(b[0]));
    objectEntries.forEach(([name], index) => {
        map.set(name, `obj_${index}`);
    });

    state.legend_synonyms
        .slice()
        .sort((a, b) => a.lineNumber - b.lineNumber || a[0].localeCompare(b[0]))
        .forEach((entry, index) => {
            map.set(entry[0], `syn_${index}`);
            synonymTargets.set(entry[0], entry[1]);
        });

    state.legend_aggregates
        .slice()
        .sort((a, b) => a.lineNumber - b.lineNumber || a[0].localeCompare(b[0]))
        .forEach((entry, index) => {
            map.set(entry[0], `agg_${index}`);
        });

    state.legend_properties
        .slice()
        .sort((a, b) => a.lineNumber - b.lineNumber || a[0].localeCompare(b[0]))
        .forEach((entry, index) => {
            map.set(entry[0], `prop_${index}`);
        });

    function resolveSynonym(name) {
        let current = name;
        const seen = new Set();
        while (synonymTargets.has(current) && !seen.has(current)) {
            seen.add(current);
            current = synonymTargets.get(current);
        }
        return current;
    }

    function canonicalName(name, resolveSynonyms) {
        const finalName = resolveSynonyms ? resolveSynonym(name) : name;
        if (!map.has(finalName)) {
            return finalName;
        }
        return map.get(finalName);
    }

    return { canonicalName };
}

function normalizeMetadataValue(key, value, options) {
    const flavorKeys = new Set(['title', 'author', 'homepage', 'youtube']);
    if (!options.includeFlavorMetadata && flavorKeys.has(key)) {
        return null;
    }
    if (!options.includeFlavorMetadata && !CANONICAL_METADATA_KEYS.has(key)) {
        return null;
    }
    return String(value).trim();
}

function canonicalizeRuleText(ruleText, canonicalName, options) {
    const tokens = [];
    let match;
    const tokenRegex = /->|\[|\]|\||\+|[^\s\[\]\|\+]+/gu;
    while ((match = tokenRegex.exec(ruleText)) !== null) {
        tokens.push(match[0]);
    }

    const result = [];
    for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i];
        const lower = token.toLowerCase();
        if (lower === 'message') {
            result.push('message');
            if (options.includeMessageText) {
                result.push(tokens.slice(i + 1).join(' ').trim());
            }
            break;
        }
        result.push(canonicalName(lower, true));
    }
    return result;
}

function canonicalizeSoundRow(soundRow, canonicalName) {
    return soundRow
        .slice(0, -1)
        .map(entry => canonicalName(String(entry[0]).trim().toLowerCase(), true));
}

function buildLevelGlyphMap(state, canonicalName) {
    const glyphMap = new Map();

    Object.keys(state.objects).forEach(name => {
        if (name.length === 1) {
            glyphMap.set(name, canonicalName(name, true));
        }
    });
    state.legend_synonyms.forEach(entry => {
        if (entry[0].length === 1) {
            glyphMap.set(entry[0], canonicalName(entry[0], true));
        }
    });
    state.legend_aggregates.forEach(entry => {
        if (entry[0].length === 1) {
            glyphMap.set(entry[0], canonicalName(entry[0], true));
        }
    });

    return glyphMap;
}

function canonicalizeLevels(levels, glyphMap, options) {
    if (!options.includeLevels) {
        return [];
    }

    return levels
        .filter(level => level.length > 0)
        .map(level => {
            if (level[0] === '\n') {
                return {
                    type: 'message',
                    text: options.includeMessageText ? String(level[1]).trim() : ''
                };
            }
            return {
                type: 'map',
                rows: level.slice(1).map(row =>
                    row.split('').map(ch => glyphMap.get(ch) || `glyph:${ch}`)
                )
            };
        });
}

function canonicalizeState(state, options) {
    const { canonicalName } = buildNameData(state);
    const glyphMap = buildLevelGlyphMap(state, canonicalName);

    const metadata = [];
    if (options.includeMetadata) {
        for (let i = 0; i < state.metadata.length; i += 2) {
            const key = state.metadata[i];
            const value = normalizeMetadataValue(key, state.metadata[i + 1], options);
            if (value !== null) {
                metadata.push({ key, value });
            }
        }
    }

    const objects = Object.entries(state.objects)
        .sort((a, b) => a[1].lineNumber - b[1].lineNumber || a[0].localeCompare(b[0]))
        .map(([name, objectData]) => {
            const canonicalObject = { name: canonicalName(name, false) };
            if (options.includeVisuals) {
                canonicalObject.colors = objectData.colors.slice();
                canonicalObject.sprite = objectData.spritematrix.slice();
            }
            return canonicalObject;
        });

    const legend = {
        synonyms: options.includeSynonyms ? state.legend_synonyms
            .slice()
            .sort((a, b) => a.lineNumber - b.lineNumber || a[0].localeCompare(b[0]))
            .map(entry => ({
                name: canonicalName(entry[0], false),
                target: canonicalName(entry[1], true)
            })) : [],
        aggregates: state.legend_aggregates
            .slice()
            .sort((a, b) => a.lineNumber - b.lineNumber || a[0].localeCompare(b[0]))
            .map(entry => ({
                name: canonicalName(entry[0], false),
                members: entry.slice(1).map(name => canonicalName(name, true))
            })),
        properties: state.legend_properties
            .slice()
            .sort((a, b) => a.lineNumber - b.lineNumber || a[0].localeCompare(b[0]))
            .map(entry => ({
                name: canonicalName(entry[0], false),
                members: entry.slice(1).map(name => canonicalName(name, true))
            })),
    };

    const collisionLayers = state.collisionLayers.map(layer =>
        layer.map(name => canonicalName(name, true))
    );

    const rules = state.rules.map(rule => canonicalizeRuleText(rule[0], canonicalName, options));

    const winConditions = options.includeWinConditions
        ? state.winconditions.map(condition =>
            condition.slice(0, -1).map(token => canonicalName(String(token).toLowerCase(), true))
        )
        : [];

    const sounds = options.includeSounds
        ? state.sounds.map(sound => canonicalizeSoundRow(sound, canonicalName))
        : [];

    const levels = canonicalizeLevels(state.levels, glyphMap, options);

    return {
        format: 'puzzlescript-canonical-v1',
        metadata,
        objects,
        legend,
        collisionLayers,
        rules,
        winConditions,
        sounds,
        levels,
    };
}

function buildSemanticObjectOrdering(state, options) {
    const uniqueObjectNames = Object.keys(state.objects);
    const playerSet = new Set(listObjectNamesFromMask(state.playerMask, state));
    const backgroundSet = new Set(listObjectNamesFromMask(state.backgroundMask, state));
    const firstSeen = new Map();
    let nextOrdinal = 0;

    function note(name) {
        if (!firstSeen.has(name)) {
            firstSeen.set(name, nextOrdinal++);
        }
    }

    function noteTerm(name) {
        if (state.objects[name]) {
            note(name);
            return;
        }
        const mask = state.objectMasks[name] || state.aggregateMasks[name];
        if (!mask) {
            note(name);
            return;
        }
        for (const concreteName of listObjectNamesFromMask(mask, state)) {
            note(concreteName);
        }
    }

    for (const rule of state.rules) {
        for (const side of [rule.lhs, rule.rhs]) {
            for (const row of side) {
                for (const cell of row) {
                    for (let i = 1; i < cell.length; i += 2) {
                        noteTerm(cell[i]);
                    }
                }
            }
        }
    }

    if (options.includeWinConditions) {
        for (const condition of state.winconditions) {
            for (const objName of uniqueObjectNames) {
                const objId = state.objects[objName].id;
                if (condition[1].get(objId) || condition[2].get(objId)) {
                    note(objName);
                }
            }
        }
    }

    if (options.includeLevels) {
        for (const level of state.levels) {
            if (level.message !== undefined) {
                continue;
            }
            for (let cellIndex = 0; cellIndex < level.n_tiles; cellIndex++) {
                const cell = level.getCell(cellIndex);
                for (const objName of uniqueObjectNames) {
                    const objId = state.objects[objName].id;
                    if (cell.get(objId)) {
                        note(objName);
                    }
                }
            }
        }
    }

    const orderedNames = uniqueObjectNames.slice().sort((a, b) => {
        const layerDelta = state.objects[a].layer - state.objects[b].layer;
        if (layerDelta !== 0) {
            return layerDelta;
        }

        const rolePriorityA = backgroundSet.has(a) ? 0 : (playerSet.has(a) ? 1 : 2);
        const rolePriorityB = backgroundSet.has(b) ? 0 : (playerSet.has(b) ? 1 : 2);
        if (rolePriorityA !== rolePriorityB) {
            return rolePriorityA - rolePriorityB;
        }

        const seenA = firstSeen.has(a) ? firstSeen.get(a) : Number.MAX_SAFE_INTEGER;
        const seenB = firstSeen.has(b) ? firstSeen.get(b) : Number.MAX_SAFE_INTEGER;
        if (seenA !== seenB) {
            return seenA - seenB;
        }

        return a.localeCompare(b);
    });

    return {
        orderedNames,
        playerSet,
        backgroundSet,
    };
}

function buildCompiledNameMap(state, options) {
    const { orderedNames, playerSet, backgroundSet } = buildSemanticObjectOrdering(state, options);
    const map = new Map();
    orderedNames.forEach((name, index) => {
        map.set(name, `obj_${index}`);
    });
    return {
        map,
        orderedNames,
        playerSet,
        backgroundSet,
    };
}

function mapTermToCanonicalSet(name, state, nameMap) {
    if (nameMap.has(name)) {
        return [nameMap.get(name)];
    }
    const mask = state.objectMasks[name] || state.aggregateMasks[name];
    if (!mask) {
        return [name];
    }
    return listObjectNamesFromMask(mask, state)
        .map(concreteName => nameMap.get(concreteName) || concreteName)
        .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function serializeCompiledCell(cell, nameMap, state) {
    if (cell.length === 2 && cell[0] === '...' && cell[1] === '...') {
        return { ellipsis: true };
    }

    const entries = [];
    for (let i = 0; i < cell.length; i += 2) {
        const mappedSet = mapTermToCanonicalSet(cell[i + 1], state, nameMap);
        const entry = { dir: cell[i] };
        if (mappedSet.length === 1) {
            entry.obj = mappedSet[0];
        } else {
            entry.objs = mappedSet;
        }
        entries.push(entry);
    }
    entries.sort((a, b) => {
        const nameA = a.obj || a.objs.join('|');
        const nameB = b.obj || b.objs.join('|');
        return nameA.localeCompare(nameB, undefined, { numeric: true }) || a.dir.localeCompare(b.dir);
    });
    return entries;
}

function extractEntryObjects(entry) {
    if (entry.obj) {
        return [entry.obj];
    }
    if (entry.objs) {
        return entry.objs.slice();
    }
    return [];
}

function relabelEntry(entry, mapper) {
    if (entry.obj) {
        return { dir: entry.dir, obj: mapper(entry.obj) };
    }
    if (entry.objs) {
        const mapped = Array.from(new Set(entry.objs.map(mapper))).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
        if (mapped.length === 1) {
            return { dir: entry.dir, obj: mapped[0] };
        }
        return { dir: entry.dir, objs: mapped };
    }
    return entry;
}

function normalizeRelabeledCell(cell, mapper) {
    if (cell.ellipsis) {
        return cell;
    }
    const relabeled = cell.map(entry => relabelEntry(entry, mapper));
    relabeled.sort((a, b) => {
        const left = a.obj || (a.objs || []).join('|');
        const right = b.obj || (b.objs || []).join('|');
        return left.localeCompare(right, undefined, { numeric: true }) || a.dir.localeCompare(b.dir);
    });
    const deduped = [];
    const seen = new Set();
    for (const entry of relabeled) {
        const key = JSON.stringify(entry);
        if (!seen.has(key)) {
            seen.add(key);
            deduped.push(entry);
        }
    }
    return deduped;
}

function collectRuleMentionedObjects(rules) {
    const mentioned = new Set();
    for (const rule of rules || []) {
        for (const side of [rule.lhs || [], rule.rhs || []]) {
            for (const row of side) {
                for (const cell of row) {
                    if (cell.ellipsis) {
                        continue;
                    }
                    for (const entry of cell) {
                        for (const objectName of extractEntryObjects(entry)) {
                            mentioned.add(objectName);
                        }
                    }
                }
            }
        }
    }
    return mentioned;
}

function collapseEquivalentObjectsInCanonical(canonical, options = {}) {
    const format = options.format || canonical.format;
    const namePrefix = options.namePrefix || 'obj_';
    const includeMetadata = options.includeMetadata !== false;
    const includeWinConditions = options.includeWinConditions !== false;
    const includeLevels = options.includeLevels !== false;
    const playerSet = new Set(canonical.playerObjects || []);
    const ruleMentioned = collectRuleMentionedObjects(canonical.rules || []);
    const retainedLayers = [];
    const retainedObjects = new Set();
    const inertBucketLabels = new Map();

    (canonical.collisionLayers || []).forEach((layer, layerIndex) => {
        const hasRetainedObject = layer.some(name => playerSet.has(name) || ruleMentioned.has(name));
        if (!hasRetainedObject) {
            return;
        }
        retainedLayers.push(layer.slice());
        layer.forEach(name => retainedObjects.add(name));
        const inertNonPlayers = layer.filter(name => !playerSet.has(name) && !ruleMentioned.has(name));
        if (inertNonPlayers.length > 0) {
            const bucketName = `__inert_layer_${layerIndex}`;
            inertNonPlayers.forEach(name => inertBucketLabels.set(name, bucketName));
        }
    });

    const objectToFamily = new Map();
    let nextFamilyIndex = 0;
    retainedLayers.forEach((layer, layerIndex) => {
        let inertFamilyName = null;
        layer.forEach(objectName => {
            if (inertBucketLabels.has(objectName)) {
                if (inertFamilyName === null) {
                    inertFamilyName = `${namePrefix}${nextFamilyIndex++}`;
                }
                objectToFamily.set(objectName, inertFamilyName);
                return;
            }
            objectToFamily.set(objectName, `${namePrefix}${nextFamilyIndex++}`);
        });
    });

    const mapper = name => objectToFamily.get(name) || name;
    const dedupeList = list => Array.from(new Set((list || []).map(mapper))).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

    const rules = (canonical.rules || []).map(rule => ({
        direction: rule.direction,
        late: !!rule.late,
        rigid: !!rule.rigid,
        randomRule: !!rule.randomRule,
        groupNumber: rule.groupNumber,
        lhs: (rule.lhs || []).map(row => row.map(cell => normalizeRelabeledCell(cell, mapper))),
        rhs: (rule.rhs || []).map(row => row.map(cell => normalizeRelabeledCell(cell, mapper))),
        commands: rule.commands || [],
    }));

    const dedupedRules = [];
    const seenRules = new Set();
    for (const rule of rules) {
        const key = JSON.stringify(rule);
        if (!seenRules.has(key)) {
            seenRules.add(key);
            dedupedRules.push(rule);
        }
    }

    return {
        format,
        metadata: includeMetadata ? (canonical.metadata || []) : [],
        playerObjects: dedupeList((canonical.playerObjects || []).filter(name => retainedObjects.has(name))),
        backgroundObjects: dedupeList((canonical.backgroundObjects || []).filter(name => retainedObjects.has(name))),
        collisionLayers: retainedLayers.map(layer => dedupeList(layer)),
        rules: dedupedRules,
        winConditions: includeWinConditions
            ? (canonical.winConditions || []).map(condition => ({
                quantifier: condition.quantifier,
                a: dedupeList(condition.a || []),
                b: dedupeList(condition.b || []),
            }))
            : [],
        levels: includeLevels
            ? (canonical.levels || []).map(level => {
                if (level.type !== 'map') {
                    return level;
                }
                return {
                    type: 'map',
                    rows: level.rows.map(row =>
                        row.map(cell => dedupeList(cell.filter(name => retainedObjects.has(name))))
                    ),
                };
            })
            : [],
    };
}

function serializeCompiledRule(rule, nameMap, state, options) {
    const commands = rule.commands
        .filter(command => !/^sfx(?:10|[0-9])$/.test(command[0]))
        .filter(command => command[0] !== 'checkpoint' && command[0] !== 'message')
        .map(command => {
            if (!options.includeMessageText || command[0] !== 'message') {
                return [command[0]];
            }
            return [command[0], String(command[1] || '').trim()];
        });

    return {
        direction: rule.direction,
        late: !!rule.late,
        rigid: !!rule.rigid,
        randomRule: !!rule.randomRule,
        groupNumber: rule.groupNumber,
        lhs: rule.lhs.map(row => row.map(cell => serializeCompiledCell(cell, nameMap, state))),
        rhs: rule.rhs.map(row => row.map(cell => serializeCompiledCell(cell, nameMap, state))),
        commands
    };
}

function listObjectsInCompiledCell(cell, state, nameMap) {
    const objects = [];
    for (const concreteName of Object.keys(state.objects)) {
        const objectId = state.objects[concreteName].id;
        if (cell.get(objectId)) {
            objects.push(nameMap.get(concreteName) || concreteName);
        }
    }
    objects.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    return objects;
}

function serializeCompiledLevels(levels, state, nameMap) {
    const result = [];
    for (const level of levels) {
        if (level.message !== undefined) {
            result.push({ type: 'message', text: '' });
            continue;
        }

        const rows = [];
        for (let y = 0; y < level.height; y++) {
            const row = [];
            for (let x = 0; x < level.width; x++) {
                const cellIndex = x * level.height + y;
                row.push(listObjectsInCompiledCell(level.getCell(cellIndex), state, nameMap));
            }
            rows.push(row);
        }
        result.push({ type: 'map', rows });
    }
    return result;
}

function serializeCompiledWinConditions(winConditions, state, nameMap) {
    return winConditions.map(condition => {
        const quantifier = condition[0];
        const maskA = condition[1];
        const maskB = condition[2];
        const objectsA = [];
        const objectsB = [];

        for (const objectName of Object.keys(state.objects)) {
            const objectId = state.objects[objectName].id;
            if (maskA.get(objectId)) {
                objectsA.push(objectName);
            }
            if (maskB.get(objectId)) {
                objectsB.push(objectName);
            }
        }

        return {
            quantifier,
            a: objectsA
                .map(name => nameMap.get(name) || name)
                .sort((a, b) => a.localeCompare(b, undefined, { numeric: true })),
            b: objectsB
                .map(name => nameMap.get(name) || name)
                .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
        };
    });
}

function listObjectsFromMask(mask, state) {
    const bitMask = Array.isArray(mask) ? mask[1] : mask;
    const objects = [];
    for (const objectName of Object.keys(state.objects)) {
        const objectId = state.objects[objectName].id;
        if (bitMask.get(objectId)) {
            objects.push(objectName);
        }
    }
    objects.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    return objects;
}

function listObjectNamesFromMask(mask, state) {
    const bitMask = Array.isArray(mask) ? mask[1] : mask;
    const objects = [];
    for (const objectName of Object.keys(state.objects)) {
        const objectId = state.objects[objectName].id;
        if (bitMask.get(objectId)) {
            objects.push(objectName);
        }
    }
    objects.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    return objects;
}

function canonicalizeCompiledState(state, options) {
    const { map: nameMap, playerSet, backgroundSet } = buildCompiledNameMap(state, options);
    const metadata = [];
    if (options.includeMetadata) {
        for (let i = 0; i < state.metadata.length; i += 2) {
            const key = state.metadata[i];
            const value = normalizeMetadataValue(key, state.metadata[i + 1], options);
            if (value !== null) {
                metadata.push({ key, value });
            }
        }
    }
    const collisionLayers = state.collisionLayers.map(layer =>
        Array.from(new Set(layer))
            .map(name => nameMap.get(name) || name)
            .sort((a, b) => a.localeCompare(b))
    );

    const rawRules = state.rules
        .map(rule => serializeCompiledRule(rule, nameMap, state, options))
        .filter(rule => rule.rhs.length > 0 || rule.commands.length > 0);
    const groupMap = new Map();
    let nextGroupNumber = 0;
    const rules = rawRules.map(rule => {
        if (!groupMap.has(rule.groupNumber)) {
            groupMap.set(rule.groupNumber, nextGroupNumber++);
        }
        return Object.assign({}, rule, {
            groupNumber: groupMap.get(rule.groupNumber)
        });
    });
    const result = {
        format: 'puzzlescript-semantic-canonical-v1',
        metadata,
        playerObjects: Array.from(playerSet).map(name => nameMap.get(name)).sort((a, b) => a.localeCompare(b)),
        backgroundObjects: Array.from(backgroundSet).map(name => nameMap.get(name)).sort((a, b) => a.localeCompare(b)),
        collisionLayers,
        rules,
    };

    if (options.includeWinConditions) {
        result.winConditions = serializeCompiledWinConditions(state.winconditions, state, nameMap);
    } else {
        result.winConditions = [];
    }

    if (options.includeLevels) {
        result.levels = serializeCompiledLevels(state.levels, state, nameMap);
    } else {
        result.levels = [];
    }

    if (options.collapseEquivalentObjects) {
        return collapseEquivalentObjectsInCanonical(result, {
            format: options.canonicalFormat || result.format,
            namePrefix: options.objectNamePrefix || 'obj_',
            includeMetadata: options.includeMetadata,
            includeWinConditions: options.includeWinConditions,
            includeLevels: options.includeLevels,
        });
    }

    return result;
}

function canonicalizeSource(source, mode = 'structural') {
    if (mode === 'ruleset' || mode === 'mechanics' || mode === 'semantic' || mode === 'family') {
        const options = modeOptions(mode);
        const compiled = getRuntime().compileSemantic(source, options.includeWinConditions);
        if (compiled.errorCount > 0 || compiled.state === null || compiled.state.invalid) {
            const message = compiled.errorStrings.join('\n');
            throw new Error(`Unable to canonicalize invalid PuzzleScript source.\n${message}`);
        }
        return canonicalizeCompiledState(compiled.state, options);
    }

    const parsed = getRuntime().parseSource(source);
    if (parsed.errorCount > 0) {
        const message = parsed.errorStrings.join('\n');
        throw new Error(`Unable to canonicalize invalid PuzzleScript source.\n${message}`);
    }

    const options = modeOptions(mode);
    return canonicalizeState(parsed.state, options);
}

function compileSemanticSource(source, options = {}) {
    const includeWinConditions = options.includeWinConditions !== false;
    const throwOnError = options.throwOnError !== false;
    const compiled = getRuntime().compileSemantic(source, includeWinConditions);
    if (throwOnError && (compiled.errorCount > 0 || compiled.state === null || compiled.state.invalid)) {
        const message = compiled.errorStrings.join('\n');
        throw new Error(`Unable to compile PuzzleScript source.\n${message}`);
    }
    return compiled;
}

function stableStringify(value) {
    return JSON.stringify(value, null, 2);
}

function hashCanonical(value) {
    return crypto.createHash('sha256').update(stableStringify(value)).digest('hex');
}

function canonicalizeFile(inputPath, mode = 'structural') {
    const source = fs.readFileSync(inputPath, 'utf8');
    return canonicalizeSource(source, mode);
}

function buildComparisonHashes(source) {
    const modes = ['full', 'structural', 'no-levels', 'mechanics', 'ruleset', 'semantic', 'family'];
    const result = {};
    for (const mode of modes) {
        result[mode] = hashCanonical(canonicalizeSource(source, mode));
    }
    return result;
}

module.exports = {
    buildComparisonHashes,
    canonicalizeFile,
    canonicalizeSource,
    compileSemanticSource,
    hashCanonical,
    stableStringify,
};
