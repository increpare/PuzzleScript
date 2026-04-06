#!/usr/bin/env node
'use strict';

const fs = require('fs');

const QUANTIFIER_TEXT = {
    '-1': 'no',
    '0': 'some',
    '1': 'all',
};

const COLOR_NAMES = [
    'black', 'white', 'gray', 'darkgray', 'lightgray',
    'red', 'darkred', 'lightred', 'brown', 'orange',
    'yellow', 'green', 'darkgreen', 'lightgreen', 'blue',
    'darkblue', 'lightblue', 'purple', 'pink',
];

const GLYPH_POOL = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!$%&*+,-/:;<=>?@[]^_{|}~';
const EXTRA_GLYPHS = '¡¢£¤¥¦§¨©«¬®¯°±²³´µ¶·¸¹»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿĀĂĄĆĈĊČĎĐĒĔĖĘĚĜĞĠĢĤĦĨĪĬĮİĲĴĶĹĻĽĿŁŃŅŇŌŎŐŒŔŖŘŚŜŞŠŢŤŦŨŪŬŮŰŲŴŶŸŹŻŽƀƁƂƄƆƇƉƊƋƌƎƏƐƑƓƔƖƗƘƜƝƟƠƢƤƧƩƬƮƯƱƲƳƵƷƸƼǍǏǑǓǕǗǙǛǞǠǤǦǨǪǬǮǰǴǶǸǺǼǾȀȂȄȆȈȊȌȎ';
const RESERVED_GLYPHS = new Set([
    'up', 'down', 'left', 'right', 'late', 'rigid', 'random', 'no', 'v', '^', '<', '>', '[', ']', '|', '=', '+', 'and', 'or', 'message'
]);

const DENSITY_PATTERNS = [
    [
        '00000',
        '00000',
        '00000',
        '00000',
        '00000',
    ],
    [
        '00000',
        '0...0',
        '0...0',
        '0...0',
        '00000',
    ],
    [
        '..0..',
        '.000.',
        '00000',
        '.000.',
        '..0..',
    ],
    [
        '0...0',
        '.0.0.',
        '..0..',
        '.0.0.',
        '0...0',
    ],
    [
        '0....',
        '.0...',
        '..0..',
        '...0.',
        '....0',
    ],
];

function parseCanonicalFile(inputPath) {
    return JSON.parse(fs.readFileSync(inputPath, 'utf8'));
}

function canonicalObjectNames(canonical) {
    const names = new Set();
    const addIfConcrete = name => {
        if (/^obj_\d+$/.test(name)) {
            names.add(name);
        }
    };
    for (const layer of canonical.collisionLayers || []) {
        for (const name of layer) {
            addIfConcrete(name);
        }
    }
    for (const name of canonical.playerObjects || []) {
        addIfConcrete(name);
    }
    for (const name of canonical.backgroundObjects || []) {
        addIfConcrete(name);
    }
    for (const rule of canonical.rules || []) {
        for (const side of [rule.lhs || [], rule.rhs || []]) {
            for (const row of side) {
                for (const cell of row) {
                    if (cell.ellipsis) {
                        continue;
                    }
                    for (const entry of cell) {
                        if (entry.obj) {
                            addIfConcrete(entry.obj);
                        }
                        for (const name of entry.objs || []) {
                            addIfConcrete(name);
                        }
                    }
                }
            }
        }
    }
    for (const condition of canonical.winConditions || []) {
        for (const name of condition.a || []) {
            addIfConcrete(name);
        }
        for (const name of condition.b || []) {
            addIfConcrete(name);
        }
    }
    for (const level of canonical.levels || []) {
        if (level.type !== 'map') {
            continue;
        }
        for (const row of level.rows) {
            for (const cell of row) {
                for (const name of cell) {
                    addIfConcrete(name);
                }
            }
        }
    }
    return Array.from(names).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function buildLayerIndex(canonical) {
    const index = new Map();
    (canonical.collisionLayers || []).forEach((layer, layerIndex) => {
        layer.forEach(name => index.set(name, layerIndex));
    });
    return index;
}

function chooseColor(name, index) {
    if (/^obj_0$/.test(name)) {
        return 'black';
    }
    return COLOR_NAMES[index % COLOR_NAMES.length];
}

function choosePattern(layerIndex, nameIndex, layerCount) {
    const densityBand = Math.min(
        DENSITY_PATTERNS.length - 1,
        Math.floor((layerIndex / Math.max(1, layerCount)) * DENSITY_PATTERNS.length)
    );
    const template = DENSITY_PATTERNS[densityBand];
    const shifted = [];
    const offset = nameIndex % 5;
    for (let rowIndex = 0; rowIndex < template.length; rowIndex++) {
        const row = template[rowIndex];
        shifted.push(row.slice(offset) + row.slice(0, offset));
    }
    return shifted;
}

function emitObjectsSection(objectNames, layerIndex) {
    const layerCount = Math.max(1, ...objectNames.map(name => (layerIndex.get(name) || 0) + 1));
    const lines = ['========', 'OBJECTS', '========', ''];
    objectNames.forEach((name, index) => {
        lines.push(name);
        lines.push(chooseColor(name, index));
        const sprite = choosePattern(layerIndex.get(name) || 0, index, layerCount);
        lines.push(...sprite);
        lines.push('');
    });
    return lines;
}

function normalizeSet(names) {
    return Array.from(new Set(names)).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function collectAliasNeeds(canonical, objectNames) {
    const allObjectsKey = JSON.stringify(objectNames);
    const roleAliases = [];
    const propertySets = [];
    const cellSets = [];

    function addPropertySet(names) {
        const normalized = normalizeSet(names);
        if (normalized.length <= 1) {
            return;
        }
        if (JSON.stringify(normalized) === allObjectsKey) {
            return;
        }
        propertySets.push(normalized);
    }

    function addCellSet(names) {
        const normalized = normalizeSet(names);
        if (normalized.length <= 1) {
            return;
        }
        cellSets.push(normalized);
    }

    addPropertySet(canonical.backgroundObjects || []);
    addPropertySet(canonical.playerObjects || []);

    for (const rule of canonical.rules || []) {
        for (const side of [rule.lhs || [], rule.rhs || []]) {
            for (const row of side) {
                for (const cell of row) {
                    if (cell.ellipsis) {
                        continue;
                    }
                    for (const entry of cell) {
                        if (entry.objs) {
                            addPropertySet(entry.objs);
                        }
                    }
                }
            }
        }
    }

    for (const condition of canonical.winConditions || []) {
        addPropertySet(condition.a || []);
        addPropertySet(condition.b || []);
    }

    for (const level of canonical.levels || []) {
        if (level.type !== 'map') {
            continue;
        }
        for (const row of level.rows) {
            for (const cell of row) {
                addCellSet(cell);
            }
        }
    }

    return { propertySets, cellSets };
}

function buildAliasDefinitions(canonical, objectNames) {
    const allObjectsKey = JSON.stringify(objectNames);
    const lines = [];
    const propertyAliasBySet = new Map();
    const cellAliasBySet = new Map();
    let propertyAliasIndex = 0;
    let cellAliasIndex = 0;

    function propertyAliasForSet(names) {
        const normalized = normalizeSet(names);
        if (normalized.length === 0) {
            return null;
        }
        if (normalized.length === objectNames.length && JSON.stringify(normalized) === allObjectsKey) {
            return null;
        }
        if (normalized.length === 1) {
            return normalized[0];
        }

        const key = normalized.join('|');
        if (!propertyAliasBySet.has(key)) {
            const aliasName = `set_${propertyAliasIndex++}`;
            propertyAliasBySet.set(key, aliasName);
        }
        return propertyAliasBySet.get(key);
    }

    function cellAliasForSet(names) {
        const normalized = normalizeSet(names);
        if (normalized.length === 0) {
            return null;
        }
        if (normalized.length === 1) {
            return normalized[0];
        }
        const key = normalized.join('|');
        if (!cellAliasBySet.has(key)) {
            const aliasName = `cell_${cellAliasIndex++}`;
            cellAliasBySet.set(key, aliasName);
        }
        return cellAliasBySet.get(key);
    }

    const { propertySets, cellSets } = collectAliasNeeds(canonical, objectNames);
    propertySets.forEach(set => propertyAliasForSet(set));
    cellSets.forEach(set => cellAliasForSet(set));

    for (const [key, aliasName] of Array.from(propertyAliasBySet.entries()).sort((a, b) => a[1].localeCompare(b[1], undefined, { numeric: true }))) {
        lines.push(`${aliasName} = ${key.split('|').join(' or ')}`);
    }
    for (const [key, aliasName] of Array.from(cellAliasBySet.entries()).sort((a, b) => a[1].localeCompare(b[1], undefined, { numeric: true }))) {
        lines.push(`${aliasName} = ${key.split('|').join(' and ')}`);
    }

    const backgroundAlias = propertyAliasForSet(canonical.backgroundObjects || []);
    const playerAlias = propertyAliasForSet(canonical.playerObjects || []);
    if (backgroundAlias) {
        lines.push(`background = ${backgroundAlias}`);
    }
    if (playerAlias) {
        lines.push(`player = ${playerAlias}`);
    }

    return {
        lines,
        cellAliasForSet,
        winAliasForSet: propertyAliasForSet,
        ruleAliasForSet: propertyAliasForSet,
    };
}

function buildLevelGlyphs(canonical, cellAliasForSet, forbiddenNames) {
    const glyphMap = new Map();
    const usedGlyphs = new Set();
    const usedGlyphNames = new Set();
    const glyphPool = GLYPH_POOL + EXTRA_GLYPHS;

    function assign(setNames, preferred) {
        const normalized = Array.from(new Set(setNames)).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
        const key = normalized.join('|');
        if (glyphMap.has(key)) {
            return glyphMap.get(key);
        }
        let glyph = null;
        if (preferred && !usedGlyphNames.has(preferred.toLowerCase())) {
            glyph = preferred;
        } else {
            for (const candidate of glyphPool) {
                const lowered = candidate.toLowerCase();
                if (!usedGlyphNames.has(lowered) && !forbiddenNames.has(lowered) && !RESERVED_GLYPHS.has(lowered)) {
                    glyph = candidate;
                    break;
                }
            }
        }
        if (glyph === null) {
            throw new Error('Too many unique cell types for single-character level glyphs.');
        }
        usedGlyphs.add(glyph);
        usedGlyphNames.add(glyph.toLowerCase());
        glyphMap.set(key, glyph);
        return glyph;
    }

    const backgroundKey = JSON.stringify((canonical.backgroundObjects || []).slice().sort());
    const playerOnlyCandidates = (canonical.playerObjects || []).map(name =>
        JSON.stringify(Array.from(new Set([...(canonical.backgroundObjects || []), name])).sort())
    );

    const legendLines = [];
    for (const level of canonical.levels || []) {
        if (level.type !== 'map') {
            continue;
        }
        for (const row of level.rows) {
            for (const cell of row) {
                const sortedCell = Array.from(new Set(cell)).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
                const cellJson = JSON.stringify(sortedCell);
                const preferred = cellJson === backgroundKey ? '.' : (playerOnlyCandidates.includes(cellJson) ? 'P' : null);
                const glyph = assign(sortedCell, preferred);
                const alias = cellAliasForSet(sortedCell);
                legendLines.push(`${glyph} = ${alias || sortedCell[0]}`);
            }
        }
    }

    const dedupedLegendLines = Array.from(new Set(legendLines));
    return { glyphMap, legendLines: dedupedLegendLines };
}

function emitLegendSection(aliasLines, glyphLegendLines) {
    const lines = ['=======', 'LEGEND', '=======', ''];
    if (aliasLines.length > 0) {
        lines.push(...aliasLines, '');
    }
    if (glyphLegendLines.length > 0) {
        lines.push(...glyphLegendLines, '');
    }
    return lines;
}

function normalizeRuleCellEntries(cell) {
    const usedObjects = new Set();
    const result = [];
    for (const entry of cell) {
        if (entry.ellipsis) {
            result.push(entry);
            continue;
        }
        if (entry.alias) {
            const dedupeKey = `alias:${entry.alias}`;
            if (!usedObjects.has(dedupeKey)) {
                usedObjects.add(dedupeKey);
                result.push(entry);
            }
            continue;
        }
        if (entry.obj) {
            if (!usedObjects.has(entry.obj)) {
                usedObjects.add(entry.obj);
                result.push(entry);
            }
            continue;
        }
        if (entry.objs) {
            const remaining = entry.objs.filter(name => !usedObjects.has(name));
            remaining.forEach(name => usedObjects.add(name));
            if (remaining.length === 1) {
                result.push({ dir: entry.dir, obj: remaining[0] });
            } else if (remaining.length > 1) {
                result.push({ dir: entry.dir, objs: remaining });
            }
        }
    }
    return result;
}

function formatRuleCell(cell) {
    if (cell.ellipsis) {
        return '...';
    }
    if (cell.length === 0) {
        return '';
    }
    return cell.map(entry => {
        const name = entry.obj || entry.alias || (entry.objs && entry.objs[0]) || '';
        return entry.dir ? `${entry.dir} ${name}` : name;
    }).join(' ');
}

function formatRuleRow(row) {
    return `[ ${row.map(formatRuleCell).join(' | ')} ]`;
}

function emitRulesSection(canonical, ruleAliasForSet) {
    const lines = ['======', 'RULES', '======', ''];
    let previousGroupNumber = null;
    for (const rule of canonical.rules || []) {
        if ((!rule.rhs || rule.rhs.length === 0) && (!rule.commands || rule.commands.length === 0)) {
            continue;
        }
        const prefix = [];
        const continuingGroup = previousGroupNumber !== null && previousGroupNumber === rule.groupNumber;
        if (continuingGroup) {
            prefix.push('+');
        }
        if (rule.late) {
            prefix.push('late');
        }
        if (rule.rigid) {
            prefix.push('rigid');
        }
        if (rule.randomRule && !continuingGroup) {
            prefix.push('random');
        }
        prefix.push(rule.direction);

        const rewriteRow = row => row.map(cell => {
            if (cell.ellipsis || cell.length === 0) {
                return cell;
            }
            return normalizeRuleCellEntries(cell.map(entry => {
                if (entry.objs) {
                    return {
                        dir: entry.dir,
                        alias: ruleAliasForSet(entry.objs)
                    };
                }
                return entry;
            }));
        });

        const lhs = (rule.lhs || []).map(row => formatRuleRow(rewriteRow(row))).join(' ');
        const rhs = (rule.rhs || []).map(row => formatRuleRow(rewriteRow(row))).join(' ');
        const commands = (rule.commands || [])
            .map(command => command.length > 1 ? `${command[0]} ${command[1]}` : command[0])
            .join(' ');
        const base = `${prefix.join(' ')} ${lhs}`;
        if (rule.rhs && rule.rhs.length > 0) {
            lines.push(commands ? `${base} -> ${rhs} ${commands}` : `${base} -> ${rhs}`);
        } else if (commands) {
            lines.push(`${base} -> ${commands}`);
        }
        previousGroupNumber = rule.groupNumber;
    }
    lines.push('');
    return lines;
}

function emitWinConditionsSection(canonical, objectNames, winAliasForSet) {
    const lines = ['==============', 'WINCONDITIONS', '==============', ''];
    const allObjectsKey = JSON.stringify(objectNames);
    for (const condition of canonical.winConditions || []) {
        const quantifier = QUANTIFIER_TEXT[String(condition.quantifier)];
        const left = winAliasForSet(condition.a) || condition.a[0];
        const normalizedB = Array.from(new Set(condition.b || [])).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
        if (JSON.stringify(normalizedB) === allObjectsKey) {
            lines.push(`${quantifier} ${left}`);
        } else {
            const right = winAliasForSet(normalizedB) || normalizedB[0];
            lines.push(`${quantifier} ${left} on ${right}`);
        }
    }
    lines.push('');
    return lines;
}

function emitLevelsSection(canonical, glyphMap) {
    const lines = ['======', 'LEVELS', '======', ''];
    for (const level of canonical.levels || []) {
        if (level.type === 'message') {
            lines.push('message');
            lines.push('');
            continue;
        }
        for (const row of level.rows) {
            const encoded = row.map(cell => {
                const key = Array.from(new Set(cell)).sort((a, b) => a.localeCompare(b, undefined, { numeric: true })).join('|');
                const glyph = glyphMap.get(key);
                if (!glyph) {
                    throw new Error(`No legend glyph defined for cell ${key}.`);
                }
                return glyph;
            }).join('');
            lines.push(encoded);
        }
        lines.push('');
    }
    return lines;
}

function emitMetadata(canonical) {
    const lines = [];
    for (const entry of canonical.metadata || []) {
        if (entry.value === 'true') {
            lines.push(entry.key);
        } else {
            lines.push(`${entry.key} ${entry.value}`);
        }
    }
    if (lines.length > 0) {
        lines.push('');
    }
    return lines;
}

function decanonicalizeSemantic(canonical) {
    if (canonical.format !== 'puzzlescript-semantic-canonical-v1') {
        throw new Error(`Unsupported canonical format: ${canonical.format}`);
    }

    const objectNames = canonicalObjectNames(canonical);
    const layerIndex = buildLayerIndex(canonical);
    const { lines: aliasLines, cellAliasForSet, winAliasForSet, ruleAliasForSet } = buildAliasDefinitions(canonical, objectNames);

    const output = [];
    output.push(...emitMetadata(canonical));
    output.push(...emitObjectsSection(objectNames, layerIndex));
    const forbiddenGlyphNames = new Set(objectNames.map(name => name.toLowerCase()));
    forbiddenGlyphNames.add('background');
    forbiddenGlyphNames.add('player');
    forbiddenGlyphNames.add('cell');
    forbiddenGlyphNames.add('set');
    const { glyphMap, legendLines } = buildLevelGlyphs(canonical, cellAliasForSet, forbiddenGlyphNames);
    output.push(...emitLegendSection(aliasLines, legendLines));
    output.push('=======', 'SOUNDS', '=======', '');
    output.push('================', 'COLLISIONLAYERS', '================', '');
    const seenLayerObjects = new Set();
    for (const layer of canonical.collisionLayers || []) {
        const filteredLayer = normalizeSet(layer).filter(name => {
            if (seenLayerObjects.has(name)) {
                return false;
            }
            seenLayerObjects.add(name);
            return true;
        });
        if (filteredLayer.length > 0) {
            output.push(filteredLayer.join(', '));
        }
    }
    output.push('');
    output.push(...emitRulesSection(canonical, ruleAliasForSet));
    output.push(...emitWinConditionsSection(canonical, objectNames, winAliasForSet));
    output.push(...emitLevelsSection(canonical, glyphMap));
    return `${output.join('\n').replace(/\n{3,}/g, '\n\n').trim()}\n`;
}

function main() {
    const args = process.argv.slice(2);
    if (args.length < 1 || args.includes('--help') || args.includes('-h')) {
        console.error('Usage: node src/decanonicalize.js <input.json> [output.ps]');
        console.error('');
        console.error('Rehydrates semantic canonical PuzzleScript JSON into a valid PuzzleScript source file.');
        process.exit(args.includes('--help') || args.includes('-h') ? 0 : 1);
    }

    const inputFile = args[0];
    const outputFile = args[1];
    const canonical = parseCanonicalFile(inputFile);
    const source = decanonicalizeSemantic(canonical);

    if (outputFile) {
        fs.writeFileSync(outputFile, source, 'utf8');
    } else {
        process.stdout.write(source);
    }
}

if (require.main === module) {
    main();
}

module.exports = {
    decanonicalizeSemantic,
};
