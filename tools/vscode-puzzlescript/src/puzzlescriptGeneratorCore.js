'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');

const SECTION_NAMES = new Set([
    'objects',
    'legend',
    'sounds',
    'collisionlayers',
    'rules',
    'winconditions',
    'levels',
]);

const DEFAULT_GENERATOR_OPTIONS = Object.freeze({
    seed: 1,
    timeMs: 5000,
    samples: '',
    jobs: 'auto',
    solverTimeoutMs: 250,
    solverStrategy: 'portfolio',
    topK: 8,
});

const DEFAULT_RULES = [
    'choose 1 [ no wall ] -> [ player ]',
    'choose 3 [ no wall no player no crate no target ] [ no wall no player no crate no target ] -> [ crate ] [ target ]',
    'choose 5 [ no wall no player no crate no target ] -> [ wall ]',
].join('\n');

const PRESETS = Object.freeze([
    {
        id: 'move-actors',
        label: 'Move actors',
        rules: [
            'choose 20 [ player | no wall no crate ] -> [ | player ]',
            'or [ crate | no wall no player no target ] -> [ | crate ]',
        ].join('\n'),
    },
    {
        id: 'walls',
        label: 'Add/remove walls',
        rules: [
            'choose 20 option 0.4 [ wall ] -> []',
            'or option 0.6 [ no wall no player no crate no target ] -> [ wall ]',
        ].join('\n'),
    },
    {
        id: 'pairs',
        label: 'Crate/target pair',
        rules: [
            'choose 1 [ crate ] [ target ] -> [] []',
            'choose 1 [ no wall no player no crate no target ] [ no wall no player no crate no target ] -> [ crate ] [ target ]',
        ].join('\n'),
    },
    {
        id: 'mixed',
        label: 'Mixed exploration',
        rules: [
            'choose 10 [ player | no wall no crate ] -> [ | player ]',
            'or [ crate | no wall no player no target ] -> [ | crate ]',
            'choose 8 option 0.4 [ wall ] -> []',
            'or option 0.6 [ no wall no player no crate no target ] -> [ wall ]',
            'choose 1 [ crate ] [ target ] -> [] []',
            'choose 1 [ no wall no player no crate no target ] [ no wall no player no crate no target ] -> [ crate ] [ target ]',
        ].join('\n'),
    },
]);

function stripLineComment(line) {
    const index = String(line).indexOf('(');
    return index >= 0 ? String(line).slice(0, index) : String(line);
}

function sectionNameForLine(line) {
    const trimmed = stripLineComment(line).trim().toLowerCase();
    if (/^=+$/.test(trimmed)) {
        return null;
    }
    return SECTION_NAMES.has(trimmed) ? trimmed : null;
}

function findPlayableLevels(source) {
    const lines = String(source || '').split('\n');
    const levels = [];
    let section = '';
    let levelIndex = -1;
    let current = null;

    function closeCurrent(endLine) {
        if (!current) {
            return;
        }
        current.endLine = endLine;
        current.text = current.rows.join('\n');
        levels.push(current);
        current = null;
    }

    for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
        const rawLine = lines[lineIndex];
        const nextSection = sectionNameForLine(rawLine);
        if (nextSection) {
            closeCurrent(lineIndex);
            section = nextSection;
            continue;
        }
        if (section !== 'levels') {
            continue;
        }

        const trimmed = stripLineComment(rawLine).trim();
        if (!trimmed) {
            closeCurrent(lineIndex);
            continue;
        }
        if (/^message\b/i.test(trimmed)) {
            closeCurrent(lineIndex);
            levelIndex += 1;
            continue;
        }
        if (!current) {
            levelIndex += 1;
            current = {
                level: levelIndex,
                startLine: lineIndex,
                endLine: lineIndex + 1,
                rows: [],
            };
        }
        current.rows.push(rawLine);
    }
    closeCurrent(lines.length);
    return levels;
}

function selectedLevelForLine(source, line) {
    const levels = findPlayableLevels(source);
    if (levels.length === 0) {
        return null;
    }
    const numericLine = Number.isInteger(line) ? line : -1;
    const containing = levels.find(level => numericLine >= level.startLine && numericLine < level.endLine);
    return containing || levels[0];
}

function generatorSidecarPath(sourcePath) {
    return `${sourcePath}.gen`;
}

function specFromParts(initRows, rules) {
    return [
        '(INIT LEVEL)',
        ...(initRows || []),
        '',
        '(GENERATION RULES)',
        String(rules || DEFAULT_RULES).trim(),
        '',
    ].join('\n');
}

function defaultSpecForLevel(level) {
    return specFromParts(level ? level.rows : [], DEFAULT_RULES);
}

function readSidecarOrDefault(sourcePath, level) {
    const sidecar = generatorSidecarPath(sourcePath);
    if (fs.existsSync(sidecar)) {
        return { path: sidecar, text: fs.readFileSync(sidecar, 'utf8'), existed: true };
    }
    return { path: sidecar, text: defaultSpecForLevel(level), existed: false };
}

function parseLegendGlyphs(source) {
    const lines = String(source || '').split('\n');
    const glyphs = new Map();
    let section = '';
    for (const line of lines) {
        const nextSection = sectionNameForLine(line);
        if (nextSection) {
            section = nextSection;
            continue;
        }
        if (section !== 'legend') {
            continue;
        }
        const uncommented = stripLineComment(line).trim();
        const match = uncommented.match(/^(.+?)\s*=\s*(.+)$/);
        if (!match) {
            continue;
        }
        const left = match[1].trim();
        if ([...left].length !== 1) {
            continue;
        }
        const right = match[2]
            .split(/\s+and\s+/i)
            .map(part => part.trim().toLowerCase())
            .filter(Boolean);
        if (right.length === 0 || /\s+or\s+/i.test(match[2])) {
            continue;
        }
        glyphs.set(left, right.sort().join('\u0000'));
    }
    return glyphs;
}

function candidateCellNames(cell) {
    return String(cell || '')
        .trim()
        .split(/\s+/)
        .map(name => name.trim().toLowerCase())
        .filter(Boolean)
        .sort();
}

function candidateToRows(candidate, source) {
    const glyphs = parseLegendGlyphs(source);
    const bySignature = new Map();
    for (const [glyph, signature] of glyphs.entries()) {
        if (!bySignature.has(signature)) {
            bySignature.set(signature, glyph);
        }
    }
    const backgroundGlyph = bySignature.get('background') || '.';
    return (candidate.cells || []).map(row => row.map(cell => {
        const names = candidateCellNames(cell);
        const exact = bySignature.get(names.join('\u0000'));
        if (exact) {
            return exact;
        }
        const withoutBackground = names.filter(name => name !== 'background');
        const compact = bySignature.get(withoutBackground.join('\u0000'));
        if (compact) {
            return compact;
        }
        return backgroundGlyph;
    }).join(''));
}

function replacementForLevel(source, targetLevel, candidate) {
    if (!targetLevel) {
        throw new Error('No playable level found to replace.');
    }
    return {
        startLine: targetLevel.startLine,
        endLine: targetLevel.endLine,
        text: candidateToRows(candidate, source).join('\n'),
    };
}

function insertionAfterLevel(source, targetLevel, candidate) {
    const lines = String(source || '').split('\n');
    const rows = candidateToRows(candidate, source).join('\n');
    if (targetLevel) {
        return {
            line: targetLevel.endLine,
            text: `\n\n${rows}`,
        };
    }
    const levelsLine = lines.findIndex(line => sectionNameForLine(line) === 'levels');
    if (levelsLine < 0) {
        throw new Error('No LEVELS section found.');
    }
    return {
        line: lines.length,
        text: `\n${rows}`,
    };
}

function resolveGeneratorPath(configuredPath, repoRoot) {
    const configured = String(configuredPath || '').trim();
    if (configured) {
        return {
            path: configured,
            exists: fs.existsSync(configured),
            source: 'setting',
        };
    }
    const candidate = path.join(repoRoot, 'build', 'native', process.platform === 'win32' ? 'puzzlescript_generator.exe' : 'puzzlescript_generator');
    return {
        path: candidate,
        exists: fs.existsSync(candidate),
        source: 'repo',
    };
}

function makeTempDir() {
    return fs.mkdtempSync(path.join(os.tmpdir(), 'puzzlescript-generator-'));
}

function normalizeRunOptions(options) {
    const merged = { ...DEFAULT_GENERATOR_OPTIONS, ...(options || {}) };
    return {
        seed: Number.parseInt(merged.seed, 10) || DEFAULT_GENERATOR_OPTIONS.seed,
        timeMs: Math.max(1, Number.parseInt(merged.timeMs, 10) || DEFAULT_GENERATOR_OPTIONS.timeMs),
        samples: String(merged.samples || '').trim(),
        jobs: String(merged.jobs || DEFAULT_GENERATOR_OPTIONS.jobs).trim() || DEFAULT_GENERATOR_OPTIONS.jobs,
        solverTimeoutMs: Math.max(1, Number.parseInt(merged.solverTimeoutMs, 10) || DEFAULT_GENERATOR_OPTIONS.solverTimeoutMs),
        solverStrategy: String(merged.solverStrategy || DEFAULT_GENERATOR_OPTIONS.solverStrategy),
        topK: Math.max(1, Number.parseInt(merged.topK, 10) || DEFAULT_GENERATOR_OPTIONS.topK),
    };
}

module.exports = {
    DEFAULT_GENERATOR_OPTIONS,
    DEFAULT_RULES,
    PRESETS,
    candidateToRows,
    defaultSpecForLevel,
    findPlayableLevels,
    generatorSidecarPath,
    insertionAfterLevel,
    makeTempDir,
    normalizeRunOptions,
    readSidecarOrDefault,
    replacementForLevel,
    resolveGeneratorPath,
    selectedLevelForLine,
    specFromParts,
};
