'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

function Random(seed) {
    this._state = seed >>> 0;
}

Random.prototype.next = function() {
    this._state = (this._state + 0x6D2B79F5) >>> 0;
    let t = this._state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

Random.prototype.integer = function(n) {
    if (!(n > 0)) {
        throw new Error('n must be positive');
    }
    return Math.floor(this.next() * n);
};

Random.prototype.pick = function(array) {
    return array[this.integer(array.length)];
};

function evaluateDataFile(filePath) {
    const context = {};
    vm.runInNewContext(fs.readFileSync(filePath, 'utf8'), context);
    return context;
}

function loadCorpus(resourceDir) {
    const testdata = evaluateDataFile(path.join(resourceDir, 'testdata.js')).testdata;
    const errors = evaluateDataFile(path.join(resourceDir, 'errormessage_testdata.js')).errormessage_testdata;
    const corpus = [];
    for (let i = 0; i < testdata.length; i++) {
        const payload = testdata[i][1];
        corpus.push({
            name: testdata[i][0],
            fixtureIndex: i,
            kind: 'simulation',
            source: payload[0],
            inputs: payload[1] || [],
            level: payload[3] !== undefined ? payload[3] : 0,
            randomSeed: payload[4] !== undefined ? payload[4] : null
        });
    }
    for (let i = 0; i < errors.length; i++) {
        const payload = errors[i][1];
        corpus.push({
            name: errors[i][0],
            fixtureIndex: i,
            kind: 'compiler-message',
            source: payload[0],
            inputs: [],
            level: 0,
            randomSeed: null
        });
    }
    return corpus;
}

const SECTION_NAMES = [
    'OBJECTS', 'LEGEND', 'SOUNDS', 'COLLISIONLAYERS', 'RULES', 'WINCONDITIONS', 'LEVELS'
];

function findSection(source, name) {
    const lines = source.split('\n');
    let start = -1;
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].trim().toUpperCase() === name) {
            start = i;
            break;
        }
    }
    if (start < 0) {
        return null;
    }
    let end = lines.length;
    for (let i = start + 1; i < lines.length; i++) {
        if (SECTION_NAMES.indexOf(lines[i].trim().toUpperCase()) >= 0) {
            end = i;
            break;
        }
    }
    return { name: name, start: start, end: end, lines: lines };
}

function replaceRange(source, start, end, insert) {
    return source.slice(0, start) + insert + source.slice(end);
}

function marksIn(text, regex) {
    const marks = [];
    const re = new RegExp(regex.source, regex.flags.indexOf('g') >= 0 ? regex.flags : regex.flags + 'g');
    let match;
    while ((match = re.exec(text))) {
        marks.push({ index: match.index, text: match[0] });
    }
    return marks;
}

function mutateSection(source, sectionName, fn) {
    const section = findSection(source, sectionName);
    if (!section) {
        return null;
    }
    const body = section.lines.slice(section.start, section.end).join('\n');
    const next = fn(body);
    if (!next) {
        return null;
    }
    const lines = section.lines.slice();
    const replacement = next.source.split('\n');
    lines.splice(section.start, section.end - section.start, ...replacement);
    return { source: lines.join('\n'), detail: next.detail };
}

function deleteRulePunctuation(source, rng) {
    return mutateSection(source, 'RULES', function(body) {
        const marks = marksIn(body, /->|[\[\]|]/);
        if (marks.length === 0) {
            return null;
        }
        const mark = marks[rng.integer(marks.length)];
        return {
            source: replaceRange(body, mark.index, mark.index + mark.text.length, ''),
            detail: 'deleted ' + mark.text
        };
    });
}

function duplicateRulePunctuation(source, rng) {
    return mutateSection(source, 'RULES', function(body) {
        const marks = marksIn(body, /->|[\[\]|]/);
        if (marks.length === 0) {
            return null;
        }
        const mark = marks[rng.integer(marks.length)];
        return {
            source: replaceRange(body, mark.index, mark.index + mark.text.length, mark.text + mark.text),
            detail: 'duplicated ' + mark.text
        };
    });
}

function swapLegendOperator(source, rng) {
    return mutateSection(source, 'LEGEND', function(body) {
        const marks = marksIn(body, /\s+(and|or)\s+/i);
        if (marks.length === 0) {
            return null;
        }
        const mark = marks[rng.integer(marks.length)];
        const swapped = /and/i.test(mark.text)
            ? mark.text.replace(/and/i, 'or')
            : mark.text.replace(/or/i, 'and');
        return {
            source: replaceRange(body, mark.index, mark.index + mark.text.length, swapped),
            detail: 'swapped ' + mark.text.trim()
        };
    });
}

function invalidViewport(source, rng) {
    const marks = marksIn(source, /\b(flickscreen|zoomscreen)\s+-?\d+x-?\d+/i);
    const replacement = rng.integer(2) === 0 ? '0x0' : '-1x3';
    if (marks.length > 0) {
        const mark = marks[rng.integer(marks.length)];
        const next = mark.text.replace(/-?\d+x-?\d+/, replacement);
        return {
            source: replaceRange(source, mark.index, mark.index + mark.text.length, next),
            detail: 'viewport ' + next
        };
    }
    return {
        source: 'flickscreen ' + replacement + '\n' + source,
        detail: 'injected flickscreen ' + replacement
    };
}

function duplicateRuleCommand(source, rng) {
    return mutateSection(source, 'RULES', function(body) {
        const marks = marksIn(body, /\b(again|cancel|restart|checkpoint|win)\b/i);
        if (marks.length === 0) {
            return null;
        }
        const mark = marks[rng.integer(marks.length)];
        return {
            source: replaceRange(body, mark.index, mark.index + mark.text.length, mark.text + ' ' + mark.text),
            detail: 'duplicated command ' + mark.text
        };
    });
}

function legendCycle(source) {
    return mutateSection(source, 'LEGEND', function(body) {
        return {
            source: body.replace(/\s*$/, '') + '\nGardenCycleA = GardenCycleB\nGardenCycleB = GardenCycleA\n',
            detail: 'added GardenCycleA/GardenCycleB synonym cycle'
        };
    });
}

function swapLineRanges(lines, first, second) {
    const lo = first.start < second.start ? first : second;
    const hi = first.start < second.start ? second : first;
    return lines.slice(0, lo.start)
        .concat(lines.slice(hi.start, hi.end))
        .concat(lines.slice(lo.end, hi.start))
        .concat(lines.slice(lo.start, lo.end))
        .concat(lines.slice(hi.end))
        .join('\n');
}

function swapSections(source, rng) {
    const lines = source.split('\n');
    const found = [];
    for (let i = 0; i < SECTION_NAMES.length; i++) {
        const section = findSection(source, SECTION_NAMES[i]);
        if (section) {
            found.push(section);
        }
    }
    const candidates = [];
    for (let i = 0; i < found.length; i++) {
        for (let j = i + 1; j < found.length; j++) {
            const swapped = swapLineRanges(lines, found[i], found[j]);
            if (swapped !== source) {
                candidates.push({
                    source: swapped,
                    detail: 'swapped ' + found[i].name + ' and ' + found[j].name
                });
            }
        }
    }
    if (candidates.length === 0) {
        return null;
    }
    return candidates[rng.integer(candidates.length)];
}

function oddWhitespace(source, rng) {
    const indexes = [];
    for (let i = 1; i < source.length - 1; i++) {
        if (source[i] === ' ' && /[A-Za-z0-9]/.test(source[i - 1]) && /[A-Za-z0-9]/.test(source[i + 1])) {
            indexes.push(i);
        }
    }
    if (indexes.length === 0) {
        return null;
    }
    const index = indexes[rng.integer(indexes.length)];
    const insert = rng.integer(2) === 0 ? '\t' : '\u00a0';
    return {
        source: replaceRange(source, index, index + 1, insert),
        detail: 'replaced a token-boundary space'
    };
}

function unterminatedComment(source, rng) {
    const lines = source.split('\n');
    const index = rng.integer(lines.length);
    lines[index] = '(' + lines[index];
    return {
        source: lines.join('\n'),
        detail: 'unterminated comment on line ' + (index + 1)
    };
}

const mutators = [
    { name: 'delete-rule-punctuation', apply: deleteRulePunctuation },
    { name: 'duplicate-rule-punctuation', apply: duplicateRulePunctuation },
    { name: 'swap-legend-operator', apply: swapLegendOperator },
    { name: 'invalid-viewport', apply: invalidViewport },
    { name: 'duplicate-rule-command', apply: duplicateRuleCommand },
    { name: 'legend-cycle', apply: legendCycle },
    { name: 'swap-sections', apply: swapSections },
    { name: 'odd-whitespace', apply: oddWhitespace },
    { name: 'unterminated-comment', apply: unterminatedComment }
];

function mutateFixture(fixture, rng, mutatorNames, options) {
    const allowed = mutators.filter(function(mutator) {
        return !mutatorNames || mutatorNames.indexOf(mutator.name) >= 0;
    });
    if (allowed.length === 0) {
        throw new Error('inapplicable mutation: no mutators selected');
    }
    const maxAttempts = options && options.maxAttempts ? options.maxAttempts : 8;
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const mutator = allowed[rng.integer(allowed.length)];
        const applied = mutator.apply(fixture.source, rng);
        if (applied) {
            return {
                mutator: mutator.name,
                fixtureName: fixture.name,
                fixtureIndex: fixture.fixtureIndex,
                kind: fixture.kind,
                source: applied.source,
                detail: applied.detail,
                inputs: fixture.inputs,
                level: fixture.level,
                randomSeed: fixture.randomSeed,
                attempt: attempt
            };
        }
    }
    throw new Error('inapplicable mutation after ' + maxAttempts + ' attempts');
}

function needValue(argv, i, name) {
    if (i + 1 >= argv.length) {
        throw new Error('Missing value for ' + name);
    }
    return argv[i + 1];
}

function needPositiveInt(argv, i, name) {
    const value = Number(needValue(argv, i, name));
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(name + ' must be a positive integer');
    }
    return value;
}

function parseArguments(argv, options) {
    const now = options && options.now ? options.now : Date.now;
    const result = {
        seed: typeof now === 'function' ? now() : now,
        count: 100,
        timeoutMs: 2000,
        shrink: true,
        replay: true,
        maxInputs: 8,
        shrinkBudget: 200,
        maxAttempts: 8,
        output: '.build/monster_garden',
        fixture: null,
        mutators: null,
        listMutators: false
    };
    const names = mutators.map(function(mutator) { return mutator.name; });
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        switch (arg) {
            case '--seed': {
                const value = Number(needValue(argv, i, 'seed'));
                if (!Number.isInteger(value) || value < 0) {
                    throw new Error('seed must be a non-negative integer');
                }
                result.seed = value;
                i++;
                break;
            }
            case '--count':
                result.count = needPositiveInt(argv, i, 'count');
                i++;
                break;
            case '--timeout-ms':
                result.timeoutMs = needPositiveInt(argv, i, 'timeout-ms');
                i++;
                break;
            case '--fixture':
                result.fixture = needValue(argv, i, 'fixture');
                i++;
                break;
            case '--mutator': {
                const selected = needValue(argv, i, 'mutator').split(',').map(function(name) {
                    return name.trim();
                }).filter(Boolean);
                for (let j = 0; j < selected.length; j++) {
                    if (names.indexOf(selected[j]) < 0) {
                        throw new Error('Unknown mutator: ' + selected[j]);
                    }
                }
                result.mutators = selected;
                i++;
                break;
            }
            case '--output':
                result.output = needValue(argv, i, 'output');
                i++;
                break;
            case '--no-shrink':
                result.shrink = false;
                break;
            case '--no-replay':
                result.replay = false;
                break;
            case '--max-inputs':
                result.maxInputs = needPositiveInt(argv, i, 'max-inputs');
                i++;
                break;
            case '--shrink-budget':
                result.shrinkBudget = needPositiveInt(argv, i, 'shrink-budget');
                i++;
                break;
            case '--max-attempts':
                result.maxAttempts = needPositiveInt(argv, i, 'max-attempts');
                i++;
                break;
            case '--list-mutators':
                result.listMutators = true;
                break;
            default:
                throw new Error('Unknown option: ' + arg);
        }
    }
    return result;
}

function failureSignature(result) {
    if (result.kind === 'crash' && result.error) {
        const message = String(result.error.message || '').split('\n')[0];
        return 'crash:' + result.error.name + ':' + message;
    }
    if (result.kind === 'invariant') {
        return 'invariant:' + String(result.detail || '');
    }
    return String(result.kind);
}

function checkLevelInvariants(level, strideObj, strideMov) {
    if (!level || typeof level !== 'object') {
        return 'level is missing';
    }
    if (!(level.width > 0) || !(level.height > 0)) {
        return 'level dimensions are invalid';
    }
    if (level.n_tiles !== level.width * level.height) {
        return 'n_tiles does not match width*height';
    }
    const expectedObjects = level.n_tiles * strideObj;
    if (!level.objects || level.objects.length !== expectedObjects) {
        return 'objects length is ' + (level.objects && level.objects.length) + ' expected ' + expectedObjects;
    }
    if (level.movements && level.movements.length !== level.n_tiles * strideMov) {
        return 'movements length is ' + level.movements.length + ' expected ' + (level.n_tiles * strideMov);
    }
    return null;
}

function isInteresting(result) {
    return result.kind === 'crash'
        || result.kind === 'timeout'
        || result.kind === 'invariant'
        || result.kind === 'nondeterministic'
        || result.kind === 'replay-divergence';
}

function shrinkSource(source, keep, budget) {
    let current = source.split('\n');
    let remaining = budget;
    let changed = true;
    while (changed && remaining > 0) {
        changed = false;
        let i = 0;
        while (i < current.length && remaining > 0) {
            const candidate = current.slice(0, i).concat(current.slice(i + 1));
            remaining--;
            if (keep(candidate.join('\n'))) {
                current = candidate;
                changed = true;
            } else {
                i++;
            }
        }
    }
    return { source: current.join('\n'), steps: budget - remaining };
}

function artifactDirName(signature, seed, index) {
    const safe = String(signature)
        .replace(/[^A-Za-z0-9._-]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .slice(0, 80);
    return (safe || 'monster') + '-s' + seed + '_' + String(index).padStart(4, '0');
}

function formatRegression(name, source) {
    return '[\n    ' + JSON.stringify(name) + ',\n    [' + JSON.stringify(source) + ', [], ""]\n],\n';
}

function writeArtifacts(outputDir, dirName, files) {
    const tmp = path.join(outputDir, dirName + '.tmp');
    const dest = path.join(outputDir, dirName);
    fs.rmSync(tmp, { recursive: true, force: true });
    fs.mkdirSync(tmp, { recursive: true });
    const names = Object.keys(files);
    for (let i = 0; i < names.length; i++) {
        fs.writeFileSync(path.join(tmp, names[i]), files[names[i]]);
    }
    fs.rmSync(dest, { recursive: true, force: true });
    fs.renameSync(tmp, dest);
    return dest;
}

module.exports = {
    Random: Random,
    loadCorpus: loadCorpus,
    mutators: mutators,
    mutateFixture: mutateFixture,
    parseArguments: parseArguments,
    failureSignature: failureSignature,
    checkLevelInvariants: checkLevelInvariants,
    isInteresting: isInteresting,
    shrinkSource: shrinkSource,
    artifactDirName: artifactDirName,
    formatRegression: formatRegression,
    writeArtifacts: writeArtifacts
};
