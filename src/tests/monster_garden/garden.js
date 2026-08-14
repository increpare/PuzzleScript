'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { spawn } = require('child_process');

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
            corpusIndex: corpus.length,
            fixtureIndex: i,
            kind: 'simulation',
            source: payload[0],
            inputs: payload[1] || [],
            level: payload[3] !== undefined ? payload[3] : 0,
            randomSeed: payload[4] !== undefined ? payload[4] : null,
            expectedOutput: payload[2] !== undefined ? payload[2] : null,
            expectedErrors: null,
            expectedErrorCount: null
        });
    }
    for (let i = 0; i < errors.length; i++) {
        const payload = errors[i][1];
        corpus.push({
            name: errors[i][0],
            corpusIndex: corpus.length,
            fixtureIndex: i,
            kind: 'compiler-message',
            source: payload[0],
            inputs: [],
            level: 0,
            randomSeed: null,
            expectedOutput: null,
            expectedErrors: payload[1],
            expectedErrorCount: payload[2]
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

function duplicateRuleLine(source, rng) {
    return mutateSection(source, 'RULES', function(body) {
        const lines = body.split('\n');
        const indexes = [];
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].indexOf('->') >= 0) {
                indexes.push(i);
            }
        }
        if (indexes.length === 0) {
            return null;
        }
        const index = indexes[rng.integer(indexes.length)];
        lines.splice(index + 1, 0, lines[index]);
        return {
            source: lines.join('\n'),
            detail: 'duplicated rule line ' + (index + 1)
        };
    });
}

function swapObjectColors(source, rng) {
    return mutateSection(source, 'OBJECTS', function(body) {
        const colorRe = /^\s*(black|white|gray|grey|red|green|blue|yellow|pink|orange|brown|purple)\s*$/i;
        const lines = body.split('\n');
        const indexes = [];
        let wantName = true;
        for (let i = 0; i < lines.length; i++) {
            const trimmed = lines[i].trim();
            if (trimmed === '') {
                wantName = true;
                continue;
            }
            if (/^=+$/.test(trimmed)) {
                continue;
            }
            if (wantName) {
                wantName = false;
                continue;
            }
            if (colorRe.test(lines[i])) {
                indexes.push(i);
            }
        }
        if (indexes.length < 2) {
            return null;
        }
        const first = rng.integer(indexes.length);
        let second = rng.integer(indexes.length - 1);
        if (second >= first) {
            second++;
        }
        const earlier = indexes[first] < indexes[second] ? indexes[first] : indexes[second];
        const later = indexes[first] < indexes[second] ? indexes[second] : indexes[first];
        if (lines[earlier].trim().toLowerCase() === lines[later].trim().toLowerCase()) {
            return null;
        }
        const swapped = lines.slice();
        const tmp = swapped[earlier];
        swapped[earlier] = swapped[later];
        swapped[later] = tmp;
        return {
            source: swapped.join('\n'),
            detail: 'swapped ' + lines[earlier].trim() + ' and ' + lines[later].trim()
        };
    });
}

function legendMapKeys(source) {
    const legend = findSection(source, 'LEGEND');
    if (!legend) {
        return [];
    }
    const keys = [];
    const body = legend.lines.slice(legend.start, legend.end);
    for (let i = 0; i < body.length; i++) {
        const match = body[i].match(/^([^\s=])\s*=/);
        if (match && keys.indexOf(match[1]) < 0) {
            keys.push(match[1]);
        }
    }
    return keys;
}

function nudgeLevelCell(source, rng) {
    const keys = legendMapKeys(source);
    if (keys.length < 2) {
        return null;
    }
    return mutateSection(source, 'LEVELS', function(body) {
        const lines = body.split('\n');
        const cells = [];
        for (let i = 0; i < lines.length; i++) {
            if (i === 0 || /^\s*LEVELS\s*$/i.test(lines[i])) {
                continue;
            }
            if (/^\s*=+\s*$/.test(lines[i]) || lines[i].trim() === '') {
                continue;
            }
            if (/^\s*message\b/i.test(lines[i])) {
                continue;
            }
            for (let j = 0; j < lines[i].length; j++) {
                if (keys.indexOf(lines[i][j]) >= 0) {
                    cells.push({ line: i, col: j, ch: lines[i][j] });
                }
            }
        }
        if (cells.length === 0) {
            return null;
        }
        const cell = cells[rng.integer(cells.length)];
        const others = keys.filter(function(key) { return key !== cell.ch; });
        const replacement = others[rng.integer(others.length)];
        const chars = lines[cell.line].split('');
        chars[cell.col] = replacement;
        lines[cell.line] = chars.join('');
        return {
            source: lines.join('\n'),
            detail: 'nudged ' + cell.ch + ' to ' + replacement
        };
    });
}

function flipWinQuantifier(source) {
    return mutateSection(source, 'WINCONDITIONS', function(body) {
        const allMatch = /\ball\b/i.exec(body);
        const someMatch = /\bsome\b/i.exec(body);
        if (!allMatch && !someMatch) {
            return null;
        }
        let from;
        let to;
        let index;
        if (allMatch && (!someMatch || allMatch.index <= someMatch.index)) {
            from = allMatch[0];
            to = 'some';
            index = allMatch.index;
        } else {
            from = someMatch[0];
            to = 'all';
            index = someMatch.index;
        }
        return {
            source: replaceRange(body, index, index + from.length, to),
            detail: 'flipped ' + from + ' to ' + to
        };
    });
}

function arrowRuleIndexes(body) {
    const lines = body.split('\n');
    const indexes = [];
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].indexOf('->') >= 0) {
            indexes.push(i);
        }
    }
    return { lines: lines, indexes: indexes };
}

function mutateArrowRule(source, rng, fn) {
    return mutateSection(source, 'RULES', function(body) {
        const found = arrowRuleIndexes(body);
        if (found.indexes.length === 0) {
            return null;
        }
        const index = found.indexes[rng.integer(found.indexes.length)];
        const next = fn(found.lines[index], rng);
        if (!next) {
            return null;
        }
        found.lines[index] = next.line;
        return { source: found.lines.join('\n'), detail: next.detail };
    });
}

function insertRuleLine(source, line) {
    return mutateSection(source, 'RULES', function(body) {
        const found = arrowRuleIndexes(body);
        const lines = body.split('\n');
        const at = found.indexes.length > 0 ? found.indexes[0] : lines.length;
        lines.splice(at, 0, line);
        return { source: lines.join('\n'), detail: 'inserted ' + line };
    });
}

const NAUGHTY_STRINGS = [
    '\u200b',
    '\ufeff',
    '\u202e',
    '\u0301Player',
    '\u0410',
    '\u{1D400}',
    'NaN',
    'Infinity',
    'null',
    'undefined',
    '__proto__',
    '%s',
    '{0}',
    '........',
    '\uFF11',
    '\u000b',
    '\u2028',
    'constructor'
];

const KEYWORD_NAMES = ['^', 'v', 'late', 'no', 'and', 'or', 'random', 'moving', 'win', '...'];
const PRELUDE_FLAGS = [
    'noundo',
    'norestart',
    'noaction',
    'scanline',
    'throttle_movement',
    'run_rules_on_level_start',
    'realtime_interval 0',
    'again_interval -1',
    'key_repeat_interval 0',
    'color_palette not-a-palette'
];
const NUDGE_INPUT_CHOICES = [0, 1, 2, 3, 4, 'tick', 'undo', 'restart'];
const POISON_SEEDS = ['', '0', 'NaN', 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'];

function blnsSlot(source, rng) {
    const naughty = NAUGHTY_STRINGS[rng.integer(NAUGHTY_STRINGS.length)];
    const title = /^\s*title\s+.+$/im.exec(source);
    if (title) {
        return {
            source: replaceRange(source, title.index, title.index + title[0].length, title[0] + naughty),
            detail: 'appended naughty string to title'
        };
    }
    return {
        source: 'title ' + naughty + '\n' + source,
        detail: 'injected naughty title'
    };
}

function keywordAsName(source, rng) {
    const name = KEYWORD_NAMES[rng.integer(KEYWORD_NAMES.length)];
    return mutateSection(source, 'OBJECTS', function(body) {
        return {
            source: body.replace(/\s*$/, '') + '\n' + name + '\nred\n',
            detail: 'added object named ' + name
        };
    });
}

function orphanLegendMember(source) {
    return mutateSection(source, 'LEGEND', function(body) {
        return {
            source: body.replace(/\s*$/, '') + '\nGardenGhost = MissingName or Player\n',
            detail: 'legend member MissingName does not exist'
        };
    });
}

function injectEllipsis(source, rng) {
    return mutateArrowRule(source, rng, function(line) {
        const pipe = line.indexOf('|');
        if (pipe < 0) {
            const arrow = line.indexOf('->');
            if (arrow < 0) {
                return null;
            }
            return {
                line: line.slice(0, arrow) + '| ... ' + line.slice(arrow),
                detail: 'injected ellipsis before arrow'
            };
        }
        return {
            line: line.slice(0, pipe + 1) + ' ... ' + line.slice(pipe + 1),
            detail: 'injected ellipsis after |'
        };
    });
}

function injectNo(source, rng) {
    return mutateArrowRule(source, rng, function(line) {
        const bracket = line.indexOf('[');
        if (bracket < 0) {
            return null;
        }
        return {
            line: line.slice(0, bracket + 1) + ' no Player ' + line.slice(bracket + 1),
            detail: 'injected no Player'
        };
    });
}

function injectControlChar(source, rng) {
    const ch = rng.integer(2) === 0 ? '\u000b' : '\u2028';
    const marks = marksIn(source, /[A-Za-z]{3,}/);
    if (marks.length === 0) {
        return {
            source: ch + source,
            detail: 'prefixed a control character'
        };
    }
    const mark = marks[rng.integer(marks.length)];
    const at = mark.index + Math.floor(mark.text.length / 2);
    return {
        source: replaceRange(source, at, at, ch),
        detail: 'inserted a control character inside ' + mark.text
    };
}

function spriteMatrixNoise(source) {
    return mutateSection(source, 'OBJECTS', function(body) {
        const lines = body.split('\n');
        let wantName = true;
        let colorLine = -1;
        for (let i = 0; i < lines.length; i++) {
            const trimmed = lines[i].trim();
            if (trimmed === '') {
                wantName = true;
                continue;
            }
            if (wantName) {
                wantName = false;
                colorLine = -1;
                continue;
            }
            if (colorLine < 0 && /[A-Za-z#]/.test(trimmed)) {
                colorLine = i;
                break;
            }
        }
        if (colorLine < 0) {
            return null;
        }
        lines.splice(colorLine + 1, 0, '0123');
        return { source: lines.join('\n'), detail: 'inserted a 4-wide sprite row' };
    });
}

function duplicateObjectName(source) {
    return mutateSection(source, 'OBJECTS', function(body) {
        const lines = body.split('\n');
        let blockStart = -1;
        let wantName = true;
        for (let i = 0; i < lines.length; i++) {
            const trimmed = lines[i].trim();
            if (trimmed === '' || /^\s*=+\s*$/.test(trimmed) || trimmed.toUpperCase() === 'OBJECTS') {
                wantName = true;
                continue;
            }
            if (wantName) {
                blockStart = i;
                break;
            }
        }
        if (blockStart < 0) {
            return null;
        }
        let blockEnd = lines.length;
        for (let i = blockStart + 1; i < lines.length; i++) {
            if (lines[i].trim() === '') {
                blockEnd = i;
                break;
            }
        }
        const block = lines.slice(blockStart, blockEnd);
        lines.splice(blockEnd, 0, '', ...block);
        return { source: lines.join('\n'), detail: 'duplicated object ' + lines[blockStart].trim() };
    });
}

function caseFlipName(source) {
    return mutateSection(source, 'OBJECTS', function(body) {
        const lines = body.split('\n');
        let wantName = true;
        for (let i = 0; i < lines.length; i++) {
            const trimmed = lines[i].trim();
            if (trimmed === '') {
                wantName = true;
                continue;
            }
            if (wantName && /[A-Za-z]/.test(trimmed) && trimmed.toUpperCase() !== 'OBJECTS') {
                const flipped = trimmed === trimmed.toLowerCase() ? trimmed.toUpperCase() : trimmed.toLowerCase();
                if (flipped === trimmed) {
                    wantName = false;
                    continue;
                }
                lines[i] = lines[i].replace(trimmed, flipped);
                return { source: lines.join('\n'), detail: 'case-flipped ' + trimmed };
            }
            wantName = false;
        }
        return null;
    });
}

function layerDrop(source) {
    return mutateSection(source, 'COLLISIONLAYERS', function(body) {
        const lines = body.split('\n');
        for (let i = 0; i < lines.length; i++) {
            if (/,/.test(lines[i]) && /[A-Za-z]/.test(lines[i])) {
                const parts = lines[i].split(',').map(function(part) { return part.trim(); }).filter(Boolean);
                if (parts.length < 2) {
                    continue;
                }
                const dropped = parts.shift();
                lines[i] = parts.join(', ');
                return { source: lines.join('\n'), detail: 'dropped ' + dropped + ' from a collision layer' };
            }
        }
        return null;
    });
}

function layerDoubleBook(source) {
    return mutateSection(source, 'COLLISIONLAYERS', function(body) {
        const names = [];
        const lines = body.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const parts = lines[i].split(',');
            for (let j = 0; j < parts.length; j++) {
                const name = parts[j].trim();
                if (name && name.toUpperCase() !== 'COLLISIONLAYERS' && !/^=+$/.test(name)) {
                    names.push(name);
                }
            }
        }
        if (names.length === 0) {
            return null;
        }
        const name = names[names.length - 1];
        return {
            source: body.replace(/\s*$/, '') + '\n' + name + '\n',
            detail: 'duplicated ' + name + ' onto a second layer'
        };
    });
}

function backgroundAsAggregate(source) {
    return mutateSection(source, 'LEGEND', function(body) {
        return {
            source: body.replace(/\s*$/, '') + '\nBackground = Player and Wall\n',
            detail: 'redefined Background as an aggregate'
        };
    });
}

function soundOnProperty(source) {
    return mutateSection(source, 'SOUNDS', function(body) {
        return {
            source: body.replace(/\s*$/, '') + '\nObstacle MOVE 12345607\n',
            detail: 'MOVE sound on property Obstacle'
        };
    });
}

function winOnUndefined(source) {
    return mutateSection(source, 'WINCONDITIONS', function(body) {
        return {
            source: body.replace(/\s*$/, '') + '\nall Floop on Player\n',
            detail: 'win condition on undefined Floop'
        };
    });
}

function emptyCellRow(source) {
    return insertRuleLine(source, '[ > ] -> cancel');
}

function commandOnLhs(source) {
    return insertRuleLine(source, 'win [ Player ] -> [ Player ]');
}

function groupPlus(source, rng) {
    return mutateArrowRule(source, rng, function(line) {
        if (/^\s*\+/.test(line)) {
            return null;
        }
        return { line: '+ ' + line, detail: 'prefixed rule with +' };
    });
}

function startloopMismatch(source) {
    return insertRuleLine(source, 'startloop');
}

function directionPrefixSalad(source, rng) {
    return mutateArrowRule(source, rng, function(line) {
        if (/^\s*late\b/i.test(line)) {
            return { line: 'randomdir perpendicular ' + line, detail: 'prefixed randomdir perpendicular' };
        }
        return { line: 'late rigid randomdir perpendicular ' + line, detail: 'prefixed late rigid randomdir perpendicular' };
    });
}

function injectAgainLoop(source) {
    return insertRuleLine(source, '[ Player ] -> [ Player ] again');
}

function injectRandomFill(source) {
    return insertRuleLine(source, 'random [ no Player ] -> [ Player ] again');
}

function preludeInjection(source, rng) {
    const flag = PRELUDE_FLAGS[rng.integer(PRELUDE_FLAGS.length)];
    if (new RegExp('^\\s*' + flag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'im').test(source)) {
        const alt = PRELUDE_FLAGS[(PRELUDE_FLAGS.indexOf(flag) + 1) % PRELUDE_FLAGS.length];
        return { source: alt + '\n' + source, detail: 'injected ' + alt };
    }
    return { source: flag + '\n' + source, detail: 'injected ' + flag };
}

function raggedLevel(source) {
    return mutateSection(source, 'LEVELS', function(body) {
        const lines = body.split('\n');
        for (let i = 0; i < lines.length; i++) {
            if (i === 0 || /^\s*LEVELS\s*$/i.test(lines[i]) || /^\s*=+\s*$/.test(lines[i])) {
                continue;
            }
            if (lines[i].trim() === '' || /^\s*message\b/i.test(lines[i])) {
                continue;
            }
            lines[i] = lines[i] + '.';
            return { source: lines.join('\n'), detail: 'appended . to a level row' };
        }
        return null;
    });
}

function messageSandwich(source) {
    return mutateSection(source, 'LEVELS', function(body) {
        const lines = body.split('\n');
        for (let i = 0; i < lines.length; i++) {
            if (i === 0 || /^\s*LEVELS\s*$/i.test(lines[i]) || /^\s*=+\s*$/.test(lines[i])) {
                continue;
            }
            if (lines[i].trim() === '' || /^\s*message\b/i.test(lines[i])) {
                continue;
            }
            lines.splice(i, 0, 'message garden sandwich');
            return { source: lines.join('\n'), detail: 'inserted a message before a map' };
        }
        return null;
    });
}

function commentEatSection(source) {
    const section = findSection(source, 'LEGEND') || findSection(source, 'RULES');
    if (!section) {
        return null;
    }
    const lines = source.split('\n');
    lines[section.start] = '(' + lines[section.start];
    return {
        source: lines.join('\n'),
        detail: 'comment ate ' + section.name
    };
}

function duplicateSection(source, rng) {
    const found = [];
    for (let i = 0; i < SECTION_NAMES.length; i++) {
        const section = findSection(source, SECTION_NAMES[i]);
        if (section) {
            found.push(section);
        }
    }
    if (found.length === 0) {
        return null;
    }
    const section = found[rng.integer(found.length)];
    const lines = source.split('\n');
    const block = lines.slice(section.start, section.end);
    lines.splice(section.end, 0, ...block);
    return { source: lines.join('\n'), detail: 'duplicated section ' + section.name };
}

function nudgeInput(source, rng, fixture) {
    const extras = NUDGE_INPUT_CHOICES;
    const inputs = ((fixture && fixture.inputs) || []).slice();
    if (inputs.length === 0) {
        inputs.push(extras[rng.integer(extras.length)]);
    } else {
        const index = rng.integer(inputs.length);
        const current = inputs[index];
        const choices = extras.filter(function(choice) { return choice !== current; });
        inputs[index] = choices[rng.integer(choices.length)];
    }
    return {
        source: source,
        detail: 'nudged inputs to ' + JSON.stringify(inputs),
        inputs: inputs
    };
}

function offByOneLevel(source, rng, fixture) {
    const level = fixture && Number.isInteger(fixture.level) ? fixture.level : 0;
    const next = rng.integer(2) === 0 ? level + 1 : level - 1;
    return { source: source, detail: 'level ' + level + ' -> ' + next, level: next };
}

function seedPoison(source, rng) {
    const seed = POISON_SEEDS[rng.integer(POISON_SEEDS.length)];
    return { source: source, detail: 'poisoned engine seed', randomSeed: seed };
}

function prefixChop(source, rng, fixture) {
    const inputs = ((fixture && fixture.inputs) || []).slice();
    if (inputs.length < 2) {
        return null;
    }
    const keep = 1 + rng.integer(inputs.length - 1);
    return {
        source: source,
        detail: 'chopped inputs to length ' + keep,
        inputs: inputs.slice(0, keep)
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
    { name: 'unterminated-comment', apply: unterminatedComment },
    { name: 'duplicate-rule-line', apply: duplicateRuleLine },
    { name: 'swap-object-colors', apply: swapObjectColors },
    { name: 'nudge-level-cell', apply: nudgeLevelCell },
    { name: 'flip-win-quantifier', apply: flipWinQuantifier },
    { name: 'blns-slot', apply: blnsSlot },
    { name: 'keyword-as-name', apply: keywordAsName },
    { name: 'orphan-legend-member', apply: orphanLegendMember },
    { name: 'inject-ellipsis', apply: injectEllipsis },
    { name: 'inject-no', apply: injectNo },
    { name: 'inject-control-char', apply: injectControlChar },
    { name: 'sprite-matrix-noise', apply: spriteMatrixNoise },
    { name: 'duplicate-object-name', apply: duplicateObjectName },
    { name: 'case-flip-name', apply: caseFlipName },
    { name: 'layer-drop', apply: layerDrop },
    { name: 'layer-double-book', apply: layerDoubleBook },
    { name: 'background-as-aggregate', apply: backgroundAsAggregate },
    { name: 'sound-on-property', apply: soundOnProperty },
    { name: 'win-on-undefined', apply: winOnUndefined },
    { name: 'empty-cell-row', apply: emptyCellRow },
    { name: 'command-on-lhs', apply: commandOnLhs },
    { name: 'group-plus', apply: groupPlus },
    { name: 'startloop-mismatch', apply: startloopMismatch },
    { name: 'direction-prefix-salad', apply: directionPrefixSalad },
    { name: 'inject-again-loop', apply: injectAgainLoop },
    { name: 'inject-random-fill', apply: injectRandomFill },
    { name: 'prelude-injection', apply: preludeInjection },
    { name: 'ragged-level', apply: raggedLevel },
    { name: 'message-sandwich', apply: messageSandwich },
    { name: 'comment-eat-section', apply: commentEatSection },
    { name: 'duplicate-section', apply: duplicateSection },
    { name: 'nudge-input', apply: nudgeInput },
    { name: 'off-by-one-level', apply: offByOneLevel },
    { name: 'seed-poison', apply: seedPoison },
    { name: 'prefix-chop', apply: prefixChop }
];

function mutationChangedJob(applied, fixture) {
    if (!applied) {
        return false;
    }
    if (applied.source !== fixture.source) {
        return true;
    }
    if (applied.inputs && JSON.stringify(applied.inputs) !== JSON.stringify(fixture.inputs || [])) {
        return true;
    }
    if (applied.level !== undefined && applied.level !== fixture.level) {
        return true;
    }
    if (applied.randomSeed !== undefined && applied.randomSeed !== fixture.randomSeed) {
        return true;
    }
    return false;
}

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
        const applied = mutator.apply(fixture.source, rng, fixture);
        if (applied && mutationChangedJob(applied, fixture)) {
            return {
                mutator: mutator.name,
                fixtureName: fixture.name,
                fixtureIndex: fixture.fixtureIndex,
                corpusIndex: fixture.corpusIndex,
                kind: fixture.kind,
                source: applied.source,
                detail: applied.detail,
                inputs: applied.inputs || fixture.inputs,
                level: applied.level !== undefined ? applied.level : fixture.level,
                randomSeed: applied.randomSeed !== undefined ? applied.randomSeed : fixture.randomSeed,
                expectedOutput: fixture.expectedOutput,
                expectedErrors: fixture.expectedErrors,
                expectedErrorCount: fixture.expectedErrorCount,
                attempt: attempt
            };
        }
    }
    throw new Error('inapplicable mutation after ' + maxAttempts + ' attempts');
}

function isInapplicableMutation(error) {
    return Boolean(error && /inapplicable/.test(String(error.message)));
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
        extraInputs: 0,
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
                const raw = needValue(argv, i, 'seed');
                if (!/^\d+$/.test(raw)) {
                    throw new Error('seed must be a non-negative integer');
                }
                const value = Number(raw);
                if (!Number.isInteger(value) || value < 0 || value > 4294967295) {
                    throw new Error('seed must be a non-negative integer at most 4294967295');
                }
                result.seed = value;
                i++;
                break;
            }
            case '--count':
                result.count = needPositiveInt(argv, i, 'count');
                i++;
                break;
            case '--timeout-ms': {
                const value = needPositiveInt(argv, i, 'timeout-ms');
                if (value > 2147483647) {
                    throw new Error('timeout-ms must be a positive integer at most 2147483647');
                }
                result.timeoutMs = value;
                i++;
                break;
            }
            case '--fixture':
                result.fixture = needValue(argv, i, 'fixture');
                i++;
                break;
            case '--mutator': {
                const selected = needValue(argv, i, 'mutator').split(',').map(function(name) {
                    return name.trim();
                }).filter(Boolean);
                if (selected.length === 0) {
                    throw new Error('mutator list is empty');
                }
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
            case '--extra-inputs':
                result.extraInputs = needPositiveInt(argv, i, 'extra-inputs');
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

const EXTRA_INPUT_CHOICES = [0, 1, 2, 3, 4, 'tick'];

function extendInputs(recorded, rng, options) {
    const maxInputs = options && options.maxInputs ? options.maxInputs : recorded.length;
    const extraInputs = options && options.extraInputs ? options.extraInputs : 0;
    const prefix = (recorded || []).slice(0, maxInputs);
    const extras = [];
    for (let i = 0; i < extraInputs; i++) {
        extras.push(rng.pick(EXTRA_INPUT_CHOICES));
    }
    return prefix.concat(extras);
}

function prepareTrialInputs(recorded, rng, options) {
    const extraInputs = options && options.extraInputs ? options.extraInputs : 0;
    if (extraInputs === 0) {
        return (recorded || []).slice();
    }
    return extendInputs(recorded, rng, options);
}

function trialMaxInputs(options, inputs) {
    const extraInputs = options && options.extraInputs ? options.extraInputs : 0;
    if (extraInputs === 0) {
        return options.maxInputs;
    }
    return (inputs || []).length;
}

function clip(value, n) {
    return String(value == null ? '' : value).slice(0, n || 80);
}

function failureSignature(result) {
    if (!result || !result.kind) {
        return 'unknown';
    }
    if (result.kind === 'timeout') {
        return 'timeout';
    }
    if (result.kind === 'crash' && result.error) {
        const message = String(result.error.message || '').split('\n')[0];
        return 'crash:' + result.error.name + ':' + message;
    }
    if (result.kind === 'invariant' || result.kind === 'semantic-mismatch') {
        return result.kind + ':' + clip(result.detail) + ':' + clip(result.fingerprint);
    }
    return result.kind + ':' + clip(result.fingerprint) + ':' + clip(result.detail);
}

function isIntArrayLike(value, length) {
    return value && typeof value.length === 'number' && value.length === length;
}

function checkLevelInvariants(level, strideObj, strideMov, state) {
    if (!(strideObj > 0) || !(strideMov > 0) || !Number.isInteger(strideObj) || !Number.isInteger(strideMov)) {
        return 'strides are invalid';
    }
    if (!level || typeof level !== 'object') {
        return 'level is missing';
    }
    if (!Number.isInteger(level.width) || !Number.isInteger(level.height) || !(level.width > 0) || !(level.height > 0)) {
        return 'level dimensions are invalid';
    }
    if (level.n_tiles !== level.width * level.height) {
        return 'n_tiles does not match width*height';
    }
    const expectedObjects = level.n_tiles * strideObj;
    if (!isIntArrayLike(level.objects, expectedObjects)) {
        return 'objects length is ' + (level.objects && level.objects.length) + ' expected ' + expectedObjects;
    }
    const expectedMovements = level.n_tiles * strideMov;
    if (!isIntArrayLike(level.movements, expectedMovements)) {
        return 'movements length is ' + (level.movements && level.movements.length) + ' expected ' + expectedMovements;
    }
    if (level.commandQueue && level.commandQueue.length) {
        return 'commandQueue is not empty';
    }
    if (level.rowCellContents && level.rowCellContents.length !== level.height) {
        return 'rowCellContents length is invalid';
    }
    if (level.colCellContents && level.colCellContents.length !== level.width) {
        return 'colCellContents length is invalid';
    }
    if (state && state.rigid) {
        if (!level.rigidMovementAppliedMask || level.rigidMovementAppliedMask.length !== level.n_tiles) {
            return 'rigidMovementAppliedMask length is invalid';
        }
        if (!level.rigidGroupIndexMask || level.rigidGroupIndexMask.length !== level.n_tiles) {
            return 'rigidGroupIndexMask length is invalid';
        }
    }
    if (state && state.idDict) {
        const objectCount = Object.keys(state.idDict).length;
        const maxBit = objectCount;
        for (let tile = 0; tile < level.n_tiles; tile++) {
            for (let word = 0; word < strideObj; word++) {
                const bits = level.objects[tile * strideObj + word];
                for (let bit = 0; bit < 32; bit++) {
                    const abs = word * 32 + bit;
                    if (abs >= maxBit && (bits & (1 << bit))) {
                        return 'object bit ' + abs + ' is set but idDict has ' + maxBit + ' entries';
                    }
                }
            }
        }
    }
    const layerMasks = state && (state.layerMasks || state.collisionMasks);
    if (layerMasks && layerMasks.length) {
        for (let tile = 0; tile < level.n_tiles; tile++) {
            for (let layer = 0; layer < layerMasks.length; layer++) {
                const mask = layerMasks[layer];
                if (!mask || !mask.data) {
                    continue;
                }
                let count = 0;
                for (let word = 0; word < strideObj; word++) {
                    const bits = level.objects[tile * strideObj + word] & mask.data[word];
                    if (bits) {
                        count += (bits >>> 0).toString(2).replace(/0/g, '').length;
                    }
                }
                if (count > 1) {
                    return 'collision layer ' + layer + ' has ' + count + ' objects at tile ' + tile;
                }
            }
        }
    }
    return null;
}

function isInteresting(result) {
    return result.kind === 'crash'
        || result.kind === 'timeout'
        || result.kind === 'invariant'
        || result.kind === 'nondeterministic'
        || result.kind === 'replay-divergence'
        || result.kind === 'semantic-mismatch';
}

function baselineOracleFields(fixture, options) {
    if (fixture.kind === 'compiler-message') {
        return {
            expectedErrors: fixture.expectedErrors,
            expectedErrorCount: fixture.expectedErrorCount
        };
    }
    if (options && options.extraInputs === 0
        && fixture.inputs && fixture.inputs.length <= options.maxInputs
        && typeof fixture.expectedOutput === 'string') {
        return { expectedOutput: fixture.expectedOutput };
    }
    return {};
}

function isHealthyKind(kind) {
    return kind === 'ok' || kind === 'compiler-error' || kind === 'compiler-warning';
}

function attributeMonster(baseline, mutantResult) {
    if (!baseline || isHealthyKind(baseline.kind)) {
        return { save: isInteresting(mutantResult), tally: mutantResult.kind, baseline: false };
    }
    if (failureSignature(baseline) === failureSignature(mutantResult)) {
        return { save: false, tally: 'baseline', baseline: true };
    }
    if (!isInteresting(mutantResult)) {
        return { save: false, tally: 'baseline', baseline: true };
    }
    return { save: true, tally: mutantResult.kind, baseline: true };
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

async function shrinkInteresting(mutant, originalResult, options) {
    const signature = failureSignature(originalResult);
    if (!options.shrink || originalResult.kind === 'timeout') {
        return { source: mutant.source, steps: 0, signature: signature, result: originalResult };
    }
    let current = mutant.source.split('\n');
    let steps = 0;
    let remaining = options.shrinkBudget;
    let changed = true;
    while (changed && remaining > 0) {
        changed = false;
        let i = 0;
        while (i < current.length && remaining > 0) {
            if (i === current.length - 1 && current[i] === '') {
                break;
            }
            let candidateSource = current.slice(0, i).concat(current.slice(i + 1)).join('\n');
            if (mutant.source.endsWith('\n') && !candidateSource.endsWith('\n')) {
                candidateSource += '\n';
            }
            remaining--;
            steps++;
            const next = await options.evaluate({
                source: candidateSource,
                inputs: mutant.inputs,
                level: mutant.level,
                randomSeed: mutant.randomSeed
            });
            if (failureSignature(next) === signature) {
                current = candidateSource.split('\n');
                changed = true;
            } else {
                i++;
            }
        }
    }
    const minimizedSource = current.join('\n');
    const verified = await options.evaluate({
        source: minimizedSource,
        inputs: mutant.inputs,
        level: mutant.level,
        randomSeed: mutant.randomSeed
    });
    steps++;
    if (failureSignature(verified) !== signature) {
        return { source: mutant.source, steps: steps, signature: signature, result: originalResult };
    }
    return { source: minimizedSource, steps: steps, signature: signature, result: verified };
}

function artifactDirName(signature, seed, index) {
    const safe = String(signature)
        .replace(/[^A-Za-z0-9._-]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .slice(0, 80);
    return (safe || 'monster') + '-s' + seed + '_' + String(index).padStart(4, '0');
}

function formatRegression(name, source, job) {
    job = job || {};
    const inputs = job.inputs || [];
    const level = job.level == null ? 0 : job.level;
    const seed = job.randomSeed == null ? null : job.randomSeed;
    return '[\n    ' + JSON.stringify(name) + ',\n    [' +
        JSON.stringify(source) + ', ' + JSON.stringify(inputs) + ', "", ' +
        JSON.stringify(level) + ', ' + JSON.stringify(seed) + ']\n],\n';
}

const KNOWN_RESULT_KINDS = [
    'ok', 'compiler-error', 'compiler-warning', 'crash',
    'invariant', 'nondeterministic', 'replay-divergence', 'semantic-mismatch'
];

function decodeBuffers(chunks) {
    if (!chunks.length) {
        return '';
    }
    return Buffer.concat(chunks).toString('utf8');
}

function runChild(command, args, stdin, timeoutMs) {
    return new Promise(function(resolve) {
        const child = spawn(command, args, { stdio: ['pipe', 'pipe', 'pipe'] });
        const stdoutChunks = [];
        const stderrChunks = [];
        let timedOut = false;
        let settled = false;
        const timer = setTimeout(function() {
            if (child.exitCode !== null || child.signalCode !== null) {
                return;
            }
            timedOut = true;
            child.kill('SIGKILL');
        }, timeoutMs);

        function finish(result) {
            if (settled) {
                return;
            }
            settled = true;
            clearTimeout(timer);
            resolve(result);
        }

        function crashResult(name, message, detail) {
            return {
                kind: 'crash',
                error: { name: name, message: String(message || '').split('\n')[0] },
                fingerprint: '',
                detail: detail || '',
                errorCount: 0
            };
        }

        function crashFromError(error) {
            if (timedOut) {
                finish({
                    kind: 'timeout',
                    error: null,
                    fingerprint: '',
                    detail: 'timeout',
                    errorCount: 0
                });
                return;
            }
            const message = error && error.message ? error.message : 'spawn failed';
            finish(crashResult((error && error.name) || 'ChildError', message, decodeBuffers(stderrChunks) || message));
        }

        child.stdout.on('data', function(chunk) { stdoutChunks.push(Buffer.from(chunk)); });
        child.stderr.on('data', function(chunk) { stderrChunks.push(Buffer.from(chunk)); });
        child.on('error', crashFromError);
        child.stdin.on('error', crashFromError);
        child.on('close', function(code, signal) {
            const stdout = decodeBuffers(stdoutChunks);
            const stderr = decodeBuffers(stderrChunks);
            const lines = stdout.split('\n').map(function(line) { return line.trim(); }).filter(Boolean);
            const hadOutput = lines.length > 0;
            let parsed;
            let parseError;
            if (hadOutput) {
                try {
                    parsed = JSON.parse(lines.pop());
                } catch (error) {
                    parseError = error;
                }
            }
            const validParsedResult = Boolean(
                parsed
                && typeof parsed === 'object'
                && !Array.isArray(parsed)
                && KNOWN_RESULT_KINDS.indexOf(parsed.kind) >= 0
            );

            if (validParsedResult && code === 0) {
                finish(parsed);
                return;
            }
            if (timedOut) {
                finish({
                    kind: 'timeout',
                    error: null,
                    fingerprint: '',
                    detail: 'timeout',
                    errorCount: 0
                });
                return;
            }
            if (!hadOutput) {
                finish(crashResult('ChildOutputError', 'empty worker stdout', stderr));
                return;
            }
            if (parseError) {
                finish(crashResult('ChildOutputError', (stdout || stderr || parseError.message).split('\n')[0], stderr));
                return;
            }
            if (!validParsedResult) {
                finish(crashResult('ChildOutputError', 'invalid worker result', stderr));
                return;
            }
            if (code && code !== 0 && parsed.kind !== 'crash') {
                finish(crashResult('ChildExitError', 'worker exited ' + code, stderr));
                return;
            }
            if (signal && signal !== 'SIGKILL') {
                finish(crashResult('ChildExitError', 'worker signal ' + signal, stderr));
                return;
            }
            finish(parsed);
        });
        try {
            child.stdin.write(stdin);
            child.stdin.end();
        } catch (error) {
            crashFromError(error);
        }
    });
}

function writeArtifacts(outputDir, dirName, files) {
    fs.mkdirSync(outputDir, { recursive: true });
    const tmp = fs.mkdtempSync(path.join(outputDir, '.' + dirName + '-'));
    const dest = path.join(outputDir, dirName);
    let bak = null;
    try {
        const names = Object.keys(files);
        for (let i = 0; i < names.length; i++) {
            fs.writeFileSync(path.join(tmp, names[i]), files[names[i]]);
        }
        try {
            fs.renameSync(tmp, dest);
        } catch (error) {
            if (error.code !== 'ENOTEMPTY' && error.code !== 'EEXIST') {
                throw error;
            }
            bak = fs.mkdtempSync(path.join(outputDir, '.' + dirName + '-old-'));
            fs.renameSync(dest, bak);
            try {
                fs.renameSync(tmp, dest);
            } catch (renameError) {
                try {
                    fs.renameSync(bak, dest);
                    bak = null;
                } catch (restoreError) {
                    // Leave bak in place if dest cannot be restored.
                }
                throw renameError;
            }
            fs.rmSync(bak, { recursive: true, force: true });
            bak = null;
        }
        return dest;
    } finally {
        if (fs.existsSync(tmp)) {
            fs.rmSync(tmp, { recursive: true, force: true });
        }
        if (bak && fs.existsSync(bak) && fs.existsSync(dest)) {
            fs.rmSync(bak, { recursive: true, force: true });
        }
    }
}

module.exports = {
    Random: Random,
    loadCorpus: loadCorpus,
    mutators: mutators,
    mutateFixture: mutateFixture,
    isInapplicableMutation: isInapplicableMutation,
    parseArguments: parseArguments,
    extendInputs: extendInputs,
    prepareTrialInputs: prepareTrialInputs,
    trialMaxInputs: trialMaxInputs,
    failureSignature: failureSignature,
    checkLevelInvariants: checkLevelInvariants,
    isInteresting: isInteresting,
    baselineOracleFields: baselineOracleFields,
    isHealthyKind: isHealthyKind,
    attributeMonster: attributeMonster,
    shrinkSource: shrinkSource,
    shrinkInteresting: shrinkInteresting,
    artifactDirName: artifactDirName,
    formatRegression: formatRegression,
    writeArtifacts: writeArtifacts,
    runChild: runChild,
    KNOWN_RESULT_KINDS: KNOWN_RESULT_KINDS
};
