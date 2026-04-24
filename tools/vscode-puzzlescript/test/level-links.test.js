#!/usr/bin/env node
'use strict';

const assert = require('assert');
const Module = require('module');

const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
    if (request === 'vscode') {
        return {
        SemanticTokensLegend: class {},
        Position: class {
            constructor(line, character) {
                this.line = line;
                this.character = character;
            }
        },
        Range: class {
            constructor(start, end) {
                this.start = start;
                this.end = end;
            }
        },
        DocumentLink: class {
            constructor(range, target) {
                this.range = range;
                this.target = target;
            }
        },
        Uri: { parse(value) { return { value }; } },
        };
    }
    return originalLoad.apply(this, arguments);
};

const { levelRows } = require('../src/extension');

const source = [
    'title Links',
    '',
    'levels',
    'P..',
    '.*.',
    '',
    'message hello',
    '',
    '..P',
    '..*',
    '',
    'rules',
    '[ Player ] -> [ Player ]',
].join('\n');

const rows = levelRows(source);
assert.deepStrictEqual(
    rows.map(row => [row.line, row.level]),
    [
        [3, 0],
        [4, 0],
        [8, 2],
        [9, 2],
    ]
);
assert.strictEqual(rows[0].start, 0);
assert.strictEqual(rows[0].end, 3);

console.log('level link tests passed');
